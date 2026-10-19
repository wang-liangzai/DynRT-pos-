"""Trainable temporal attention and cross-modal fusion on frozen encoders."""
import random
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler,normalize

MODALITIES=('text','context_asr','audio','frames')
MAX_FRAMES=12

class Preprocessor:
    def fit(self,records,indices):
        self.scalers={}
        for name in MODALITIES:
            rows=[records[i][name] for i in indices]
            x=np.concatenate(rows) if name=='frames' else np.stack(rows)
            self.scalers[name]=StandardScaler().fit(x)
        return self
    def transform(self,records,indices):
        batch={name:[] for name in MODALITIES}; masks=[]; available=[]
        for i in indices:
            r=records[i]
            for name in MODALITIES:
                value=np.atleast_2d(r[name]); value=normalize(self.scalers[name].transform(value)).astype(np.float32)
                if name=='frames':
                    selected=np.linspace(0,len(value)-1,min(MAX_FRAMES,len(value))).astype(int)
                    frame=np.zeros((MAX_FRAMES,value.shape[-1]),dtype=np.float32); frame[:len(selected)]=value[selected]
                    mask=np.arange(MAX_FRAMES)>=len(selected); masks.append(mask); value=frame
                else: value=value[0]
                batch[name].append(value)
            available.append(float(r['context_available']))
        result={name:torch.from_numpy(np.stack(values)) for name,values in batch.items()}
        result['frame_mask']=torch.from_numpy(np.stack(masks)); result['context_available']=torch.tensor(available,dtype=torch.float32)
        return result

class TemporalFusion(nn.Module):
    def __init__(self,dims,dropout=.3,hidden=64):
        super().__init__()
        self.projections=nn.ModuleDict({m:nn.Sequential(nn.Linear(dims[m],hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(dropout)) for m in MODALITIES})
        self.position=nn.Parameter(torch.randn(1,MAX_FRAMES,hidden)*.02)
        self.temporal_score=nn.Linear(hidden,1)
        self.cross_attention=nn.MultiheadAttention(hidden,4,dropout=dropout,batch_first=True)
        self.norm=nn.LayerNorm(hidden)
        self.gate=nn.Linear(hidden,1)
        self.classifier=nn.Sequential(nn.Linear(hidden*3,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,2))
    def forward(self,batch):
        text=self.projections['text'](batch['text'])
        context=self.projections['context_asr'](batch['context_asr'])
        context=context*batch['context_available'][:,None]
        audio=self.projections['audio'](batch['audio'])
        frames=self.projections['frames'](batch['frames'])+self.position
        scores=self.temporal_score(frames).squeeze(-1).masked_fill(batch['frame_mask'],-1e4)
        video=(scores.softmax(-1).unsqueeze(-1)*frames).sum(1)
        tokens=torch.stack((text,context,audio,video),dim=1)
        if self.training:
            keep=(torch.rand(tokens.shape[:2],device=tokens.device)>.15).float()
            keep[:,0]=1 # retain target sentence, randomly drop other modalities
            tokens=tokens*keep[:,:,None]
        padding=torch.zeros(tokens.shape[:2],dtype=torch.bool,device=tokens.device)
        padding[:,1]=batch['context_available']==0
        cross,_=self.cross_attention(tokens,tokens,tokens,key_padding_mask=padding,need_weights=False)
        fused=self.norm(tokens+cross)
        gate=self.gate(fused).squeeze(-1).masked_fill(padding,-1e4).softmax(-1)
        pooled=(fused*gate[:,:,None]).sum(1)
        difference=torch.abs(text-context)*batch['context_available'][:,None]
        interaction=text*context
        return self.classifier(torch.cat((pooled,difference,interaction),dim=-1))

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def subset(batch,indices,device):
    return {k:v[indices].to(device) for k,v in batch.items()}

@torch.inference_mode()
def probabilities(model,batch,device):
    model.eval(); result=[]
    for start in range(0,len(batch['text']),128):
        result.append(model(subset(batch,slice(start,start+128),device)).softmax(-1)[:,1].cpu().numpy())
    return np.concatenate(result)

def fit_network(records,y,train,valid=None,dropout=.3,epochs=50,seed=20261018):
    from sklearn.metrics import f1_score
    seed_all(seed); torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prep=Preprocessor().fit(records,train); data=prep.transform(records,train)
    validation=prep.transform(records,valid) if valid is not None else None
    dims={m:records[0][m].shape[-1] for m in MODALITIES}
    model=TemporalFusion(dims,dropout).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=.02)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    counts=np.bincount(y[train],minlength=2)
    weights=torch.tensor(len(train)/(2*np.maximum(counts,1)),dtype=torch.float32,device=device)
    loss_fn=nn.CrossEntropyLoss(weight=weights,label_smoothing=.05)
    target=torch.tensor(y[train],dtype=torch.long)
    best=-1; best_state=None; best_epoch=epochs; patience=0
    for epoch in range(1,epochs+1):
        model.train()
        for indices in torch.randperm(len(train)).split(64):
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(subset(data,indices,device)),target[indices].to(device))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        if valid is not None:
            score=f1_score(y[valid],probabilities(model,validation,device)>=.5,average='macro',zero_division=0)
            if score>best:
                best=score; best_epoch=epoch; patience=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else: patience+=1
            if epoch>=15 and patience>=10: break
    if best_state is not None: model.load_state_dict(best_state)
    return model,prep,dict(epochs=best_epoch,dropout=dropout,dimensions=dims,validation_macro_f1=best)

def bundle_scores(bundle,records,indices):
    """Label-free runtime shared by training evaluation and external inference."""
    if bundle['kind']=='stack':
        values=np.stack([bundle_scores(b,records,indices) for b in bundle['branches']],axis=1)
        return bundle['meta'].decision_function(values)
    if bundle['kind']=='svm':
        x=np.concatenate([bundle['scalers'][m].transform(np.stack([records[i][m] for i in indices])) for m in bundle['modalities']],axis=1)
        return bundle['model'].decision_function(x)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=TemporalFusion(bundle['info']['dimensions'],bundle['info']['dropout']).to(device)
    model.load_state_dict(bundle['state'])
    return probabilities(model,bundle['preprocessor'].transform(records,indices),device)-.5
