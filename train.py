"""Speaker-disjoint train/validation/test, validation-only model selection."""
import json,hashlib,platform,argparse
import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix
from features import ROOT,location,CONFIG,save_json

def metrics(y,p):
    return dict(n=len(y),accuracy=accuracy_score(y,p),macro_f1=f1_score(y,p,average='macro',zero_division=0),sarcasm_f1=f1_score(y,p,zero_division=0),sarcasm_recall=recall_score(y,p,zero_division=0),sarcasm_precision=precision_score(y,p,zero_division=0),confusion_matrix=confusion_matrix(y,p,labels=[0,1]).tolist())

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--split-only',action='store_true'); args=parser.parse_args()
    data=json.loads((location('dataset')/'sarcasm_data.json').read_text(encoding='utf-8'))
    keys=sorted(data); y=np.array([int(data[k]['sarcasm']) for k in keys])
    groups=np.array([data[k]['show']+'::'+data[k]['speaker'] for k in keys])
    # First deterministic partition with sufficient examples in each class.
    splitpath=ROOT/'artifacts/split.json'
    if splitpath.exists():
        split=json.loads(splitpath.read_text(encoding='utf-8'))
        tr,va,te=[np.array([keys.index(k) for k in split[s]]) for s in ('train','validation','test')]
    else:
        for seed in range(CONFIG['seed'],CONFIG['seed']+100):
            dev,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(keys,y,groups))
            a,b=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+1).split(dev,y[dev],groups[dev]))
            tr,va=dev[a],dev[b]
            if all(np.bincount(y[ix],minlength=2).min()>=15 for ix in (tr,va,te)): break
        else: raise RuntimeError('No eligible split')
        split={s:[keys[i] for i in ix] for s,ix in [('train',tr),('validation',va),('test',te)]}
        split.update(seed=seed,protocol='speaker disjoint by show::speaker; not official benchmark split')
        save_json(splitpath,split)
    assert not(set(groups[tr])&set(groups[va]) or set(groups[tr])&set(groups[te]) or set(groups[va])&set(groups[te]))
    assert len(set(tr)|set(va)|set(te))==len(keys)
    if args.split_only:
        print(json.dumps({s:dict(count=len(ix),labels=np.bincount(y[ix],minlength=2).tolist(),speakers=sorted(set(groups[ix]))) for s,ix in [('train',tr),('validation',va),('test',te)]},indent=2)); return
    blocks={name:[] for name in ['text','reference','context','vision','motion','audio']}
    for k in keys:
        with np.load(ROOT/'cache'/(k+'.npz')) as z:
            for name in blocks: blocks[name].append(z[name])
    blocks={name:np.stack(value) for name,value in blocks.items()}
    scalers={}; scaled={}
    for name,x in blocks.items():
        # Fit normalization on TRAIN ONLY. Each modality has equal vector norm.
        scaler=make_pipeline(StandardScaler(),Normalizer())
        scaler.fit(x[tr]); scaled[name]=scaler.transform(x); scalers[name]=scaler
    variants={'text':['text'],'image_text':['text','vision'],'video':['text','vision','motion','audio'],'reference_video':['reference','vision','motion','audio'],'reference_context_video':['reference','context','vision','motion','audio']}
    selected={}; validation=[]
    for name,mods in variants.items():
        x=np.concatenate([scaled[m] for m in mods],axis=1)
        best=None
        for kernel in ('linear','rbf'):
            for c in (.01,.1,1,10,100):
                model=SVC(C=c,kernel=kernel,class_weight='balanced',random_state=CONFIG['seed'])
                model.fit(x[tr],y[tr]); m=metrics(y[va],model.predict(x[va]))
                candidate=dict(variant=name,kernel=kernel,C=c,validation=m)
                validation.append(candidate)
                if best is None or m['macro_f1']>best['validation']['macro_f1']:
                    best=dict(candidate,model=model,modalities=mods)
        selected[name]=best
        print(name,'validation',best['validation'],flush=True)
    # Freeze selection before computing test performance; no test-based tuning.
    save_json(ROOT/'artifacts/selection.json',dict(candidates=validation,selected={k:{a:b for a,b in v.items() if a!='model'} for k,v in selected.items()}))
    report={'protocol':split['protocol'],'split_counts':{s:len(split[s]) for s in ('train','validation','test')},'results':{}}
    original=json.loads((location('reference_repo')/'deployment/mustard_evaluation/report.json').read_text(encoding='utf-8'))
    old={r['id']:r['predicted'] for r in original['results']}
    report['original_dynrt_reference_midframe']=metrics(y[te],np.array([old[keys[i]] for i in te]))
    report['majority_baseline']=metrics(y[te],np.full(len(te),int(y[tr].mean()>=.5)))
    for name,best in selected.items():
        mods=best['modalities']; x=np.concatenate([scaled[m] for m in mods],axis=1)
        pred=best['model'].predict(x[te]); margin=best['model'].decision_function(x[te])
        report['results'][name]=dict(validation=best['validation'],test=metrics(y[te],pred),predictions=[dict(id=keys[i],expected=int(y[i]),predicted=int(p),margin=float(s)) for i,p,s in zip(te,pred,margin)])
        joblib.dump(dict(model=best['model'],scalers={m:scalers[m] for m in mods},modalities=mods,language='en',feature_version=1,selection='validation macro F1; test not used',train_ids=split['train']),ROOT/'artifacts'/(name+'.joblib'))
    report['environment']={'python':platform.python_version(),'data_sha256':hashlib.sha256((location('dataset')/'sarcasm_data.json').read_bytes()).hexdigest()}
    save_json(ROOT/'artifacts/report.json',report)
    print(json.dumps({k:v['test'] for k,v in report['results'].items()},indent=2),flush=True)
if __name__=='__main__': main()
