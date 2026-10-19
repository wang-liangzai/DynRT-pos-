"""Five outer speaker-group folds; inner group CV chooses models, never outer labels.

All 690 rows receive exactly one out-of-fold prediction. Candidate grid is frozen
in protocol.json before evaluation. No gold transcripts are loaded as features.
"""
import json,time,hashlib
import joblib
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from features import ROOT,location,save_json
from train import metrics
from fusion_v2 import fit_network,probabilities,bundle_scores

OUT=ROOT/'artifacts_v2'
SEED=20261018
CANDIDATES=[dict(name='baseline',kind='svm',context=False,C=1,kernel='linear')]+[dict(name=f'context_{kernel}_{c}',kind='svm',context=True,C=c,kernel=kernel) for kernel,c in [('linear',.1),('linear',1),('linear',10),('rbf',1),('rbf',10)]]+[dict(name=f'attention_{d}',kind='attention',dropout=d) for d in (.3,.5)]

def summary(y,p):
    result=metrics(y,p); tn,fp,fn,tp=np.array(result['confusion_matrix']).ravel()
    result.update(non_sarcasm_recall=float(tn/max(1,tn+fp)),false_positive_rate=float(fp/max(1,tn+fp)),balanced_accuracy=balanced_accuracy_score(y,p),predicted_non_sarcastic=int(sum(p==0)),predicted_sarcastic=int(sum(p==1)))
    return result

def svm_fit(records,y,indices,candidate):
    mods=['text','vision','motion','audio']+(['context_asr'] if candidate['context'] else [])
    scalers={}; blocks=[]
    for m in mods:
        x=np.stack([records[i][m] for i in indices]); scaler=make_pipeline(StandardScaler(),Normalizer()).fit(x)
        scalers[m]=scaler; blocks.append(scaler.transform(x))
    model=SVC(C=candidate['C'],kernel=candidate['kernel'],class_weight='balanced')
    model.fit(np.concatenate(blocks,axis=1),y[indices])
    return dict(kind='svm',model=model,scalers=scalers,modalities=mods)

def fit_candidate(records,y,indices,candidate,epochs=None):
    if candidate['kind']=='svm': return svm_fit(records,y,indices,candidate)
    model,prep,info=fit_network(records,y,indices,dropout=candidate['dropout'],epochs=epochs or 25,seed=SEED)
    return dict(kind='attention',state={k:v.cpu() for k,v in model.state_dict().items()},preprocessor=prep,info=info)

def main():
    OUT.mkdir(exist_ok=True)
    data=json.loads((location('dataset')/'sarcasm_data.json').read_text(encoding='utf-8'))
    keys=sorted(data); y=np.array([int(data[k]['sarcasm']) for k in keys]); groups=np.array([data[k]['show']+'::'+data[k]['speaker'] for k in keys])
    records=[]
    for key in keys:
        with np.load(ROOT/'cache_v2'/(key+'.npz')) as z: records.append({m:z[m] for m in z.files})
    outer=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(keys,y,groups))
    protocol=dict(seed=SEED,candidates=CANDIDATES,selection='mean inner 3-fold macro F1; default class threshold; no outer test tuning',input='actual ASR only, including context video ASR; no reference text',outer_folds=[dict(train=[keys[i] for i in tr],test=[keys[i] for i in te]) for tr,te in outer],dataset_sha256=hashlib.sha256((location('dataset')/'sarcasm_data.json').read_bytes()).hexdigest())
    if (OUT/'protocol.json').exists():
        previous=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
        if previous!=protocol: raise RuntimeError('Protocol changed. Use a new output directory instead of reusing completed folds.')
    save_json(OUT/'protocol.json',protocol)
    seen=np.zeros(len(keys),dtype=int); all_rows=[]; fold_reports=[]; selection_scores={c['name']:[] for c in CANDIDATES}; epoch_counts={c['name']:[] for c in CANDIDATES}
    started=time.monotonic()
    for fold,(dev,test) in enumerate(outer):
        assert not set(groups[dev])&set(groups[test])
        fold_path=OUT/f'fold_{fold+1}.json'
        if fold_path.exists():
            saved=json.loads(fold_path.read_text(encoding='utf-8')); fold_reports.append(saved); all_rows.extend(saved['predictions']); seen[test]+=1
            for item in saved['selection']:
                selection_scores[item['candidate']['name']].append(item['mean_macro_f1']); epoch_counts[item['candidate']['name']].extend(item['epochs'])
            continue
        inner=list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED+fold+1).split(dev,y[dev],groups[dev]))
        selection=[]
        for candidate in CANDIDATES:
            scores=[]; epochs=[]
            for inner_index,(a,b) in enumerate(inner):
                train,valid=dev[a],dev[b]
                assert not set(groups[train])&set(groups[valid])
                if candidate['kind']=='svm':
                    bundle=svm_fit(records,y,train,candidate); pred=bundle_scores(bundle,records,valid)>0
                else:
                    model,prep,info=fit_network(records,y,train,valid,dropout=candidate['dropout'],epochs=50,seed=SEED+inner_index)
                    pred=probabilities(model,prep.transform(records,valid),next(model.parameters()).device)>.5
                    epochs.append(info['epochs'])
                scores.append(metrics(y[valid],pred)['macro_f1'])
            item=dict(candidate=candidate,mean_macro_f1=float(np.mean(scores)),inner_macro_f1=scores,epochs=epochs)
            selection.append(item); selection_scores[candidate['name']].append(item['mean_macro_f1']); epoch_counts[candidate['name']].extend(epochs)
            print(f'fold {fold+1}/5 candidate {candidate["name"]} inner_macro_f1={item["mean_macro_f1"]:.4f} elapsed={time.monotonic()-started:.1f}s',flush=True)
        winner=max(selection,key=lambda row:row['mean_macro_f1'])
        # Persist inner-only selection before computing any outer-test predictions.
        save_json(OUT/f'fold_{fold+1}_selection.json',dict(selection=selection,winner=winner))
        chosen=fit_candidate(records,y,dev,winner['candidate'],int(np.median(winner['epochs'])) if winner['epochs'] else None)
        chosen.update(train_ids=[keys[i] for i in dev],candidate=winner['candidate'],protocol='outer-group-held-out')
        joblib.dump(chosen,OUT/f'fold_{fold+1}.joblib')
        scores=bundle_scores(chosen,records,test); pred=(scores>0).astype(int)
        base=svm_fit(records,y,dev,CANDIDATES[0]); baseline=(bundle_scores(base,records,test)>0).astype(int)
        rows=[dict(id=keys[i],fold=fold+1,expected=int(y[i]),predicted=int(p),baseline_predicted=int(b),score=float(s),show=data[keys[i]]['show']) for i,p,b,s in zip(test,pred,baseline,scores)]
        report=dict(fold=fold+1,winner=winner['candidate'],selection=selection,metrics=summary(y[test],pred),baseline=summary(y[test],baseline),predictions=rows)
        save_json(fold_path,report); fold_reports.append(report); all_rows.extend(rows); seen[test]+=1
        print(f'fold {fold+1} saved; no candidate changes based on outer scores',flush=True)
    assert np.all(seen==1) and len(all_rows)==len(keys)
    all_rows.sort(key=lambda r:r['id']); truth=np.array([r['expected'] for r in all_rows]); pred=np.array([r['predicted'] for r in all_rows]); baseline=np.array([r['baseline_predicted'] for r in all_rows])
    result=dict(count=len(keys),protocol='five outer speaker-group folds, three inner group folds; every row predicted once without its speaker in training; exploratory reused dataset',metrics=summary(truth,pred),baseline=summary(truth,baseline),folds=[{k:v for k,v in f.items() if k not in ('selection','predictions')} for f in fold_reports],by_show={},predictions=all_rows)
    for show in sorted({r['show'] for r in all_rows}):
        ix=np.array([r['show']==show for r in all_rows]); result['by_show'][show]=summary(truth[ix],pred[ix])
    usable=np.array([not json.loads((ROOT/'cache'/(r['id']+'.json')).read_text(encoding='utf-8'))['asr_empty'] for r in all_rows])
    result['nonempty_asr']=dict(coverage=float(usable.mean()),metrics=summary(truth[usable],pred[usable]))
    save_json(OUT/'report.json',result)
    # A deployment model may use all labels, but its training accuracy is never reported as generalization.
    choice=max(CANDIDATES,key=lambda c:np.mean(selection_scores[c['name']]))
    epochs=epoch_counts[choice['name']]
    final=fit_candidate(records,y,np.arange(len(keys)),choice,int(np.median(epochs)) if epochs else None)
    final.update(candidate=choice,train_ids=keys,protocol='all-data deployment fit; do not evaluate on these 690 as unseen data',selection='mean inner validation scores across outer folds, not outer scores')
    joblib.dump(final,OUT/'deployment.joblib')
    save_json(OUT/'deployment_selection.json',dict(candidate=choice,mean_inner_scores={k:float(np.mean(v)) for k,v in selection_scores.items()}))
    print(json.dumps({k:v for k,v in result.items() if k not in ('predictions','folds')},indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
