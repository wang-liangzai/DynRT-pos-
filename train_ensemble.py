"""Stacking with group-held-out meta-training scores, never in-sample scores.

Exploratory follow-up on the same fixed outer folds. Outer labels are only scored
after all base models, meta-model regularization and weights have been fitted.
"""
import json,time
import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from features import ROOT,location,save_json
from fusion_v2 import fit_network,probabilities,bundle_scores
from train_v2 import CANDIDATES,SEED,svm_fit,fit_candidate,summary

OUT=ROOT/'artifacts_ensemble'

def select_branches(selection):
    baseline=CANDIDATES[0]
    context=max([r for r in selection if r['candidate']['name'].startswith('context_')],key=lambda r:r['mean_macro_f1'])['candidate']
    attention=max([r for r in selection if r['candidate']['kind']=='attention'],key=lambda r:r['mean_macro_f1'])['candidate']
    return [baseline,context,attention]

def fit_stack(records,y,groups,dev,candidates,seed):
    splits=list(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(dev,y[dev],groups[dev]))
    scores=np.zeros((len(dev),len(candidates))); seen=np.zeros(len(dev),dtype=int); epochs={c['name']:[] for c in candidates}
    audits=[]
    for split_index,(a,b) in enumerate(splits):
        tr,va=dev[a],dev[b]
        assert not set(groups[tr])&set(groups[va])
        for j,c in enumerate(candidates):
            if c['kind']=='svm':
                model=svm_fit(records,y,tr,c); scores[b,j]=bundle_scores(model,records,va)
            else:
                # Early stopping uses an extra group holdout from this base model's
                # training rows, never the rows used to generate meta-training scores.
                fit,stop=next(StratifiedGroupKFold(3,shuffle=True,random_state=seed+split_index+20).split(tr,y[tr],groups[tr]))
                _,_,info=fit_network(records,y,tr[fit],tr[stop],dropout=c['dropout'],epochs=50,seed=SEED+split_index)
                epochs[c['name']].append(info['epochs'])
                model=fit_candidate(records,y,tr,c,info['epochs'])
                scores[b,j]=bundle_scores(model,records,va)
        seen[b]+=1; audits.append(dict(train=tr.tolist(),validation=va.tolist()))
    assert np.all(seen==1)
    choices=[]
    for c in (.01,.1,1,10):
        values=[]
        for a,b in splits:
            meta=make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight='balanced',max_iter=1000))
            meta.fit(scores[a],y[dev[a]]); values.append(summary(y[dev[b]],meta.predict(scores[b]))['macro_f1'])
        choices.append(dict(C=c,mean_macro_f1=float(np.mean(values))))
    best=max(choices,key=lambda r:r['mean_macro_f1'])
    meta=make_pipeline(StandardScaler(),LogisticRegression(C=best['C'],class_weight='balanced',max_iter=1000)).fit(scores,y[dev])
    branches=[fit_candidate(records,y,dev,c,int(np.median(epochs[c['name']])) if epochs[c['name']] else None) for c in candidates]
    bundle=dict(kind='stack',branches=branches,meta=meta,candidate=dict(name='group_oof_stack',branches=candidates),meta_training='group-held-out base predictions only')
    return bundle,dict(meta_choices=choices,chosen_meta=best,inner_splits=audits,meta_scores=scores.tolist(),base_candidates=candidates,epochs=epochs)

def main():
    OUT.mkdir(exist_ok=True)
    labels=json.loads((location('dataset')/'sarcasm_data.json').read_text(encoding='utf-8')); keys=sorted(labels)
    y=np.array([int(labels[k]['sarcasm']) for k in keys]); groups=np.array([labels[k]['show']+'::'+labels[k]['speaker'] for k in keys])
    records=[]
    for k in keys:
        with np.load(ROOT/'cache_v2'/(k+'.npz')) as z: records.append({m:z[m] for m in z.files})
    protocol=json.loads((ROOT/'artifacts_v2/protocol.json').read_text(encoding='utf-8'))
    stack_protocol=dict(outer_folds=protocol['outer_folds'],meta_C=[.01,.1,1,10],method='group-held-out stacking; separate group early stopping within each base-fit split',status='exploratory follow-up; not a new unseen external benchmark')
    if (OUT/'protocol.json').exists() and json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))!=stack_protocol:
        raise RuntimeError('Stacking protocol changed; use a new output directory')
    save_json(OUT/'protocol.json',stack_protocol)
    rows=[]; fold_reports=[]; start=time.monotonic()
    for i,fold in enumerate(protocol['outer_folds']):
        target=OUT/f'fold_{i+1}.json'
        if target.exists():
            report=json.loads(target.read_text(encoding='utf-8'))
            assert {r['id'] for r in report['predictions']}==set(fold['test'])
            rows.extend(report['predictions']); fold_reports.append(report); continue
        dev=np.array([keys.index(k) for k in fold['train']]); test=np.array([keys.index(k) for k in fold['test']])
        selection=json.loads((ROOT/f'artifacts_v2/fold_{i+1}_selection.json').read_text(encoding='utf-8'))['selection']
        candidates=select_branches(selection)
        bundle,audit=fit_stack(records,y,groups,dev,candidates,SEED+i+1)
        bundle['train_ids']=fold['train']
        joblib.dump(bundle,OUT/f'fold_{i+1}.joblib'); save_json(OUT/f'fold_{i+1}_fit_audit.json',audit)
        prediction_scores=bundle_scores(bundle,records,test); pred=(prediction_scores>0).astype(int)
        predictions=[dict(id=keys[j],fold=i+1,expected=int(y[j]),predicted=int(p),score=float(s),show=labels[keys[j]]['show']) for j,p,s in zip(test,pred,prediction_scores)]
        report=dict(fold=i+1,metrics=summary(y[test],pred),predictions=predictions)
        save_json(target,report); rows.extend(predictions); fold_reports.append(report)
        print(f'Stack fold {i+1}/5 saved elapsed={time.monotonic()-start:.1f}s',flush=True)
    assert len(rows)==len(keys) and len({r['id'] for r in rows})==len(keys)
    rows.sort(key=lambda r:r['id']); truth=np.array([r['expected'] for r in rows]); pred=np.array([r['predicted'] for r in rows])
    report=dict(count=len(rows),metrics=summary(truth,pred),folds=[dict(fold=f['fold'],metrics=f['metrics']) for f in fold_reports],predictions=rows,by_show={})
    for show in sorted({r['show'] for r in rows}):
        idx=np.array([r['show']==show for r in rows]); report['by_show'][show]=summary(truth[idx],pred[idx])
    valid=np.array([not json.loads((ROOT/'cache'/(r['id']+'.json')).read_text(encoding='utf-8'))['asr_empty'] for r in rows])
    report['nonempty_asr']=dict(coverage=float(valid.mean()),metrics=summary(truth[valid],pred[valid]))
    save_json(OUT/'report.json',report)
    # Choose base families for deployment from previously stored inner-validation
    # scores; never choose from outer metrics just computed.
    mean_inner=json.loads((ROOT/'artifacts_v2/deployment_selection.json').read_text(encoding='utf-8'))['mean_inner_scores']
    selection=[dict(candidate=c,mean_macro_f1=mean_inner[c['name']]) for c in CANDIDATES]
    bundle,audit=fit_stack(records,y,groups,np.arange(len(keys)),select_branches(selection),SEED+99)
    bundle.update(train_ids=keys,protocol='all-data deployment model; only outer fold predictions are evaluation')
    joblib.dump(bundle,OUT/'deployment.joblib'); save_json(OUT/'deployment_audit.json',audit)
    print(json.dumps(report['metrics'],indent=2),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1): main()
