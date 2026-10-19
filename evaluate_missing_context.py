"""Evaluate missing conversation history with the SAME held-out fold models.

This is a robustness check, not additional fitting or threshold selection.
"""
import json
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from features import ROOT,save_json
from fusion_v2 import bundle_scores
from train_v2 import summary

def main():
    folder=ROOT/'artifacts_ensemble'
    original=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    by_id={r['id']:r for r in original['predictions']}
    # The empty text embedding was computed by the same pretrained encoder.
    empty=None
    for path in sorted((ROOT/'cache_v2').glob('*.npz')):
        with np.load(path) as z:
            if not float(z['context_available']): empty=z['context_asr'].copy(); break
    if empty is None: raise RuntimeError('No cached empty-context embedding')
    rows=[]
    for i,fold in enumerate(protocol['outer_folds']):
        model=joblib.load(folder/f'fold_{i+1}.joblib')
        records=[]
        for key in fold['test']:
            assert key not in model['train_ids']
            with np.load(ROOT/'cache_v2'/(key+'.npz')) as z: row={m:z[m] for m in z.files}
            row['context_asr']=empty.copy(); row['context_available']=np.array(0,dtype=np.float32); records.append(row)
        scores=bundle_scores(model,records,np.arange(len(records)))
        for key,score in zip(fold['test'],scores): rows.append(dict(id=key,fold=i+1,expected=by_id[key]['expected'],predicted=int(score>0),score=float(score)))
    assert len(rows)==len(by_id) and len({r['id'] for r in rows})==len(by_id)
    result=dict(protocol='same trained outer models; context replaced by empty-text encoding and availability=0; no refitting or threshold tuning',metrics=summary(np.array([r['expected'] for r in rows]),np.array([r['predicted'] for r in rows])),predictions=rows)
    usable=[r for r in rows if not json.loads((ROOT/'cache'/(r['id']+'.json')).read_text(encoding='utf-8'))['asr_empty']]
    result['nonempty_asr']=dict(coverage=len(usable)/len(rows),metrics=summary(np.array([r['expected'] for r in usable]),np.array([r['predicted'] for r in usable])))
    save_json(folder/'missing_context_report.json',result)
    print(json.dumps(result['metrics'],indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=1): main()
