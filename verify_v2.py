"""Check raw media -> prediction parity with saved features on outer-held-out rows."""
import json
import numpy as np
import joblib
from features import ROOT,location,save_json
from fusion_v2 import bundle_scores
from predict_v2 import PredictorV2

def main():
    protocol=json.loads((ROOT/'artifacts_v2/protocol.json').read_text(encoding='utf-8'))
    results=[]; engine=PredictorV2(ROOT/'artifacts_ensemble/fold_1.joblib')
    for fold in (1,2):
        path=ROOT/f'artifacts_ensemble/fold_{fold}.joblib'; engine.bundle=joblib.load(path)
        key=protocol['outer_folds'][fold-1]['test'][0]
        assert key not in engine.bundle['train_ids']
        result=engine.predict(location('dataset')/'utterances_final'/(key+'.mp4'),context_video=location('dataset')/'context_final'/(key+'_c.mp4'))
        cached_text=json.loads((ROOT/'cache'/(key+'.json')).read_text(encoding='utf-8'))['asr_text']
        context=json.loads((ROOT/'cache_v2'/(key+'.json')).read_text(encoding='utf-8'))['context_asr']
        assert result['text']==cached_text and result['context_text']==context
        with np.load(ROOT/'cache_v2'/(key+'.npz')) as z: record={k:z[k] for k in z.files}
        expected=float(bundle_scores(engine.bundle,[record],np.array([0]))[0])
        error=abs(expected-result['score']); assert error<1e-4,(key,error)
        results.append(dict(id=key,fold=fold,score_error=error,prediction=result))
    save_json(ROOT/'artifacts_ensemble/inference_verification.json',results)
    print(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
