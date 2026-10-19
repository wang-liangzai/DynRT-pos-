"""Raw-video deployment path must reproduce cached evaluation features/scores."""
import json
import numpy as np
import joblib
from features import ROOT,location,save_json
from predict import Predictor

def main():
    split=json.loads((ROOT/'artifacts/split.json').read_text(encoding='utf-8'))
    engine=Predictor('video'); results=[]
    for key in split['test'][:3]:
        prediction=engine.predict(location('dataset')/'utterances_final'/(key+'.mp4'))
        cached=json.loads((ROOT/'cache'/(key+'.json')).read_text(encoding='utf-8'))
        assert prediction['text']==cached['asr_text'],(key,'ASR mismatch')
        if prediction['status']!='ok': continue
        with np.load(ROOT/'cache'/(key+'.npz')) as parts:
            b=engine.bundle
            x=np.concatenate([b['scalers'][m].transform(parts[m][None]) for m in b['modalities']],axis=1)
            expected=float(b['model'].decision_function(x)[0])
        error=abs(expected-prediction['decision_margin'])
        assert error<1e-4,(key,error)
        results.append(dict(id=key,margin_error=error,**prediction))
    save_json(ROOT/'artifacts/inference_verification.json',results)
    print(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
