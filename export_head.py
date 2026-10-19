"""Export trained SVM + scaling to portable NumPy; verify against sklearn."""
import json
import joblib
import numpy as np
from features import ROOT

class NumpyHead:
    def __init__(self,path):
        self.data=np.load(path,allow_pickle=False)
        self.modalities=self.data['modalities'].tolist()
    def decision(self,parts):
        d=self.data; vectors=[]
        for m in self.modalities:
            z=(np.asarray(parts[m],dtype=np.float64)-d[m+'_mean'])/d[m+'_scale']
            z=z/max(float(np.linalg.norm(z)),1e-12); vectors.append(z)
        x=np.concatenate(vectors)
        if str(d['kernel'])=='linear': kernel=d['support']@x
        else: kernel=np.exp(-float(d['gamma'])*np.sum((d['support']-x)**2,axis=1))
        return float(kernel@d['dual']+d['intercept'])

def main():
    checks={}
    for name in ('image_text','video'):
        b=joblib.load(ROOT/'artifacts'/(name+'.joblib')); model=b['model']
        arrays=dict(modalities=np.array(b['modalities']),kernel=np.array(model.kernel),gamma=np.array(model._gamma),support=model.support_vectors_,dual=model.dual_coef_[0],intercept=model.intercept_[0])
        for m in b['modalities']:
            arrays[m+'_mean']=b['scalers'][m][0].mean_; arrays[m+'_scale']=b['scalers'][m][0].scale_
        path=ROOT/'artifacts'/(name+'_head.npz'); np.savez_compressed(path,**arrays)
        head=NumpyHead(path); errors=[]
        for feature_path in sorted((ROOT/'cache').glob('*.npz')):
            with np.load(feature_path) as parts:
                x=np.concatenate([b['scalers'][m].transform(parts[m][None]) for m in b['modalities']],axis=1)
                expected=float(model.decision_function(x)[0]); actual=head.decision(parts)
                errors.append(abs(expected-actual))
                assert (expected>0)==(actual>0),'Export changed prediction'
        assert max(errors)<1e-4
        checks[name]=dict(samples=len(errors),max_margin_error=max(errors))
    (ROOT/'artifacts/export_verification.json').write_text(json.dumps(checks,indent=2))
    print(checks)
if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1): main()
