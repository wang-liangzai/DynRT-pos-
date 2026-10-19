import unittest,json
import numpy as np
import torch
from features import ROOT,location
from fusion_v2 import MODALITIES,Preprocessor,TemporalFusion

class V2Tests(unittest.TestCase):
    def test_masked_frames_do_not_affect_output(self):
        dims=dict(text=8,context_asr=8,audio=4,frames=6)
        model=TemporalFusion(dims,dropout=0).eval()
        x={m:torch.randn(2,12,d) if m=='frames' else torch.randn(2,d) for m,d in dims.items()}
        x.update(frame_mask=torch.arange(12)[None].repeat(2,1)>=3,context_available=torch.tensor([1.,0.]))
        with torch.no_grad():
            before=model(x); x['frames'][:,3:]=10000; after=model(x)
        torch.testing.assert_close(before,after)
    def test_gradients_reach_all_trainable_modules(self):
        dims=dict(text=8,context_asr=8,audio=4,frames=6)
        model=TemporalFusion(dims,dropout=0)
        x={m:torch.randn(4,12,d) if m=='frames' else torch.randn(4,d) for m,d in dims.items()}
        x.update(frame_mask=torch.zeros(4,12,dtype=torch.bool),context_available=torch.ones(4))
        torch.nn.functional.cross_entropy(model(x),torch.tensor([0,1,0,1])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
    def test_scaler_fits_training_rows_only(self):
        records=[dict(text=np.full(8,i,dtype=np.float32),context_asr=np.full(8,i,dtype=np.float32),audio=np.full(4,i,dtype=np.float32),frames=np.full((3,6),i,dtype=np.float32),context_available=np.array(1)) for i in (0,2,1000)]
        prep=Preprocessor().fit(records,[0,1]); self.assertTrue(np.all(prep.scalers['text'].mean_==1))
        self.assertEqual(prep.scalers['frames'].n_samples_seen_,6)
    def test_oof_no_leakage(self):
        path=ROOT/'artifacts_v2/report.json'
        if not path.exists(): self.skipTest('Full CV not completed')
        protocol=json.loads((ROOT/'artifacts_v2/protocol.json').read_text(encoding='utf-8'))
        labels=json.loads((location('dataset')/'sarcasm_data.json').read_text(encoding='utf-8'))
        group=lambda k:labels[k]['show']+'::'+labels[k]['speaker']
        seen=[]
        for fold in protocol['outer_folds']:
            self.assertFalse(set(fold['train'])&set(fold['test']))
            self.assertFalse({group(k) for k in fold['train']}&{group(k) for k in fold['test']})
            seen.extend(fold['test'])
        self.assertEqual(len(seen),len(set(seen))); self.assertEqual(set(seen),set(labels))
        result=json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(len(result['predictions']),len(labels))
    def test_ensemble_meta_predictions_group_held_out(self):
        folder=ROOT/'artifacts_ensemble'
        if not (folder/'report.json').exists(): self.skipTest('Stacking not completed')
        labels=json.loads((location('dataset')/'sarcasm_data.json').read_text(encoding='utf-8')); keys=sorted(labels)
        group=lambda i:labels[keys[i]]['show']+'::'+labels[keys[i]]['speaker']
        protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
        for i,fold in enumerate(protocol['outer_folds']):
            audit=json.loads((folder/f'fold_{i+1}_fit_audit.json').read_text(encoding='utf-8'))
            seen=[]
            for split in audit['inner_splits']:
                self.assertFalse({group(i) for i in split['train']}&{group(i) for i in split['validation']})
                self.assertTrue({keys[j] for j in split['train']+split['validation']}<=set(fold['train']))
                seen.extend(split['validation'])
            self.assertEqual(len(seen),len(set(seen)))
            self.assertEqual({keys[j] for j in seen},set(fold['train']))
if __name__=='__main__': unittest.main()
