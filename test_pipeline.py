import unittest
import numpy as np
from features import audio_features,frames_from_video,location

class PipelineTests(unittest.TestCase):
    def test_silent_and_short_audio_finite(self):
        for n in (0,40,16000):
            x=audio_features(np.zeros(n,dtype=np.float32))
            self.assertEqual(x.shape,(67,)); self.assertTrue(np.isfinite(x).all())
    def test_temporal_sampling(self):
        frames,times=frames_from_video(location('dataset')/'utterances_final/1_10009.mp4')
        self.assertEqual(len(frames),len(times)); self.assertGreater(len(times),1)
        self.assertEqual(times[0],0)
        self.assertTrue(all(b>a for a,b in zip(times,times[1:])))
        self.assertTrue(all(abs(t-round(t))<.1 for t in times))
    def test_saved_split_and_train_only(self):
        import json,joblib
        from features import ROOT
        path=ROOT/'artifacts/split.json'
        if not path.exists(): self.skipTest('Training not completed')
        s=json.loads(path.read_text()); data=json.loads((location('dataset')/'sarcasm_data.json').read_text())
        groups=[{data[k]['show']+'::'+data[k]['speaker'] for k in s[name]} for name in ('train','validation','test')]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        bundle=joblib.load(ROOT/'artifacts/video.joblib')
        self.assertEqual(set(bundle['train_ids']),set(s['train']))
        for scaler in bundle['scalers'].values(): self.assertEqual(scaler[0].n_samples_seen_,len(s['train']))

if __name__=='__main__': unittest.main()
