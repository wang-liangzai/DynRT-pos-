"""V2 prediction on external input. No dataset labels/transcripts are read."""
import argparse,json,re,time
from pathlib import Path
import joblib
import numpy as np
from features import ROOT,Encoders,Speech,read_audio,frames_from_video,audio_features,save_json
from fusion_v2 import bundle_scores

class PredictorV2:
    def __init__(self,model_path=None):
        self.bundle=joblib.load(model_path or ROOT/'artifacts_v2/deployment.joblib')
        self.enc=Encoders(); self.speech=Speech()
    def predict(self,path,text=None,context_text='',context_video=None):
        started=time.monotonic(); wave=read_audio(path)
        if text is None: text=self.speech.transcribe(wave)
        if context_video is not None: context_text=self.speech.transcribe(read_audio(context_video))
        if re.search(r'[\u3400-\u9fff]',(text or '')+' '+context_text): raise ValueError('English-trained model; Chinese requires retraining')
        if not (text or '').strip(): return dict(status='insufficient_evidence',reason='empty_transcript',text=text)
        images,times=frames_from_video(path); visual=self.enc.encode_images(images)
        row=dict(text=self.enc.encode_text(text),context_asr=self.enc.encode_text(' '.join(context_text.split()[-150:])),context_available=np.array(float(bool(context_text.strip())),dtype=np.float32),frames=visual,vision=visual.mean(0),motion=visual.std(0),audio=audio_features(wave))
        score=float(bundle_scores(self.bundle,[row],np.array([0]))[0]); prediction=int(score>0)
        return dict(status='ok',label=prediction,label_name=['non_sarcastic','sarcastic'][prediction],text=text,context_text=context_text,context_available=bool(context_text.strip()),score=score,score_type='uncalibrated decision score; zero is class boundary, not a confidence probability',frame_times=times,seconds=time.monotonic()-started,model=self.bundle['candidate'],model_scope=self.bundle.get('protocol','held-out fold model; training IDs stored in model artifact'),notice='Evaluate using saved out-of-fold reports or new external videos; do not report training-video predictions as unseen-test accuracy')

def main():
    p=argparse.ArgumentParser(); p.add_argument('path'); p.add_argument('--context-video'); p.add_argument('--context-text',default=''); p.add_argument('--text'); p.add_argument('--model'); p.add_argument('--output')
    a=p.parse_args(); result=PredictorV2(a.model).predict(a.path,a.text,a.context_text,a.context_video)
    if a.output: save_json(Path(a.output),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
