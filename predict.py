"""Image+text or actual video+audio inference. No labels are read."""
import argparse,json,re,time
import joblib
import numpy as np
from PIL import Image
from features import *

class Predictor:
    def __init__(self,mode='video',device=None):
        if mode not in ('video','image_text'): raise ValueError('Use video or image_text')
        self.mode=mode
        self.bundle=joblib.load(ROOT/'artifacts'/(mode+'.joblib'))
        self.enc=Encoders(device)
        self.speech=Speech() if mode=='video' else None
    def predict(self,path,text=None):
        started=time.monotonic()
        if self.mode=='video':
            wave=read_audio(path)
            if text is None: text=self.speech.transcribe(wave)
            images,times=frames_from_video(path)
        else:
            if not text: raise ValueError('Image inference requires --text')
            images=[Image.open(path).convert('RGB')]; times=[0.]
        if re.search(r'[\u3400-\u9fff]',text or ''): raise ValueError('This trained version supports English; Chinese requires target-language training')
        if not (text or '').strip():
            return dict(status='insufficient_evidence',reason='empty_transcript',text=text)
        visual=self.enc.encode_images(images)
        parts=dict(text=self.enc.encode_text(text),vision=visual.mean(0),motion=visual.std(0))
        if self.mode=='video': parts['audio']=audio_features(wave)
        b=self.bundle
        x=np.concatenate([b['scalers'][m].transform(parts[m][None]) for m in b['modalities']],axis=1)
        pred=int(b['model'].predict(x)[0]); margin=float(b['model'].decision_function(x)[0])
        return dict(status='ok',label=pred,label_name=['non_sarcastic','sarcastic'][pred],text=text,decision_margin=margin,score_type='uncalibrated_svm_margin_not_probability',frame_times=times,seconds=time.monotonic()-started,mode=self.mode)

def main():
    p=argparse.ArgumentParser(); p.add_argument('path'); p.add_argument('--mode',choices=['video','image_text'],default='video'); p.add_argument('--text'); p.add_argument('--output'); p.add_argument('--device')
    a=p.parse_args(); result=Predictor(a.mode,a.device).predict(a.path,a.text)
    if a.output: save_json(Path(a.output),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
