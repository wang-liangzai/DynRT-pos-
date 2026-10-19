"""Frozen pretrained encoders and identical training/inference preprocessing."""
import os,sys,json,subprocess
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
from scipy import signal
from scipy.fft import dct

ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
def location(key):
    p=Path(CONFIG[key]); return p if p.is_absolute() else (ROOT/p).resolve()
def save_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'); temp.replace(path)

def read_audio(path):
    raw=subprocess.check_output([str(location('ffmpeg')),'-v','error','-i',str(path),'-vn','-ac','1','-ar','16000','-f','s16le','pipe:1'],timeout=120)
    return np.frombuffer(raw,dtype='<i2').astype(np.float32)/32768

def audio_features(wave):
    if len(wave)<400: wave=np.pad(wave,(0,400-len(wave)))
    _,_,z=signal.stft(wave,fs=16000,nperseg=400,noverlap=240,nfft=512,boundary=None)
    power=np.abs(z)**2+1e-10
    hz=np.linspace(0,8000,257); mel=lambda x:2595*np.log10(1+x/700)
    edges=700*(10**(np.linspace(mel(0),mel(8000),42)/2595)-1)
    bank=np.maximum(0,np.minimum((hz[None,:]-edges[:-2,None])/(edges[1:-1,None]-edges[:-2,None]),(edges[2:,None]-hz[None,:])/(edges[2:,None]-edges[1:-1,None])))
    cep=dct(np.log(bank@power+1e-10),type=2,axis=0,norm='ortho')[:20]
    rms=np.sqrt(power.sum(0)); centroid=(hz[:,None]*power).sum(0)/power.sum(0)
    frames=np.lib.stride_tricks.sliding_window_view(wave,400)[::160]
    zcr=np.mean(np.diff(np.signbit(frames),axis=1),axis=1)
    stats=np.concatenate([cep.mean(1),cep.std(1),np.mean(np.abs(np.diff(cep,axis=1)),axis=1) if cep.shape[1]>1 else np.zeros(20),[rms.mean(),rms.std(),centroid.mean(),centroid.std(),zcr.mean(),zcr.std(),len(wave)/16000]])
    return stats.astype(np.float32)

class Speech:
    def __init__(self):
        sys.path.insert(0,str(location('asr_dependencies')))
        from vosk import Model,SetLogLevel
        SetLogLevel(-1)
        previous=Path.cwd()
        try:
            os.chdir(location('asr_model').parent)
            self.model=Model(location('asr_model').name)
        finally: os.chdir(previous)
    def transcribe(self,wave):
        from vosk import KaldiRecognizer
        rec=KaldiRecognizer(self.model,16000)
        pcm=(np.clip(wave,-1,1)*32767).astype('<i2').tobytes()
        parts=[]
        for start in range(0,len(pcm),8000):
            if rec.AcceptWaveform(pcm[start:start+8000]): parts.append(json.loads(rec.Result()).get('text',''))
        parts.append(json.loads(rec.FinalResult()).get('text',''))
        return ' '.join(parts).strip()

def frames_from_video(path):
    cap=cv2.VideoCapture(str(path)); frames=[]; times=[]
    try:
        fps=cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps<=0: raise ValueError('Invalid video FPS')
        i=0; next_time=0.
        while True:
            ok,frame=cap.read()
            if not ok: break
            t=i/fps
            if t+1e-6>=next_time:
                frames.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
                times.append(t); next_time+=1/CONFIG['sample_hz']
            i+=1
    finally: cap.release()
    if not frames: raise ValueError('No video frames')
    indices=np.linspace(0,len(frames)-1,min(len(frames),CONFIG['max_frames'])).astype(int)
    return [frames[i] for i in indices],[times[i] for i in indices]

class Encoders:
    def __init__(self,device=None):
        import timm
        from timm.models.vision_transformer import checkpoint_filter_fn
        from torchvision import transforms
        from transformers import RobertaModel,RobertaTokenizer
        torch.set_num_threads(4)
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        repo=location('reference_repo')
        self.tokenizer=RobertaTokenizer.from_pretrained(str(repo/'roberta-base'),local_files_only=True)
        self.text=RobertaModel.from_pretrained(str(repo/'roberta-base'),local_files_only=True).eval().to(self.device)
        self.vision=timm.create_model('vit_base_patch32_clip_224',pretrained=False,num_classes=512)
        weights=torch.load(repo/'vit_base_patch32_224/pytorch_model.bin',map_location='cpu',weights_only=True)
        self.vision.load_state_dict(checkpoint_filter_fn(weights,self.vision),strict=True)
        self.vision.eval().to(self.device)
        self.transform=transforms.Compose([transforms.Resize(224,interpolation=transforms.InterpolationMode.BICUBIC),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    @torch.inference_mode()
    def encode_text(self,text):
        data=self.tokenizer(text or '',return_tensors='pt',max_length=192,truncation=True).to(self.device)
        out=self.text(**data).last_hidden_state
        mask=data['attention_mask'].unsqueeze(-1)
        return ((out*mask).sum(1)/mask.sum(1).clamp_min(1))[0].cpu().numpy()
    @torch.inference_mode()
    def encode_images(self,images):
        result=[]
        for start in range(0,len(images),8):
            x=torch.stack([self.transform(im.convert('RGB')) for im in images[start:start+8]]).to(self.device)
            result.append(self.vision(x).cpu().numpy())
        return np.concatenate(result)

def assemble(parts,modalities):
    return np.concatenate([parts[m] for m in modalities]).astype(np.float32)
