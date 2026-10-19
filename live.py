"""Continuous single-speaker capture, energy endpointing, bounded inference queue.

Requires sounddevice/PortAudio and a microphone. Ctrl+C stops capture.
No camera or microphone is activated by importing this module.
"""
import argparse,collections,queue,threading,time,tempfile,wave,subprocess
import cv2
import numpy as np
from features import ROOT,location,save_json
from predict import Predictor

def main():
    import sounddevice as sd
    p=argparse.ArgumentParser()
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--microphone',type=int)
    p.add_argument('--rms-threshold',type=float,default=.015)
    p.add_argument('--silence-seconds',type=float,default=.8)
    p.add_argument('--max-seconds',type=float,default=12)
    p.add_argument('--output',default=str(ROOT/'live_results'))
    p.add_argument('--engine',choices=['v1','v2','ensemble'],default='v1')
    a=p.parse_args()
    if a.max_seconds<=0 or a.max_seconds>30: p.error('max-seconds must be in (0,30]')
    if a.engine in ('v2','ensemble'):
        from predict_v2 import PredictorV2
        predictor=PredictorV2(ROOT/'artifacts_ensemble/deployment.joblib' if a.engine=='ensemble' else None)
    else: predictor=Predictor('video')
    output=__import__('pathlib').Path(a.output); output.mkdir(parents=True,exist_ok=True)
    stop=threading.Event(); video_frames=collections.deque(maxlen=450)
    lock=threading.Lock(); audio_queue=queue.Queue(maxsize=400); jobs=queue.Queue(maxsize=2)
    cap=cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    if not cap.isOpened(): raise RuntimeError('Cannot open camera')
    def camera():
        while not stop.is_set():
            ok,frame=cap.read()
            if not ok: stop.set(); return
            if frame.shape[1]>384:
                frame=cv2.resize(frame,(384,round(frame.shape[0]*384/frame.shape[1])))
            with lock: video_frames.append((time.monotonic(),frame))
            stop.wait(.065)
    def callback(indata,frames,timing,status):
        if status: print('Audio warning:',status,flush=True)
        try: audio_queue.put_nowait((time.monotonic(),indata[:,0].copy()))
        except queue.Full:
            print('Audio overflow; stopping to avoid invalid synchronization',flush=True); stop.set()
    def infer():
        history=collections.deque(maxlen=3)
        while not stop.is_set() or not jobs.empty():
            try: stamp,audio,frames=jobs.get(timeout=.2)
            except queue.Empty: continue
            try:
                with tempfile.TemporaryDirectory(prefix='sarcasm_') as tmp:
                    from pathlib import Path
                    d=Path(tmp); sr=16000; duration=len(audio)/sr
                    with wave.open(str(d/'audio.wav'),'wb') as f:
                        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr); f.writeframes((np.clip(audio,-1,1)*32767).astype('<i2').tobytes())
                    h,w=frames[0][1].shape[:2]
                    writer=cv2.VideoWriter(str(d/'silent.avi'),cv2.VideoWriter_fourcc(*'MJPG'),10,(w,h))
                    if not writer.isOpened(): raise RuntimeError('VideoWriter failed')
                    try:
                        timestamps=np.array([t for t,_ in frames])
                        start=stamp-duration
                        for t in np.arange(0,duration,.1): writer.write(frames[int(np.argmin(abs(timestamps-(start+t))))][1])
                    finally: writer.release()
                    subprocess.run([str(location('ffmpeg')),'-v','error','-y','-i',str(d/'silent.avi'),'-i',str(d/'audio.wav'),'-c:v','mpeg4','-c:a','aac','-shortest',str(d/'clip.mp4')],check=True,timeout=60)
                    if a.engine in ('v2','ensemble'): result=predictor.predict(d/'clip.mp4',context_text=' '.join(history))
                    else: result=predictor.predict(d/'clip.mp4')
                    if result.get('text'): history.append(result['text'])
                result['capture_end_monotonic']=stamp
                save_json(output/f'{time.time_ns()}.json',result)
                print(result,flush=True)
            except Exception as exc: print('Inference error:',repr(exc),flush=True)
            finally: jobs.task_done()
    workers=[threading.Thread(target=camera,daemon=True),threading.Thread(target=infer,daemon=True)]
    for worker in workers: worker.start()
    active=[]; pre=collections.deque(maxlen=3); silence=0
    try:
        with sd.InputStream(device=a.microphone,samplerate=16000,channels=1,dtype='float32',blocksize=1600,callback=callback):
            print('Listening. Ctrl+C stops. Energy endpointing requires room-specific threshold tuning.',flush=True)
            while not stop.is_set():
                try: stamp,block=audio_queue.get(timeout=.5)
                except queue.Empty: continue
                speech=np.sqrt(np.mean(block**2))>=a.rms_threshold
                if not active:
                    pre.append(block)
                    if not speech: continue
                    active=list(pre); pre.clear()
                else: active.append(block)
                silence=0 if speech else silence+.1
                if silence>=a.silence_seconds or len(active)*.1>=a.max_seconds:
                    audio=np.concatenate(active); active=[]; silence=0
                    with lock: frames=[(t,f.copy()) for t,f in video_frames if stamp-len(audio)/16000<=t<=stamp]
                    if len(audio)>=8000 and frames:
                        try: jobs.put_nowait((stamp,audio,frames))
                        except queue.Full: print('Inference busy: segment dropped (not classified)',flush=True)
    except KeyboardInterrupt: pass
    finally:
        stop.set(); cap.release()
        for worker in workers: worker.join(timeout=3)

if __name__=='__main__': main()
