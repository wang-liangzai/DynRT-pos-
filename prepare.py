"""Label-free extraction; each sample is cached for resuming."""
import json,time
import numpy as np
from features import *

def main():
    data=json.loads((location('dataset')/'sarcasm_data.json').read_text(encoding='utf-8'))
    out=ROOT/'cache'; out.mkdir(exist_ok=True)
    enc=Encoders(); speech=Speech(); started=time.monotonic()
    for i,(key,row) in enumerate(sorted(data.items())):
        path=out/(key+'.npz')
        if path.exists(): continue
        video=location('dataset')/'utterances_final'/(key+'.mp4')
        wave=read_audio(video); transcript=speech.transcribe(wave)
        frames,times=frames_from_video(video); visual=enc.encode_images(frames)
        features=dict(text=enc.encode_text(transcript),reference=enc.encode_text(row['utterance']),context=enc.encode_text(' '.join(row['context'][-3:])),vision=visual.mean(0),motion=visual.std(0),audio=audio_features(wave))
        assert all(np.isfinite(v).all() for v in features.values())
        save_json(out/(key+'.json'),dict(id=key,asr_text=transcript,frame_times=times,asr_empty=not bool(transcript)))
        with (out/(key+'.tmp')).open('wb') as f: np.savez_compressed(f,**features)
        (out/(key+'.tmp')).replace(path)
        if i%10==0: print(f'{i+1}/{len(data)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    print('All features ready',flush=True)
if __name__=='__main__': main()
