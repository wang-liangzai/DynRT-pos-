"""Actual context ASR and ordered visual features; never read annotation text."""
from concurrent.futures import ThreadPoolExecutor
import json,time
import numpy as np
from features import *

OUT=ROOT/'cache_v2'

def main():
    OUT.mkdir(exist_ok=True)
    # IDs come from files, not utterance/reference/label fields.
    keys=sorted(p.stem for p in (ROOT/'cache').glob('*.npz'))
    enc=Encoders(); speech=Speech(); started=time.monotonic()
    def context(key):
        meta=OUT/(key+'.json')
        if meta.exists(): return key,json.loads(meta.read_text(encoding='utf-8'))
        path=location('dataset')/'context_final'/(key+'_c.mp4')
        text=speech.transcribe(read_audio(path)) if path.exists() else ''
        result={'context_asr':text,'source':str(path),'context_available':bool(text)}
        save_json(meta,result)
        return key,result
    todo=[k for k in keys if not (OUT/(k+'.npz')).exists()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(key,meta) in enumerate(pool.map(context,todo,buffersize=8)):
            images,times=frames_from_video(location('dataset')/'utterances_final'/(key+'.mp4'))
            visual=enc.encode_images(images)
            with np.load(ROOT/'cache'/(key+'.npz')) as old:
                values={m:old[m] for m in ('text','audio','vision','motion')}
            values.update(context_asr=enc.encode_text(' '.join(meta['context_asr'].split()[-150:])),frames=visual.astype(np.float32),context_available=np.array(float(meta['context_available']),dtype=np.float32))
            with (OUT/(key+'.tmp')).open('wb') as f: np.savez_compressed(f,**values)
            (OUT/(key+'.tmp')).replace(OUT/(key+'.npz'))
            if i%10==0: print(f'{len(keys)-len(todo)+i+1}/{len(keys)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    print('V2 feature extraction complete',flush=True)
if __name__=='__main__': main()
