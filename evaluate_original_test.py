"""Frozen-model domain-transfer evaluation on the original DynRT image/text test.

No fitting, threshold tuning, train-set reads, or fabricated audio/context.
The newer audio/video ensemble is not applicable to this image-only input.
"""
import csv,hashlib,json,pickle,sys,time,types,re
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from features import ROOT,location,save_json
from predict import Predictor as NewPredictor
from train_v2 import summary

OUT=ROOT/'original_test_evaluation'

def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

@torch.inference_mode()
def main():
    repo=location('reference_repo'); source=repo/'input/prepared_clean'
    data={name:pickle.load((source/('test_'+name)).open('rb')) for name in ('id','text','labels')}
    assert len(data['id'])==len(data['text'])==len(data['labels'])
    assert len(set(map(str,data['id'])))==len(data['id'])
    OUT.mkdir(exist_ok=True); (OUT/'samples').mkdir(exist_ok=True)
    protocol=dict(dataset=str(source),split='test only',count=len(data['id']),test_files_sha256={n:digest(source/('test_'+n)) for n in data},new_model='artifacts/image_text.joblib',new_model_sha256=digest(ROOT/'artifacts/image_text.joblib'),original_checkpoint=str(repo/'exp/05-12-08_11_24/checkpoints/model_best.pth.tar'),original_checkpoint_sha256=digest(repo/'exp/05-12-08_11_24/checkpoints/model_best.pth.tar'),original_scoring='main+0.5*language+0.5*vision; fixed CPU seed2 routing noise matching deployed export',new_scoring='frozen image_text SVM; original decision boundary; no fitting',ensemble_status='not evaluated: this split has images and text, no genuine audio/context')
    if (OUT/'protocol.json').exists() and json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))!=protocol:
        raise RuntimeError('Dataset or model changed. Use a new output directory.')
    save_json(OUT/'protocol.json',protocol)
    newer=NewPredictor('image_text')
    sys.path.insert(0,str(repo)); sys.dont_write_bytecode=True
    from deployment.predictor import Predictor as OriginalPredictor
    from deployment.evaluate_mustard import frozen_gumbel
    from model.TRAR.trar import HardRoutingBlock
    original=OriginalPredictor()
    for module in original.model.modules():
        if hasattr(module,'fused_attn'): module.fused_attn=False
        if isinstance(module,HardRoutingBlock):
            module.register_buffer('fixed_noise',None)
            module.gumbel_softmax=types.MethodType(frozen_gumbel,module)
    torch.manual_seed(2)
    original.model(original.inputs(Image.new('RGB',(224,224)),'reference'))
    original.device=newer.enc.device; original.model.to(original.device)
    rows=[]; failures=[]; started=time.monotonic()
    for index,(key,text,label) in enumerate(zip(data['id'],data['text'],data['labels'])):
        key=str(key); path=OUT/'samples'/(key+'.json')
        guarded=bool(re.search(r'[\u3400-\u9fff]',text))
        if path.exists():
            cached=json.loads(path.read_text(encoding='utf-8')); cached['language_guard_bypassed_for_benchmark']=guarded
            rows.append(cached); continue
        try:
            image=repo/'dataset_image'/(key+'.jpg')
            if guarded:
                # Benchmark the unchanged raw string, including mixed-script tags.
                # Bypass only the application language guard, not preprocessing.
                with Image.open(image) as source_image: pil=source_image.convert('RGB')
                parts=dict(text=newer.enc.encode_text(text),vision=newer.enc.encode_images([pil]).mean(0))
                bundle=newer.bundle
                x=np.concatenate([bundle['scalers'][m].transform(parts[m][None]) for m in bundle['modalities']],axis=1)
                result=dict(label=int(bundle['model'].predict(x)[0]),decision_margin=float(bundle['model'].decision_function(x)[0]))
                ids=original.tokenizer(text)['input_ids'][:original.opt['len']]
                mask=[False]*len(ids)+[True]*(original.opt['len']-len(ids))
                ids+=[original.tokenizer.pad_token_id]*(original.opt['len']-len(ids))
                inputs={original.opt['input1']:torch.tensor([ids],device=original.device),original.opt['input2']:original.transform(pil).unsqueeze(0).to(original.device),original.opt['input3']:torch.tensor([mask],dtype=torch.bool,device=original.device)}
            else:
                result=newer.predict(image,text)
                if result['status']!='ok': raise RuntimeError(result)
                inputs=original.inputs(image,text)
            a,b,c,_=original.model(inputs); scores=(a+.5*b+.5*c)[0].cpu().numpy()
            if not np.isfinite(scores).all(): raise ValueError('Nonfinite original score')
            row=dict(id=key,expected=int(label),new_prediction=int(result['label']),new_margin=float(result['decision_margin']),original_prediction=int(scores.argmax()),original_scores=scores.tolist(),language_guard_bypassed_for_benchmark=guarded)
            save_json(path,row); rows.append(row)
        except Exception as exc:
            failures.append(dict(id=key,error=repr(exc)))
        if index%100==0 or index+1==len(data['id']): print(f'{index+1}/{len(data["id"])} scored={len(rows)} failures={len(failures)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    truth=np.array([r['expected'] for r in rows]); new=np.array([r['new_prediction'] for r in rows]); old=np.array([r['original_prediction'] for r in rows])
    report=dict(requested=len(data['id']),scored=len(rows),failures=failures,mixed_script_items_scored_without_application_guard=sum(r.get('language_guard_bypassed_for_benchmark',False) for r in rows),new_image_text=summary(truth,new),original_dynrt=summary(truth,old),majority_test_class_accuracy=float(np.bincount(truth).max()/len(truth)),results=rows)
    save_json(OUT/'report.json',report)
    with (OUT/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines=['# 新项目在原 DynRT 图文测试集上的效果','',f'全量测试：请求 {report["requested"]} 条，成功 {report["scored"]} 条，失败 {len(failures)} 条。使用原 input/prepared_clean/test_id、test_text、test_labels 与 dataset_image。','', '| 模型 | 准确率 | 宏 F1 | 讽刺精确率 | 讽刺召回率 | 讽刺 F1 | 非讽刺召回率 |','|---|---:|---:|---:|---:|---:|---:|']
    for title,m in [('新项目现有图文分支',report['new_image_text']),('原 DynRT 当前部署权重',report['original_dynrt'])]:
        lines.append('|'+title+'|'+'|'.join(f'{m[k]:.2%}' for k in ['accuracy','macro_f1','sarcasm_precision','sarcasm_recall','sarcasm_f1','non_sarcasm_recall'])+'|')
    lines+=['','## 测试范围与限制','','- 新项目使用已有 image_text.joblib，模型此前在 MUStARD 视频数据训练部分学习；本次不读取原图文训练集、不重新训练、不调阈值。','- 原 DynRT 使用 exp/05-12-08_11_24 的现有权重和完整组合评分。历史训练日志或其他评估脚本若评分方式不同，数字可能不同。','- 这是固定模型跨数据域测试。新模型并非在原图文数据训练部分重新训练后的架构对比。','- 最新音视频集成模型没有在这里测试，因为原测试集没有真实音频、前文视频；没有用伪造音频补输入并冒充完整模型效果。','- 全部样本逐条结果见 predictions.csv/report.json；protocol.json 保存数据和权重指纹。']
    lines+=['',f'- {report["mixed_script_items_scored_without_application_guard"]} 条含中文字符。模型级基准测试保留原始字符串并跳过应用层语言拦截；分词、图像变换和分类器不变。实际应用入口仍会拒绝这些输入。']
    (OUT/'原始测试集评估.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2),flush=True)
    if failures: raise RuntimeError('Some samples failed; inspect report and resume')

if __name__=='__main__': main()
