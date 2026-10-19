import csv,json
from features import ROOT,save_json

def main():
    from train import metrics
    import numpy as np
    r=json.loads((ROOT/'artifacts/report.json').read_text(encoding='utf-8'))
    titles={'text':'真实ASR文字','image_text':'真实ASR＋视觉','video':'真实ASR＋视觉＋音频','reference_video':'标注文字＋视觉＋音频','reference_context_video':'标注文字＋上下文＋视觉＋音频'}
    lines=['# 新项目实测报告','',f'划分数量：{r["split_counts"]}。训练、验证、测试按节目和说话人组合分组，同一说话人不跨集合。','', '所有 SVM 超参数按验证集宏 F1 选择。下面测试集没有参与标准化拟合、SVM 训练或超参数选择。','', '| 方法 | 测试样本数 | 准确率 | 宏 F1 | 讽刺召回率 | 讽刺 F1 |','|---|---:|---:|---:|---:|---:|']
    rows=[('旧 DynRT＋标注文字＋中间帧',r['original_dynrt_reference_midframe']),('训练集多数类基线',r['majority_baseline'])]+[(titles[n],v['test']) for n,v in r['results'].items()]
    for title,m in rows:
        lines.append('|'+title+'|'+str(m['n'])+'|'+'|'.join(f'{m[k]:.2%}' for k in ['accuracy','macro_f1','sarcasm_recall','sarcasm_f1'])+'|')
    lines+=['','## 正确解读','','- 旧模型与新模型在同一批测试 ID 上比较，但旧模型用标注文字，新模型部署分支用真实语音转写；这是端到端方案比较，不是只改变单一网络组件的严格消融。','- 新模型在此视频数据的训练部分进行了监督学习，旧模型没有。这说明领域适配的实际收益，不能单独证明新架构比旧架构优越。','- 视觉来源是视频抽帧；单张图片入口尚无独立图片测试结论。','- reference 为理想转写对照，不代表真实现场性能；上下文来自标注，仅供研究。','- 空转写时，离线批量评估仍用空字符串编码及其他模态评分；实际 predict.py 拒绝判断，因此原始批量指标和实际可用覆盖率需分别报告。','- 原始 690 条数据此前已做旧模型探索，当前分组也不保证节目片段级完全隔离；需要新采集场景的外部验证。','- 本版英文模型，不支持直接据此声称中文对话效果。','- 实时摄像头入口已实现，但未完成硬件实测；尚未部署或验证华为 NPU。','','查看 report.json 获取逐条预测、混淆矩阵和验证集结果。']
    (ROOT/'artifacts/实测报告.md').write_text('\n'.join(lines),encoding='utf-8')
    with (ROOT/'artifacts/test_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(['variant','id','expected','predicted','margin'])
        for name,result in r['results'].items():
            for row in result['predictions']: w.writerow([name,row['id'],row['expected'],row['predicted'],row['margin']])
    coverage={}
    for n,result in r['results'].items():
        if n.startswith('reference'): continue
        usable=[row for row in result['predictions'] if not json.loads((ROOT/'cache'/(row['id']+'.json')).read_text(encoding='utf-8'))['asr_empty']]
        coverage[n]=dict(coverage=len(usable)/len(result['predictions']),abstentions=len(result['predictions'])-len(usable),scored_metrics=metrics(np.array([x['expected'] for x in usable]),np.array([x['predicted'] for x in usable])))
        print(n,'nonempty_asr',len(usable),'total',len(result['predictions']))
    save_json(ROOT/'artifacts/deployment_coverage.json',coverage)
    with (ROOT/'artifacts/实测报告.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## 实际入口的覆盖率\n\n测试集中 1 条语音转写为空，实际入口会拒绝判断。音视频模型其余 99 条的准确率为 '+format(coverage['video']['scored_metrics']['accuracy'],'.2%')+'，讽刺 F1 为 '+format(coverage['video']['scored_metrics']['sarcasm_f1'],'.2%')+'。详细指标见 deployment_coverage.json。\n')

if __name__=='__main__': main()
