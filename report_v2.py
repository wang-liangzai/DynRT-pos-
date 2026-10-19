import json,csv
from features import ROOT

def main():
    folder=ROOT/'artifacts_v2'; r=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    lines=['# V2 全量分组交叉验证','',f'共 {r["count"]} 条，每条恰好由未使用其说话人训练的外层模型预测一次。5 个外层说话人分组折；每个外层训练部分内再做 3 折说话人分组验证来选择模型。','', '实际输入为视频、音轨及其自动转写。上下文也来自 context_final 视频音轨的自动转写，不使用标注台词。','', '| 方法 | 准确率 | 宏 F1 | 讽刺召回率 | 非讽刺召回率 | 讽刺精确率 | 讽刺 F1 | 预测讽刺数 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in [('V1 固定线性 SVM，同样的外层折重新训练',r['baseline']),('V2 内层验证选择的模型',r['metrics'])]:
        lines.append('|'+name+'|'+'|'.join(f'{m[k]:.2%}' for k in ['accuracy','macro_f1','sarcasm_recall','non_sarcasm_recall','sarcasm_precision','sarcasm_f1'])+'|'+str(m['predicted_sarcastic'])+'|')
    lines+=['','## 各折结果','','| 折 | 内层选中模型 | 测试条数 | 准确率 | 宏 F1 | 讽刺召回率 | 非讽刺召回率 |','|---|---|---:|---:|---:|---:|---:|']
    for f in r['folds']:
        m=f['metrics']; lines.append('|'+str(f['fold'])+'|'+f['winner']['name']+'|'+str(m['n'])+'|'+'|'.join(f'{m[k]:.2%}' for k in ['accuracy','macro_f1','sarcasm_recall','non_sarcasm_recall'])+'|')
    lines+=['','## 各节目表现','','| 节目 | 条数 | 准确率 | 宏 F1 | 讽刺召回率 | 非讽刺召回率 |','|---|---:|---:|---:|---:|---:|']
    for show,m in r['by_show'].items(): lines.append('|'+show+'|'+str(m['n'])+'|'+'|'.join(f'{m[k]:.2%}' for k in ['accuracy','macro_f1','sarcasm_recall','non_sarcasm_recall'])+'|')
    lines+=['','## 约束与解读','','- 此前这些数据已用于探索，当前是对同一数据池的更完整分组评估，不是全新的外部测试。','- 所有超参数选择、神经模型早停都在内层验证完成，外层结果不参与选择。不根据某条测试样本的答案写规则。','- 全量覆盖表示每条都有折外预测，不表示每条都正确，也不能保证每个节目或说话人都改善。','- 同一说话人不会跨外层训练/测试，但说话人隔离不等于场景或对话上下文完全隔离；未来需要 episode 级元数据或新场景验证。','- 视频上下文在数据集内可获得，现场由历史识别文字提供；第一句话没有上下文时可能下降。尚未做所有视频移除上下文的额外评估。','- 空转写在离线指标中使用其他模态继续评分；实际推理入口拒绝判断。nonempty_asr 单列实际文字可用覆盖率。','- deployment.joblib 使用全部 690 条重新训练供新视频使用，不能在这 690 条上测它再宣称是泛化效果。折外结果只来自 fold_*.joblib。','- 当前为英文模型，中文、华为 NPU 和真实多人摄像头场景尚未验证。','','## 混淆矩阵','','行是真实 [非讽刺,讽刺]，列是预测 [非讽刺,讽刺]：',f'`{r["metrics"]["confusion_matrix"]}`','']
    (folder/'全量评估报告.md').write_text('\n'.join(lines),encoding='utf-8')
    with (folder/'oof_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(r['predictions'][0])); writer.writeheader(); writer.writerows(r['predictions'])
    print(json.dumps({'v1':r['baseline'],'v2':r['metrics']},indent=2))

if __name__=='__main__': main()
