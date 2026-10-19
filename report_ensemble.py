import json,csv
from features import ROOT

def main():
    folder=ROOT/'artifacts_ensemble'
    result=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    prior=json.loads((ROOT/'artifacts_v2/report.json').read_text(encoding='utf-8'))
    lines=['# 全量 690 条音视频讽刺检测：代码改进与实测','', '本轮不使用标注台词作为推理输入。目标句与前文均从实际视频音轨转写；标签仅用于训练监督和指标统计。','', '此前 100 条测试中新模型预测了 38 条讽刺、62 条非讽刺，38 条真实讽刺识别对 19 条，所以召回率 50%；并不是所有样本都猜同一个答案。','', '下面改用固定五折、按说话人分组的全量折外评估。每条视频恰好由未使用该说话人训练的模型预测一次。不能与之前单个 100 条划分直接相减比较。','', '| 方案 | 准确率 | 宏 F1 | 讽刺精确率 | 讽刺召回率 | 非讽刺召回率 | 讽刺 F1 |','|---|---:|---:|---:|---:|---:|---:|']
    for title,m in [('V1 同划分重新训练',prior['baseline']),('上下文/注意力单模型选择',prior['metrics']),('分组折外集成',result['metrics'])]:
        lines.append('|'+title+'|'+'|'.join(f'{m[k]:.2%}' for k in ['accuracy','macro_f1','sarcasm_precision','sarcasm_recall','non_sarcasm_recall','sarcasm_f1'])+'|')
    m=result['metrics']
    lines+=['','## 不是全部猜同一类','',f'690 条中预测讽刺 {m["predicted_sarcastic"]} 条、非讽刺 {m["predicted_non_sarcastic"]} 条。','', '| 真实标签 | 预测非讽刺 | 预测讽刺 |','|---|---:|---:|',f'|非讽刺|{m["confusion_matrix"][0][0]}|{m["confusion_matrix"][0][1]}|',f'|讽刺|{m["confusion_matrix"][1][0]}|{m["confusion_matrix"][1][1]}|','','## 修改思路与对应代码','','1. prepare_v2.py：真实前文语音识别、有顺序的视觉帧特征；不把数据集标注文字送进模型。','2. fusion_v2.py：学习帧权重和跨模态关系；加入文本与上下文差异、模态门控、模态随机丢弃；优化器覆盖全部可训练参数。','3. train_v2.py：外层五折覆盖所有数据，内层三折选择超参数。保留简单模型对照，避免把模型复杂度当成效果。','4. train_ensemble.py：结合无上下文基线、真实上下文模型、时序注意力模型。融合层使用分组折外分数训练，不能使用基模型对训练样本的拟合分数。神经基模型的早停另外在其训练部分内留出说话人。','5. predict_v2.py、live.py：复用同样的推理流程，实时前文来自最近三段实际识别文字。','','## 各节目表现及局限','','| 节目 | 样本数 | 非讽刺/讽刺数量 | 准确率 | 讽刺召回率 | 非讽刺召回率 |','|---|---:|---|---:|---:|---:|']
    for show,met in result['by_show'].items():
        neg=sum(met['confusion_matrix'][0]); pos=sum(met['confusion_matrix'][1])
        negrec=f'{met["non_sarcasm_recall"]:.2%}' if neg else '无法评估：无非讽刺样本'
        posrec=f'{met["sarcasm_recall"]:.2%}' if pos else '无法评估：无讽刺样本'
        lines.append(f'|{show}|{met["n"]}|{neg}/{pos}|{met["accuracy"]:.2%}|{posrec}|{negrec}|')
    nonempty=result['nonempty_asr']
    lines+=['','整体指标提升不代表所有样本、节目、两类召回率都同时提升。非讽刺召回率有所下降；BBT 的非讽刺误报仍较多。某些节目标签极不平衡或只有正类，其高准确率不能说明模型能区分两类。','',f'11 条目标句转写为空，实际推理入口拒绝判断。其余 {nonempty["metrics"]["n"]} 条覆盖率 {nonempty["coverage"]:.2%}，准确率 {nonempty["metrics"]["accuracy"]:.2%}，讽刺召回率 {nonempty["metrics"]["sarcasm_recall"]:.2%}。','', '这是对已用于探索的数据池的迭代实验，集成方案是在观察早期全量结果后提出的，不能称为全新盲测。没有按某条测试答案调整规则，但仍需外部新数据确认泛化提升。说话人隔离也不保证相邻场景、共享前文完全隔离。','', '部署模型用全部数据重新训练，不能在这些训练视频上测它并当成测试成绩。report.json/oof_predictions.csv 使用的才是折外模型。模型版本和各折训练 ID、内层选择及集成拟合审计均保存在本目录。','', '当前是英文实验。摄像头与麦克风实机链路、中文、华为 NPU 尚未验证。','', '运行集成推理：','', '```powershell','python predict_v2.py path/to/new_video.mp4 --model artifacts_ensemble/deployment.joblib --context-video path/to/previous_context.mp4','python live.py --engine ensemble --camera 0 --microphone 1','```']
    missing_path=folder/'missing_context_report.json'
    if missing_path.exists():
        missing=json.loads(missing_path.read_text(encoding='utf-8'))['metrics']
        lines+=['','## 没有前文时的额外检查','',f'同一批折外模型，不重训、不调阈值，将全部上下文置为空：准确率 {missing["accuracy"]:.2%}，讽刺召回率 {missing["sarcasm_recall"]:.2%}，讽刺精确率 {missing["sarcasm_precision"]:.2%}，非讽刺召回率 {missing["non_sarcasm_recall"]:.2%}。','', '没有前文时可以运行，但更容易把非讽刺误判为讽刺；不能只看召回率上升就认为更好。此项仍是数据集上的输入缺失检查，不等于真实摄像头环境测试。']
    (folder/'改进与全量评估.md').write_text('\n'.join(lines),encoding='utf-8')
    with (folder/'oof_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(result['predictions'][0])); writer.writeheader(); writer.writerows(result['predictions'])
    print('Report and all 690 out-of-fold predictions saved.')
if __name__=='__main__': main()
