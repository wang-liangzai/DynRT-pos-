# VideoSarcasm：图文与音视频讽刺检测

最新改进实测见 `artifacts_ensemble/改进与全量评估.md`。集成版通过折外预测训练融合器，不使用基模型对其训练样本的拟合输出；外层分组测试用于最后评分。相比同划分重新训练的 V1，点估计指标有所改善，但不是所有类别或节目都提升。全部实验属于同一已查看数据池上的探索性改进，尚需新的外部测试。

```powershell
python prepare_v2.py
python train_v2.py
python train_ensemble.py
python report_ensemble.py
python test_v2.py
python verify_v2.py
python predict_v2.py path/to/new_video.mp4 --model artifacts_ensemble/deployment.joblib --context-video path/to/context.mp4
python live.py --engine ensemble --camera 0 --microphone 1
```

`--context-video` 可以省略。无前文输入的全量检查见 `artifacts_ensemble/missing_context_report.json`，这不等于真实摄像头测试。集成版的部署模型、各折模型及全部 690 条折外预测都在 `artifacts_ensemble`。原 V1、V2 结果保留用于对照。

一键执行或断点续跑：`python run_v2.py --verify-media`。已有特征时可加 `--skip-preparation`。运行前先安装 requirements.txt 中的依赖并检查 config.json 的模型、语音识别依赖、数据集和 FFmpeg 路径。sounddevice 仅实时摄像头麦克风入口需要。

## V2：时序注意力、真实上下文与全量评估

V1 文件与报告保留，不覆盖。V2 新增：

- `prepare_v2.py`：从真实上下文视频的音轨转写前文，缓存有顺序的多帧视觉特征；不读取标注台词。
- `fusion_v2.py`：可训练时序注意力、跨模态注意力、模态门控、目标句与上下文交互；训练时随机丢弃部分模态，所有可训练参数纳入优化器。
- `train_v2.py`：外层五折说话人分组评估覆盖全部 690 条；内层三折分组验证负责模型选择和早停。对比 V1 线性 SVM、加入真实 ASR 上下文的 SVM、两种正则强度的注意力融合网络。
- `predict_v2.py`：使用真实输入推理，可提供 `--context-video` 或 `--context-text`；不加载数据集标签或台词。
- `live.py --engine v2`：把之前三段实际识别结果作为上下文。

```powershell
python prepare_v2.py
python train_v2.py
python report_v2.py
python test_v2.py
python predict_v2.py ../utterances_final/1_10009.mp4 --context-video ../context_final/1_10009_c.mp4
python live.py --engine v2 --camera 0 --microphone 1
```

新缓存与模型在 `cache_v2`、`artifacts_v2`。训练支持按已完成的外层折续跑；修改候选参数或预处理时必须使用新的输出目录，不能混用先前折结果。`protocol.json` 先于评估落盘，记录候选参数和划分。

`deployment.joblib` 是所有数据重训后的部署模型，只应用于新视频；其在已训练视频上的结果不能用来证明泛化。`report.json` 和 `oof_predictions.csv` 才是全部视频的折外效果。不会保证所有视频或所有节目都提升；用两类召回率、精确率和各折结果判断，不能用“全部猜讽刺”来抬高召回率。

独立的新项目，以邻近 DynRT-main 的图文建模思路和本地预训练模型资源为参照。原工程和旧权重保持不变。新项目没有复制原训练程序，也不是把旧模型换一个阈值。

## 本版改动

- 完整预训练 RoBERTa 文本编码器，带 attention mask；原模型配置仅使用一层编码器。
- 明确且严格加载本地 CLIP ViT-B/32 视觉权重，使用配套归一化。原构造代码使用 pretrained=False，这只是代码层面的发现，不代表已证明旧 checkpoint 完全没有学习视觉特征。
- 视频每秒抽帧，最多均匀保留 32 帧；视觉均值及标准差汇总。后者是帧间变化统计，不是显式动作理解。
- 实际音轨 Vosk 英文转写；67 维 MFCC/频谱/能量等音频统计特征。
- 冻结编码器，以小数据更容易训练的 SVM 建立可运行基线。融合为标准化、各模态 L2 归一化后拼接。不是完整复现 DynRT 的动态路由网络。
- 训练、验证、测试按 show::speaker 隔离；标准化只拟合训练集；超参数只依据验证集宏 F1 选择。固定模型后评估测试集。

## 文件与运行

在本目录运行。路径在 config.json 中配置，复制到另一台设备时必须更新。当前直接复用旁边 DynRT-main 的 RoBERTa 和 CLIP 权重，不会再次下载数百 MB 权重。

```powershell
python prepare.py
python train.py
python export_head.py
python -m unittest test_pipeline.py
python verify_inference.py
python predict.py ../utterances_final/1_10009.mp4 --output demo_result.json
python predict.py path/to/image.jpg --mode image_text --text "Your English sentence"
```

prepare.py 缓存每条视频结果，中断后重新运行会跳过已完成条目。修改编码器或预处理后必须使用新的缓存目录或显式移走旧缓存；不能混用旧特征。模型在 artifacts/*.joblib，逐条测试结果在 artifacts/report.json，划分在 artifacts/split.json。只加载可信的 joblib 文件。

也可用 `python run_all.py` 完成提取、训练、分类头导出、测试和报告生成。`export_head.py` 导出只依赖 NumPy 的分类头并核对全部样本分数；编码器仍需要 PyTorch 或后续 ONNX/OM 运行时。`verify_inference.py` 从原始视频重新走部署入口，核对缓存评估与实际推理的分数一致性。

video.joblib 使用真实 ASR 文字、视觉均值、视觉变化、音频；image_text.joblib 使用 ASR 风格文字和单帧/多帧平均视觉特征。图像入口可运行，但本次训练数据是视频片段，尚未单独验证静态图像数据集效果。

reference_video、reference_context_video 是使用标注台词/上下文的研究对照，不能把它们的结果声称为真实语音识别端到端效果。默认 predict.py 只使用实际输入和转写，不读取标签或标注台词。--text 是显式人工提供文字时的覆盖选项。

## 摄像头与麦克风

安装 sounddevice 和可用的 PortAudio/设备驱动后运行：

```powershell
python -m sounddevice
python live.py --camera 0 --microphone 1 --rms-threshold 0.015
```

先用设备列表确认麦克风编号。实时入口持续采集，按音频能量和静音结束分句，把对应画面和音频交给视频推理。阈值需按房间噪声调整；它是工程初版，不是经过验证的神经 VAD。推理队列满时会明确报告丢段。空转写输出证据不足，模型 margin 不是置信概率。默认只落盘文本结果，不持久保存原始音视频。

实时硬件链路、多人说话、中文和华为 NPU 尚未验证。当前英文模型拒绝显式中文文字输入；英文 ASR 不能可靠识别中文，也不能自动保证识别到的语言正确。目标为中文时需要中文 ASR、中文编码器与中文场景数据。

## 评估边界

这是 690 条影视对白上的探索性领域适配实验。之前这些数据已被旧模型评估过，因此不能宣称是完全从未查看过的新外部测试。当前说话人划分不保证同一场景、相邻对白或重复上下文完全隔离；没有 episode 元数据，因此需要后续人工核验和真实环境外部测试。不是官方 MUStARD 五折结果，不能直接比较论文表格。不同 split 的准确率不能与此前平衡 100 条结果直接相减。

开发板阶段应先测 CPU 端到端耗时，再导出文本/视觉编码器到 ONNX/OM 并检查数值一致性。SVM 可在 CPU 运行；本项目尚未生成或声称通过华为 OM/NPU 验证。

## 参考

- 本地参考：../../DynRT-main/model/DynRT.py、config/DynRT.json、deployment/predictor.py。
- MUStARD 官方项目与基线：https://github.com/soujanyaporia/MUStARD
- 数据集论文：https://aclanthology.org/P19-1455/

本项目参考其模态对照与数据组织思路；本次划分、特征及训练代码不同于官方基线。数据和预训练权重继续遵守各自许可证。
