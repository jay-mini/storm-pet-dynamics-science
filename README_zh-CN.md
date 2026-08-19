# STORM PET dynamics 科学复现包

本目录只包含 Aβ/Tau 科学流程：获授权数据预处理、全数据 SuStaIn 训练与安全推理包、 OT-CFM 训练和轨迹重建。

主流程依次运行：

1. `scripts/01_prepare_data.py`
2. `scripts/02_train_sustain.py`
3. `scripts/02_export_sustain_bundle.py`
4. `scripts/02_extract_sustain_results.py`
5. `scripts/02_assign_stage_bins.py`
6. `scripts/03_train_ot_cfm.py`

命令示例和数据约定见英文 [README](README.md)。`data/demo/` 仅为合成输入格式示例，
不能替代获授权队列，也不能复现论文数值。正式 SuStaIn 训练只产生正文使用的 full-data
模型；CV 与 Appendix 不在本公开范围。

发布前还必须由作者选择代码许可证；当前生成清单会明确标记
`publication_ready: false`，详见 `docs/OPEN_SOURCE_CHECKLIST_CN.md`。
