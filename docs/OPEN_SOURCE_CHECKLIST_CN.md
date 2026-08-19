# 公开发布验收清单

## 已由生成器强制检查

- 公开树来自白名单，不从私有主仓库直接打包。
- 不包含 `web/`、`storm_pet/web/`、FastAPI、网页测试或部署注册表。
- 不包含 `private/`、`incoming/`、pickle、PyTorch checkpoint 或本地绝对路径。
- 公开 CSV 表头不包含 PTID、RID、LONIUID 或 DOB。
- 每个文件记录 SHA-256 和大小，见 `PUBLIC_RELEASE_MANIFEST.json`。

## 发布前人工检查

- 选择并加入明确的源代码许可证；这是当前唯一硬阻塞项。
- 补充正式作者、仓库 URL、论文 DOI 和引用信息。
- 确认 pySuStaIn、POT 等第三方许可证及引用要求。
- 决定是否单独发布经过治理审查的预训练参数；默认不公开模型参数。
- 在全新环境运行 `python -m pytest tests/unit`（当前为 38 项）。
- 仅在获授权环境用真实队列运行正文全流程，检查主文结果；不得上传输入队列或个体输出。

## 边界说明

该包复现 SuStaIn 与 OT-CFM 正文科学流程，不承诺复现 Appendix 的 CV/消融，也不包含
ddHodge 或网页。若将来公开 ddHodge，应作为新的白名单范围独立审计。
