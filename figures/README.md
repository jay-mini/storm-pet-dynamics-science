# 正文图片复现

本目录集中保存论文正文图所需的精简源数据、绘图代码和参考导出文件。所有绘图脚本只读取各自目录下的 `data/`，默认将结果写入同目录的 `outputs/`。

| 目录 | 内容 | 绘图脚本 |
| --- | --- | --- |
| `abeta_sustain/` | Aβ SuStaIn 主图与汇总图 | `plot_fig_abeta_sustain_1.py`、`plot_combined_sustain_summary_panel.py` |
| `tau_sustain/` | Tau SuStaIn 主图与汇总图 | `plot_fig_tau_sustain_ap_2.py`、`plot_combined_sustain_summary_panel.py` |
| `abeta_dynamics/` | Aβ OT-CFM dynamics 图 | `plot_fig_abeta_cfm_1.py` |
| `tau_dynamics/` | Tau OT-CFM dynamics 图 | `plot_integrated_dynamics_panel_3x3.py` |

## 环境与复现

在仓库根目录运行：

```powershell
python -m pip install -e ".[figures]"
python figures/reproduce_all.py
```

也可以只运行部分图，例如：

```powershell
python figures/reproduce_all.py --only abeta-sustain-main tau-dynamics
```

可用任务名可通过 `python figures/reproduce_all.py --help` 查看。随目录保留的 PNG、JPG 和 PDF 是参考导出；重新运行脚本会覆盖 `outputs/` 中同名文件。

Tau SuStaIn 使用随仓库提供的 `fsaverage5_surface_data.npz`，不需要额外配置 FreeSurfer `SUBJECTS_DIR`。数据包仅包含作图直接需要的精简数值或去标识化记录，不包含模型检查点或完整临床队列表。

## 数据发布说明

本目录当前未加入 `PUBLIC_RELEASE_MANIFEST.json`。其中部分纵向绘图表包含参与者对齐的派生值，且 Aβ dynamics 参考文件保留了 ADNI PTID；在获得相应数据发布许可并完成治理审查前，不应把这些文件纳入公开发布包。该限制不影响在获授权的本地项目中复现图片。
