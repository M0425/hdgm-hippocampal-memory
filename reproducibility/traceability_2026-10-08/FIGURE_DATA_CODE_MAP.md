# 主图—数据—脚本—代码版本对应表

> 后续恢复进度：V1入口已在 pythonProject32/检查点.py 找回；V4十个权重SHA和CA1内部snapshot100指纹均与结果记录匹配；位置/分叉确认性权重各60份齐全。当前状态及证据限制见 ../checkpoint_binding_2026-10-08/README.md。以下原对应表保留恢复前状态以便追溯。

核对日期：2026-10-08。核对基于本地已提交归档的内容；GitHub 已知归档提交为 `c4a5f5a3dac7b59da806ffda146b62c6f298d7d9`。本次未确认远端是否有后续提交，也未修改远端仓库。

所有路径相对于 `outputs/hdgm_reproducibility/`。方括号为 SHA-256 前12位；完整哈希见 CSV/JSON。文件名中的版本号、快照哈希、历史运行身份是不同证据。只有有运行协议哈希匹配时才称“运行版本精确匹配”。

| 图/面板 | 论文资产 | 数据 | 绘图入口 | 模型/评估代码与版本 | 核验状态及缺口 |
|---|---|---|---|---|---|
| 1 A–E | `manuscript/Codex 图像 2026年9月15日 11_37_51.png` [240895490008] | 概念说明，无数值数据 | 最终栅格图；绘图源未归档 | 概念图；机制细节对应 Methods 和模型实现<br>入口未识别 | 最终图像已锁定；没有可执行绘图源<br>缺口：若有原始可编辑设计文件可补入；无需数值复现实验 |
| 2 B–C | `manuscript/figure2_E_rebalanced_v2.pdf` [f49f952d1fe5] | `figure_sources/figure2_E_rebalanced_v2_outputs/data/fourclass_rapid_acquisition_v4_primary_results.json` [f3dd617c2f73]<br>`figure_sources/figure2_E_rebalanced_v2_outputs/data/figure2_missing_outputs/figure2_endpoint_predictions.npz` [358409bc6eb9] | `figure_sources/figure2_E_rebalanced_v2_outputs/plot_figure2_E_rebalanced_v2.py` [43df3b788a75] | V4；split_seed_version=4；150 updates；10 seeds 11…99,111<br>`code/HDGM主实验 四分类.py` [99e1bbe61cdd] | 绘图输入完整；训练源与历史运行的字节身份未证明<br>缺口：AUC CI 原始估计过程未保存；部分统计注释写死；需记录检查点版本关系 |
| 2 D–E | `manuscript/figure2_E_rebalanced_v2.pdf` [f49f952d1fe5] | `figure_sources/figure2_E_rebalanced_v2_outputs/data/figure2_missing_outputs/figure2_missing_summary.json` [59dee6f1d774]<br>`figure_sources/figure2_E_rebalanced_v2_outputs/data/figure2_missing_outputs/figure2_history_controls.csv` [4f81e9c4f1a2]<br>`figure_sources/figure2_E_rebalanced_v2_outputs/data/figure2_missing_outputs/figure2_shortcut_diagnostics.csv` [b1a365df6a1e]<br>`figure_sources/figure2_E_rebalanced_v2_outputs/data/figure2_missing_outputs/figure2_delay_transfer.csv` [762bd3ed1d30]<br>`figure_sources/figure2_E_rebalanced_v2_outputs/data/figure2_missing_outputs/figure2_endpoint_predictions.npz` [358409bc6eb9] | `figure_sources/figure2_E_rebalanced_v2_outputs/plot_figure2_E_rebalanced_v2.py` [43df3b788a75] | 冻结 V4 update-150；10 paired seeds<br>`code/绘图数据导出.py` [4ccbf3f6aa9f] | CSV/NPZ/冻结协议完整；原检查点本体未入包<br>缺口：明确 delay 的三比较 Holm 家族；检查点按保存的 manifest/hash 找回 |
| 3 A,C | `manuscript/figure3_refined_v2.png` [dda18483d7c8] | `figure_sources/figure3_refined_v2_outputs/data/operation.json` [4e67b8bf02d1] | `figure_sources/figure3_refined_v2_outputs/plot_figure3_refined.py` [757f27360710]<br>`figure_sources/figure3_refined_v2_outputs/original_plot.py` [7692126f181f] | V4 semantic operation-cut V2；150；10 seeds<br>`code/探针节点.py` [b52b9be27967] | 绘图数据 SHA 与数值审计吻合；嵌入 V4 的 SHA 与运行协议精确匹配<br>缺口：探针节点.py 评估脚本自身的历史运行 SHA 未存；基础主实验恢复成功但检查点未打包 |
| 3 B | `manuscript/figure3_refined_v2.png` [dda18483d7c8] | `figure_sources/figure3_refined_v2_outputs/data/gate.json` [8e627cde7aa6] | `figure_sources/figure3_refined_v2_outputs/plot_figure3_refined.py` [757f27360710]<br>`figure_sources/figure3_refined_v2_outputs/original_plot.py` [7692126f181f] | V4 stage-specific perturbation V1；zero/mean_clamp；150<br>入口未识别 | 数据和基础模型版本可核验；V1 门控干预入口未识别<br>缺口：探针节点.py 是 V2 operation-cut/shuffle，不等同于 V1 zero/mean-clamp；找回实际 V1 评估脚本 |
| 3 D–E | `manuscript/figure3_refined_v2.png` [dda18483d7c8] | `figure_sources/figure3_refined_v2_outputs/data/ca1.json` [9e06680fc2fd] | `figure_sources/figure3_refined_v2_outputs/plot_figure3_refined.py` [757f27360710]<br>`figure_sources/figure3_refined_v2_outputs/original_plot.py` [7692126f181f] | CA1 V6.2；split_seed_version=62；从150-update容器选取内部snapshot[100]<br>`code/CA1终.py` [c2cdb766d437] | 10条记录均保存 checkpoint_update=100/source_run_max_updates=150，和加载代码一致；100/150歧义已解释<br>缺口：历史评估脚本自身SHA未存；缺checkpoint本体以复核内部snapshot的state_dict指纹；不要用V6.1替代V6.2 |
| 4 B,F(position) | `manuscript/figure4_refined.pdf` [dad17dd3ec6a] | `figure_sources/figure4_refined_outputs/data/position.json` [6514559171e0] | `figure_sources/figure4_refined_outputs/plot_figure4_refined.py` [4c2ca1595aa6]<br>`figure_sources/figure4_refined_outputs/original_plot.py` [ec06cc468a8a] | Place V5 confirmatory；10 seeds 101…1010；sensory/action residualization<br>`code/位置群体.py` [00d649606788] | 绘图数组和输入 SHA 经原数值审计核验<br>缺口：位置.py 不含已用 residualizer，应以位置群体.py 为候选；缺原运行 source SHA/全检查点 |
| 4 C–F(history) | `manuscript/figure4_refined.pdf` [dad17dd3ec6a] | `figure_sources/figure4_refined_outputs/data/branch.json` [901390dcb24c] | `figure_sources/figure4_refined_outputs/plot_figure4_refined.py` [4c2ca1595aa6]<br>`figure_sources/figure4_refined_outputs/original_plot.py` [ec06cc468a8a] | Splitter V7.3 双 endpoint；10 seeds 101…1010<br>`code/分叉.py` [29dd1d11b1e3] | 绘图数组与输入 SHA 经原数值审计核验；训练快照候选<br>缺口：10-seed 数据没有绑定训练源的 source SHA；需找回原 checkpoint/config/日志 |
| 5 B–F | `manuscript/figure5_no_titles.pdf` [7cecc4e654a2] | `data/pythonProject32/hdgm_difficulty_v2_1_frozen_outputs_confirmatory/difficulty_v2_1_multiseed_results.json` [97232c8139fc]<br>`data/New project/figure5_integrated_BC_v2_provenance.json` [a7753445ca20] | `data/New project/rebuild_figure5_integrated.py` [0195e2a2ddc8]<br>`data/New project/rebuild_figure5_panel_b_bifurcation.py` [d53858b075bf] | difficulty V2.1 frozen confirmatory；repr150/behavior300；10 seeds 101…1010<br>`code/混叠噪声.py` [8362a6944019]<br>`code/分叉.py` [29dd1d11b1e3] | 数据 SHA 与绘图 provenance 精确匹配；完整集成绘图仍缺依赖<br>缺口：归档漏掉 response-atlas 和去标题链；本次已找回；capacity_boundary.py 仍未找到；需移除本机绝对路径 |
| A1 A–D | `manuscript/figure_A1.pdf` [e8f9862e962b] | `pilot/checkpoint_diagnostics_v1/summary.json` [526e8719faec]<br>`pilot/checkpoint_diagnostics_v1/protocol.json` [43f4e58bfd2c]<br>`appendix_A1/plotted_data.csv` [c2909482b5ea] | `appendix_A1/build_figure.py` [988d34b7afd3] | 3 seeds 11,22,33 ×4 variants ×150/300；24 records/checkpoints<br>`code/分析_三种子原版.py` [f82bfe819293]<br>`code/分叉.py` [29dd1d11b1e3]<br>`code/补充评估.py` [0e6f8bd3c1e8] | 训练来源/诊断来源 SHA 与协议吻合；24检查点具备且已核验<br>缺口：补充评估代码默认从code同目录读输出，与归档pilot目录不符，需增路径参数（当前未提供）；当前分析.py不能替代原版 |

## 哪些缺口值得补

| 优先级 | 补充内容 | 判断及完成标准 |
|---|---|---|
| P0 | 图2统计出处与自动注释 | 旧脚本把 +9.8 pp、CI [3.0,16.5]、p=.023/.070 写死。已存审计以20,000次seed bootstrap、seed24680重算CI为[3.0158,16.5393]pp，舍入匹配；三比较Holm重算为.0703125。值得补统计记录和自动注释，现有数值无需因“写死”而推翻。新审计是可复算的重构出处，不能冒充原绘图时的估计日志。 |
| P0 | 图3 B 的 V1 评估脚本 | 涉及关键门控干预。找到实际 zero/mean-clamp 实现，核对时窗、均值校准、银行、种子与冻结操作；数据存在不等于执行代码已完整归档。 |
| 已解释，无需新实验 | 图3 D/E 的100/150 | CA1终.py 从150-update运行的checkpoint容器中读取 snapshots[100]；10条记录均保存update100/max_updates150。只需在版本表说明并补权重本体，文件名150不表示探针用了最终snapshot150。 |
| P1 | 图5完整绘图与去标题链 | response-atlas 本次找回且与原 provenance 哈希相同；capacity_boundary.py 仍缺。先找回源文件；若无法找回，保留批准的PDF并提供数值重建入口，诚实注明不支持原版式一键重建。 |
| P1 | 训练/checkpoint版本绑定 | 优先补锁定V4和图4/图5共用的splitter检查点、配置和日志。先盘点已有文件，不必重新训练；避免重复复制相同权重。 |
| P1 | 脚本相对路径与命令说明 | 图2、3、4、A1绘图已有相对入口；不要笼统重写全部代码。图5和pilot诊断需参数化路径；恢复后的修改版本应单独hash记录。 |
| P2 | 图1原设计文件、全训练轨迹 | 有则收集；非现有数值结论闭合的首要条件。已保存的概念图足以查看，不能声称可用训练代码重建这张美术图。 |
| 条件性科学补充 | 冻结探针稳健性 | 属于分析稳健性而非归档修复。若继续强调 pooled/conditional 差异的机制解释，值得控制试次/采样/探针优化后验证；当前窄主张可保留探索性限制，不能用它替代缺失的源代码链。 |

## 本次补回与版本核验

- 安全解码 `探针节点.py` 中压缩的 V4 源代码，SHA `cd27093dc8c8d9cbb99085a880d48122a8853703dc7e054d0334bd0acedd9d8c` 与图3 operation/gate 协议吻合；无需执行训练。现有主实验四分类快照的不同哈希不再被误当作这份嵌入源的身份。
- `figure5c_response_atlas.py` 从桌面恢复，SHA `cbbf44f5ea585c24d1530d4b78ef0999a78eeeb0303018776f7af5f39b796d7a` 与原绘图 provenance 一致。
- 恢复图5标题删除脚本、源PDF/SVG、源CSV和图注到 `recovered/figure5/`。edit_manifest 的源PDF哈希匹配；工作目录保存的去标题输出与论文最终PDF逐字节哈希相同。因此图5最后的去标题加工链已核对闭合（本次未重新执行加工）。这些恢复文件尚未加入远端归档。
- A1 的运行来源是 `分析_三种子原版.py`，不是当前同名分析脚本。其训练来源、分叉来源、诊断脚本的SHA均匹配协议。
- 147个已归档文件与MANIFEST哈希匹配；本次未重新训练、执行探针拟合或改图。

## 可执行绘图入口（静态核对；本次未运行）

在解压后的 `hdgm_reproducibility/` 根目录：
```sh
python figure_sources/figure2_E_rebalanced_v2_outputs/plot_figure2_E_rebalanced_v2.py
python figure_sources/figure3_refined_v2_outputs/plot_figure3_refined.py
python figure_sources/figure4_refined_outputs/plot_figure4_refined.py
python appendix_A1/build_figure.py
```

图3/4新版入口依赖同目录 original_plot.py 的函数，但不会调用其旧 main() 中的上传路径或插画素材。图5集成入口目前不能照此运行。上述绘图依赖 NumPy/Matplotlib，A1另需Pillow；数值验证入口另见README。
