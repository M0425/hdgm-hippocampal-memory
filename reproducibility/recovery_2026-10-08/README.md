# 图3 V1入口与主实验权重绑定

核对日期：2026-10-08。本材料保留2026-10-08的恢复和核对记录。没有重新训练模型、拟合探针或重跑干预实验。原先对应表中“V1入口未找到、主权重未盘点”的状态由本报告更新。

## 恢复的 V1 入口

原文件为 `C:/Users/LENOVO/Desktop/pythonProject32/检查点.py`。保留原字节并归档为 `code/HDGM_v4_stage_specific_perturbation_v1.py`；Downloads 同名副本哈希一致。

代码包含 phase-blind per-unit mean 校准、EC/CA3/CA1 的 zero/mean-clamp、encoding/maintenance/readout 时窗、冻结模型、相同测试银行与原模型控制等价检查。保存结果中 ten-seed 的无干预控制等价误差为零。当前源代码未重新执行，因此这一结论来自历史结果记录。

配套基础模型为 `code/HDGM_v4_rapid_acquisition_embedded.py`，从已归档 `探针节点.py` 解码恢复；SHA-256 为 `cd27093dc8c8d9cbb99085a880d48122a8853703dc7e054d0334bd0acedd9d8c`，与 gate/operation 协议精确匹配。原 V1 JSON 未保存评估脚本自身哈希；因此可确认现存代码与协议语义对应，无法单凭现存副本证明当时执行的脚本逐字节身份。具体核验见 `V1_SOURCE_RECOVERY.json`。

## 权重盘点结果

| 组 | 桌面主目录文件数 | 关键文件/内部快照 | 与论文结果的绑定 |
|---|---:|---|---|
| V4 四分类 | 10 | 每seed一个150-update容器，内含10/25/50/75/100/150 | 图2控制与图3operation/gate记录的文件SHA匹配；CA1的snapshot100张量指纹匹配 |
| Place V5 confirmatory | 60 | 10seed × 25/50/100/150/200/300 | 关键150/300检查点seed、update、配置匹配图4；历史结果未记checkpoint SHA |
| Splitter V7.3 confirmatory | 60 | 10seed × 25/50/100/150/200/300 | 关键150/300配置匹配图4；图5base_config仅运行名称后缀有差异，计算参数匹配 |

全搜索范围（Desktop、Downloads、Codex工作目录）共516个候选权重文件，按SHA去重为267份；包含开发实验、旧版本、调试与归档副本，不能将516解释为主实验数量。完整文件盘点见 `ALL_CHECKPOINT_FILES.csv`。同一哈希副本计数便于后续归档去重。本次未删除任何副本。

对50个关键文件读取了内部元数据：V4十个容器，位置20个endpoint文件，分叉20个endpoint文件。80条图—检查点核对均通过：40条为已有结果保存的文件SHA或CA1张量指纹；另外40条为位置/分叉的seed、update和配置匹配。两种证据强度不同，后一种不证明历史字节身份。图5的名称后缀差异在JSON中完整保留。

CA1实际选择的是容器内部 `snapshots[100]`；文件名150表示源运行预算，十个内部张量指纹均与CA1结果记录相同。无需因文件名而重做实验。

## 文件说明

- `ALL_CHECKPOINT_FILES.csv`：每个候选权重的原路径、大小、SHA与重复副本数。
- `SELECTED_WEIGHT_METADATA.json`：50个关键权重的完整配置、种子、endpoint、状态指纹及配置差异。
- `FIGURE_CHECKPOINT_BINDINGS.csv/.json`：80条绑定及证据类型。
- `V1_SOURCE_RECOVERY.json`：V1代码恢复来源、哈希、语义核对与历史身份限制。
- `SUMMARY.json`：核对范围和失败项（当前为空）。
- `run_figure3_v1.py`：可移植命令准备入口；默认只输出命令，带 `--execute` 才执行冻结评估。训练模型不会由此入口重新训练，但V1表征分析会运行原有Ridge分类诊断。

元数据使用CPU及 `torch.load(weights_only=True)` 读取，仅明确允许标准NumPy数组/dtype恢复；没有实例化模型或执行forward。文件和张量指纹分别记录；配置哈希使用排序键的UTF-8 JSON。

## 使用入口

在当前目录执行：

```sh
python run_figure3_v1.py
```

默认使用同目录 weights/v4/ 中的已归档容器；该目录不存在时才回退到原元数据位置。迁移到其他电脑后显式传入：

```sh
python run_figure3_v1.py --checkpoint-dir /path/to/paper_rapid_v4_outputs/checkpoints
```

只有确定要重新评估时才添加 `--execute`，并选择空的 `--output-dir`。依赖包括PyTorch、NumPy、Matplotlib和scikit-learn；本次没有执行该评估，也没有宣称新环境中的端到端验证已完成。原始十种子结果继续保留。

## 下一步值得补的材料

1. 本次上传集合已包含恢复的V1源、基础模型、清单、配置和关键权重，具体远端提交状态以仓库提交记录为准。
2. 对V4十个容器、位置和分叉关键endpoint权重做去重归档；暂不必复制所有开发/旧版本权重。本次仓库更新已复制V4十个容器，以及位置/分叉的150和300关键endpoint各20个文件，见ARCHIVED_WEIGHT_LOCATORS.csv。
3. 位置/分叉/图5的历史脚本身份仍缺原运行source SHA；优先寻找旧日志或配置版本记录。现存脚本作为快照保存，勿伪造历史哈希。
4. 图5尚缺 `rebuild_figure5_capacity_boundary.py`；属于另一项绘图复现缺口。
