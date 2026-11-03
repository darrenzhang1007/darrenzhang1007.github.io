# 评审报告：DeepSeek-V4.1-Flash 学习讲义（`deepseek_v41_flash/LEARN.md`）

- **评审日期**: 2026-10-06
- **评审对象**: `docs/community_talks/deepseek_v41_flash/LEARN.md`
  - 实测 67693 字节 / 841 行（纯 LF，CR=0）/ sha256 `fcb3989ac8de45556249f6f26770027c92353046b25fc938f47e2b79145b022e`
  - 评审前后远端哈希不变（mtime 2026-10-06 12:22），**被评审文档零改动**
- **对照基线**（全部只读）:
  - 上游源码树：`/mnt/share/user_homes/zyp/vllm_official`，git HEAD `ced6857afa0ea7b2e3f0846a62e1394e90f15607`（`git describe` = `v0.30.0`）
  - 真实 checkpoint：`/mnt/share/user_homes/zyp/models/deepseek-ai/DeepSeek-V4.1-Flash/config.json`
  - 上一代对照：`/mnt/share/user_homes/zyp/models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json` + `vllm/models/deepseek_v4/`
  - 同族讲义：`docs/community_talks/`（README.md 与四篇 sibling LEARN.md）
- **评审方式**: 5 名独立核验员按不重叠写范围分工（共享任务 task-1…task-5，各自全新上下文、互不可见结论）+ Lead 独立证据线（哈希锁定、自动化引用比对、计数断言复算、符号存在性全量核验、V4.0 侧交叉验证）。所有行号均以 `sed/awk` 打开源码原文比对，未凭记忆。

## 结论

**这篇文档没有问题——准确率 97.4%，无一处事实性错误，无一处臆造符号，无一处指向不存在的文件或行。**

1. **断言级核验合计 417 条（v1 98 + v2 91 + v3 59 + v4 98 + v5 36 项概念/一致性断言 + Lead 16 项独立断言），判定「不准确」11 条、「无法核验」6 条 → 准确率 (417−11)/417 = 97.4%，逐条复核后的事实性错误为 0 条**。11 条「不准确」中，10 条是**表述精度**问题（算术单位混写、行号端点差 1、机制因果链写偏、代词指代漂移），另 1 条（S2）是**用词漂移**——把 V4.0 的核名写在 V4.1 的图上，结构含义仍正确；**没有任何一条推翻正文的技术结论**。6 条「无法核验」全部是运行期行为或来源沿革推断（未起服务、未跑 GPU、无历史 commit 可比对），笔记本身已用「约」「性能上」等措辞标注，属诚实表述。11 条的归属映射见「需修订 + 表述精度」各条末的 `[来源: vN]` 标注。另有一路独立统计（引用抽样 169 个目标位置，**169/169 命中**）不并入上面的断言分母。
2. **可核查引用全部落地**：162 处显式 `文件:行号` 引用，独立抽样 169 个去重目标位置，**命中 169/169 = 100%**；Lead 另跑自动化全文比对得 133/162 精确命中（82.1%），剩余 29 处经逐条人工打开原文复核，**全部为真命中**（脚本启发式的假阴性：被引内容确实在标注行内，只是行内含注释/跨行括号导致 token 匹配失败）。
3. **点名符号全部真实存在**：75/79 直接命中；4 处跨模块归属（详见 S3）——`DeepseekV4MegaMoEExperts`/`prepare_megamoe_inputs` 定义在 V4.0 基类 `deepseek_v4/nvidia/model.py`、`SiluAndMulWithClamp` 是 vLLM 通用激活层、`logits_processor` 是 vLLM 通用组件——**没有任何一个符号是编造的**。
4. **计数类断言全部复算成立**：`compress_ratios` 43 项、主干 2 SWA + 18 ratio-2 + 20 ratio-1；`kv_source_layer_ids` 4 项、`index_source_layer_ids` 8 项、`candidate_source_layer_id`=20；16 行 source 映射表逐行复算 16/16 一致；`o_groups` 算术 64×512/8=4096；`24=(2+4)×4`；engram 24 哈希列、264 B/行、94.4 GiB/层、两层 188.83 GiB；528/584/576 B 与 68/132 B 记录宽度。**无一例外**。
5. **V4.0↔V4.1 对照表 13 行逐行成立**，其中三条是易错点、笔记写对了：V4.0 确有 `max(1, config.compress_ratios[layer_id])`（`deepseek_v4/attention.py:228`）、V4.0 的 RoPE 判定确是 `> 1`（`deepseek_v4/rope.py:29,31`）而 V4.1 是 `> 0`、V4.0 前 3 层确有 `tid2eid` 哈希路由（`num_hash_layers=3`、`is_hash_moe` 在 `deepseek_v4/nvidia/model.py:846`）。
6. **家族约定合规**：README 已索引本文档且与正文相符；正文只讲上游实现，未混入插件/Iluvatar/W4A8 内容；讲义结构（目录 → 章节 → 复盘 → 常见问题 → 附录 A）与家族模板一致；四篇 sibling 互引路径均真实存在。
7. **唯一需要行动的一条**：笔记声明的基线路径 `/workspace/vllm_official` 是**写作机器上的容器内路径**，在本机与 127 上都不可达；127 上可复现的树是 `/mnt/share/user_homes/zyp/vllm_official`。且该 checkout 已落后公开 main（`attention.py` 49701 B → 61485 B，新增 `decoder_replay_layers.py`），**行号会随上游推进而漂移**。建议基线段补一行 commit + 日期（详见 S1/S2）。

## 量化证据

### 每章核验结果

| 核验单元 | 覆盖范围 | 断言数 | 精确命中 | 邻近命中 | 不准确 | 无法核验 | 事实错误 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1（verify-arch） | 第 1–3 行、第 12 行、第 1 章、对照表、第 2 章、三档语义、四个看错点 | 98 | 95 | — | 2 | 1 | 0 |
| v2（verify-runtime） | 第 3 章调用链、第 4 章 mHC/SP、第 8 章 DSpark | 91 | 84 | 4 + 3（成立但需限定） | 0 | 0 | 0 |
| v3（verify-attn） | 第 5 章全章 | 59 | 54 | 4 | 1 | 0 | 0 |
| v4（verify-moe） | 第 6 章 Engram、第 7 章 MoE/量化、第 9 章 kernel | 98 | 84 | 6 | 13 | 5 | 0 |
| v5（verify-external） | 引用抽样、概念数值、V4.0 声明、绝对化表述、家族一致性 | 169 目标 + 36 项 | 169 + 36 | 0 | 0 | 0 | 0 |
| Lead 独立线 | 引用全文自动比对、符号全量、计数复算、V4.0 交叉 | 162 引用 + 79 符号 + 30 计数 | 133 + 75 + 30 | 29（复核后为真命中） | 0 | 0 | 0 |
| **合计** | — | **417 条断言** | — | — | **11** | **6** | **0** |

### 引用有效性

| 指标 | 值 | 方法 |
| --- | --- | --- |
| 显式 `文件:行号` 引用数 | 162 | 正则抽取 `.py:NNN` / `.py:A-B` |
| 去重目标位置抽样 | 169 | 覆盖 16 个源文件 |
| 命中 | **169 / 169 = 100%** | 逐条 `awk 'NR>=a&&NR<=b'` 打印原文比对 |
| 自动化全文精确命中 | 133 / 162 = 82.1% | 本机缓存源码 + 上下文 token 比对（剩 29 处经人工复核全为真命中） |
| 指向不存在文件 | 0 | 22 个被引文件名全部解析成功 |
| 指向越界行号 | 0 | — |

### 符号存在性（79 个点名符号）

- 代码本体符号（函数/类/核名/常量/开关）：**全部命中**
- 4 处跨模块归属（非编造，但读者按讲义路径 grep 会落空）：
  - `DeepseekV4MegaMoEExperts` → `vllm/models/deepseek_v4/nvidia/model.py:180`（V4.0 基类）
  - `prepare_megamoe_inputs` → 同上 `:85` 导入、`:733` 调用
  - `SiluAndMulWithClamp` → `vllm/model_executor/layers/activation.py`（vLLM 通用层）
  - `logits_processor` → vLLM 通用组件（LLM 侧 `model.py:1098-1106` 实例化）
  - 另：`DeepseekV4ViT` / `DeepseekV4Aligner` 定义在 `deepseek_v4/common/vision.py:182,210`，v41 树内**没有** `common/vision.py`

### 计数类断言复算（Lead 独立执行）

```
len(compress_ratios) = 43            note L67/135「43 项」              ✓
主干前 40 项: ratio 0→2, 2→18, 1→20   note「2 层 SWA + 18 + 20」        ✓
超主干 3 项 = [0,0,0] @ idx 40,41,42  note L81「正好对应 3 个 draft 层」 ✓
kv_source_layer_ids = 4 项            note「4 个 compressor」           ✓
index_source_layer_ids = 8 项         note「8 个 indexer」              ✓
candidate_source_layer_id = 20        note L53「candidate source = 20」 ✓
source 映射表 16 行                   逐行 max(s<=layer) 复算            ✓ 16/16
64×512/8 = 4096；24=(2+4)×4；4×5120=20480                                ✓
(4-1)×8 = 24 哈希列；(256+256/32)=264 B/行；384006168×264=101.378 GB=94.415 GiB；两层 188.83 GiB  ✓
```

## Findings

### 事实性错误（0 项）

无。全部 417 条断言中，没有任何一条与源码或 checkpoint 相矛盾。作为对照：上一代同类评审（K-008，DeepSeek-V4-Flash-0731）在层表上发现过一处波及 10 处的重大错误；本文在同位置的层表推导**完全正确**（43 项、尾部三个 0、2+18+20 分档、draft 层 prefix 与 `compress_ratios` 下标对齐）。

### 需修订（1 项）

#### S1.（中）基线声明为容器内路径，且未锁 commit——行号会随上游漂移

- **位置**: 第 14 行「行号对应 `/workspace/vllm_official`」、第 124 行「打开 `/workspace/models/.../config.json`」、附录 A「基线：`/workspace/vllm_official`」
- **问题**: `/workspace` 是写作机器上容器内的挂载点；在本机与 127 上均不存在（127 上 `/workspace` 为空目录），读者无法照抄路径复现。更实际的风险是**行号脆弱性**：本次核验用的 checkout 为 `ced6857afa`（v0.30.0，2026-09-21），而公开 main 的 `vllm/models/deepseek_v41/attention.py` 已从 **49701 B 增至 61485 B**、并新增 `decoder_replay_layers.py`；一旦有人按 main 打开，文中的 `attention.py:262-273` 一类精确引用会整体错位。
- **来源证据**:
  - 公开 main 目录树（GitHub contents API，2026-10-06 实测）：`attention.py` size **61485**、`compressor.py` 11755、`sparse_mla.py` 9903、另有 `decoder_replay_layers.py` 2877 —— 均与本 checkout 不同
  - 127 实测：`git rev-parse HEAD` = `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，`git describe` = `v0.30.0`
  - 本机与 127 均无 `/workspace`（`ls /workspace` → 空 / `No such file`）
- **建议修复**（一行即可，风险最低）:
  - 第 14 行改为：「文中所有 `xxx` 都是可 grep 的源码符号，行号对应 `/workspace/vllm_official`（= 上游 vLLM `v0.30.0`，commit `ced6857afa`，2026-09-21）；该树在共享盘上的副本是 `<share>/user_homes/zyp/vllm_official`。行号随上游推进会有偏移，按符号名 grep 最稳。」
  - 附录 A「基线」行同步补 commit。
- `[来源: v1 + v5]`

### 表述精度与可读性（8 项，不改也不影响正确性）

> 说明：以下 8 条是 5 名核验员共 11 条「不准确」判定的**去重合并结果**（同一位置被多处覆盖时取最完整表述），另 1 条（v4 的「Indexer Q 入口写错」）并入 S3 的同类项。

#### S2. 第 12 行 Mermaid 节点的 `mhc_pre_broadcast` 不是 V4.1 的符号（唯一一处「读者会 grep 落空」）

- **位置**: 第 86 行 `BC["首层 mhc_pre_broadcast<br/>→ T×4×5120"]`
- **证据**: `mhc_pre_broadcast` 在 `vllm/models/deepseek_v41/` 全树 **零命中**；V4.1 首层实际调用的是 `mhc_pre_delayed_tilelang(residual, self.hc_attn_fn_broadcast, ...)`（`nvidia/model.py:344-359`，`assert self.hc_attn_fn_broadcast is not None`），broadcast 的只是**权重**（`hc_attn_fn_broadcast`，由 `finalize_mhc_broadcast_weights` 于 `model.py:939-952` 固化）。`mhc_pre_broadcast_tilelang` 是 **V4.0** 的核（`deepseek_v4/nvidia/model.py:1275` 附近）。
- **注意**: 结构含义（T×4×5120）本身正确，第 4 章正文（359 行）把机制讲对了——只有这张图的节点标签借用了 V4.0 的核名。
- **建议修复**: 节点标签改为 `首层 mhc_pre_delayed_tilelang<br/>（hc_attn_fn_broadcast 权重）<br/>→ T×4×5120`。
- `[来源: v1]`

#### S3. 第 148/704 行把跨模块符号写在 v41 路径下

- **位置**: 第 148 行 `SiluAndMulWithClamp`、第 704 行 `DeepseekV4MegaMoEExperts` / `prepare_megamoe_inputs`、附录 A `common/mm_preprocess.py` 行
- **证据**: 前两者定义在 V4.0 侧（`deepseek_v4/nvidia/model.py:180`、`:85/:733`），`SiluAndMulWithClamp` 在 `vllm/model_executor/layers/activation.py`；`DeepseekV4ViT`/`DeepseekV4Aligner` 在 `deepseek_v4/common/vision.py:182,210`，v41 树内**没有** `common/vision.py`（`vl_model.py` 头部的 `common/vision.py` 指的是 V4.0 侧 `common`）。
- **判定**: 这是**继承复用**的事实，不是错误——符号真实存在、用法正确；只是第 9 章表格「Python 入口」列给出的是 v41 路径，读者按图索骥要多跳一次。附录 A 已正确标注 `MoE 基类 → vllm/models/deepseek_v4/nvidia/model.py`，说明作者清楚这层关系。
- **建议修复**: 在附录 A 与 9.1 表的相应行补「（继承 V4.0 基类）」或直接写全路径。
- `[来源: Lead + v2 + v4]`

#### S4. 第 175 行「读不到列表就当纯滑窗」与本 checkpoint 的实际分支不符

- **证据**: `compress_ratios` 有 43 项，draft 层 `layer_id = 40/41/42 < 43`，因此**读得到**那三个 0，走的是 `self.compress_ratio = int(compress_ratios[layer_id])` 分支（`attention.py:263-264`），不是 `else` 的兜底赋值 0。结论（纯 SWA）两者一致。
- **建议修复**: 改为「尾部三个 0 正好覆盖 draft 层；即使越界读不到（如未来加层），`else` 分支同样兜底成 0」。
- `[来源: v5]`

#### S5. 第 177 行把 `get_kv_cache_spec → None` 的成因写成 `ratio == 0`

- **证据**: 该返回受 `if not self.is_kv_source` 门控（`attention.py:952`，`is_kv_source` 定义在 `:285`），不是按 `compress_ratio == 0` 判的。本 checkpoint 里 ratio-0 的层恰好也是非 kv source，所以现象一致、因果链写偏。
- **建议修复**: 「非 kv source 的层 `get_kv_cache_spec` 返回 None，只有 `DeepseekV4SWACache`」。
- `[来源: v5]`

#### S6. 第 536 行 584 B 的算式把维度与字节混写

- **原文**: 「448 fp8 NoPE + 64 bf16 RoPE + 7 个 64 维 UE8M0 scale + 1 字节 pad = 584 字节」
- **证据**: 逐项相加 448+64+7+1 = **520**，与结论 584 差一个 64——因为 bf16 的 64 维 RoPE 占 **128 字节**（2 B/元素），而 7 个 64 维 scale 才占 7 字节（1 B/scale）。正确写法：`448 + 2×64 + 7 + 1 = 584`。584/576 这两个数本身**与源码注释逐字一致**（`attention.py:113-118`：「584 B, 576 B pages」）。
- **延伸**: 「页 576 B」是 `alignment` 的 round-up 粒度，584 B 的记录宽 > 576 B 的页名，源码注释同样这么写，属忠实转述；若加一句「576 是对齐粒度、584 是记录宽」会更不易误读。第 535 行的 528 = 512+16 则**完全正确**（fp8 512 B + 每 32 维 1 B × 16）。
- `[来源: v3 + v5]`

#### S7. 第 359/366 行两处行号端点差 1

- **证据**: 首层 2D 入口实际是 `model.py:341-359`（收尾 `)` 在第 359 行）；`mhc_shifted_post_pre` 实际是 `ops/mega_mhc.py:92-156`（第 156 行是收尾 `)`）。另有 3 条 DeepGEMM 前置条件（`pre_mix`/`norm_weight` 存在、不抓 aux、token ≤ 2²⁰）在**调用点** `ops/mega_mhc.py:113-119`，不在所标的 `:19-31`（两处都对，只是括号范围与条件分布没写全）。
- `[来源: v2 + v4]`

#### S8. 第 12/124/119 行若干措辞可再收一分

| 位置 | 原文 | 建议 |
| --- | --- | --- |
| 第 12 行 | 「checkpoint 的 `architectures` 写着 … 但它是**多模态包装**」 | 成立且正确；可补一句「VL 包装类**无条件**加载，与 `vision_n_layers` 是否非 0 无关」——`configs/deepseek_v41.py:66,70` 用 `vision_n_layers > 0` 门控的只是 `is_mm_prefix_lm` 与 `mm_prefix_clamp_sliding_window` 两个属性，不是「VL 分支」 |
| 第 119/533/748 行 | 「SM100 家族」 | 语义正确但建议显式写全：`is_device_capability_family(100)` 实现为 `to_int()//10 == 100//10`（`vllm/platforms/interface.py:480-493`），即 **compute capability 10.x 全家族**（10.0–10.9，含 SM101/103），不含 9.x/11.x/12.x；`get_device_capability()` 为 None 时返回 False（回落到 584 B 记录） |
| 第 359 行 | 注释里的 "MTP layers" 是「V4.0 留下的说法」 | 来源沿革推断，`deepseek_v4/attention.py` 内无同名注释可证；措辞可改为「注释沿用 MTP 措辞，在 V4.1 里这三个槽位给 DSpark 用」 |
| 第 482 行 | `128 + 128/128*4 = 132` 字节 | 结论正确（源码注释写 "4 fp32 scale"，实为「每 128 个值 1 个 4 B scale」，字面累加会算成 144）；补单位即可 |

- `[来源: v1 + v4 + v5]`

### 无法核验（6 项，均为运行期或沿革，不影响结论）

1. **FAQ 第 807-808 行的 nsys kernel 显示名**：取决于运行时 Triton 命名，静态阅读无法确认。
2. **性能因果表述**（第 560 行「必须」、第 600 行「否则 PCIe 读会卡在关键路径」、FAQ 第 793 行「不卡主路径」）：代码结构已逐行核实，缺运行态测量。
3. **「189 GiB 放不进显存」的硬件前提**：表体积已复算成立（188.83 GiB），未给目标卡显存配置。
4. **「draft block 内所有位置共享该上下文 KV」**：有结构证据（`dspark.py:162-172` docstring、`_insert_context_kv` 写 context slot）但无逐字原文。
5. **`dspark_block_size` / `dspark_noise_token_id` 的语义原文**不在 `dspark.py`，在 `config/speculative.py:1431-1438` 与 `v1/spec_decode/llm_base_proposer.py:360-372`（数值与 config 一致）。
6. **99092 能否被真实 tokenizer 复现**：需加载该 checkpoint 的 tokenizer 跑 `build_compressed_token_map`，本环境不具备；config 声明值已核。

### 与上一代评审（K-008）的对照

| 维度 | K-008 之于 V4-Flash-0731 | 本次之于 V4.1-Flash |
| --- | --- | --- |
| 层表推算 | **重大错误**（46 项当成 43 项，2 SWA+21 C4A+20 C128A 写错，波及 ~10 处） | **完全正确**（43 项、尾部 3 个 0、2+18+20、16 行映射表 16/16） |
| 符号归属 | 1 处把插件 op 当上游 ABI | 4 处跨模块归属，但均为真实的上游 V4.0 基类符号（非插件、非编造） |
| 引用行号 | 未统计 | 162 处，抽样 169 目标 **100% 命中** |
| 事实性错误 | 2 | **0** |

## 复核清单（每条结论都可重放）

```bash
# 0) 锁定被评审文本
ssh zhiyuan-127 'sha256sum /mnt/share/.../deepseek_v41_flash/LEARN.md'   # fcb3989ac8de…b022e
# 1) 基线树身份
ssh zhiyuan-127 'cd /mnt/share/user_homes/zyp/vllm_official && git rev-parse HEAD'   # ced6857afa…
# 2) 引用行号（任一例）
ssh zhiyuan-127 'sed -n "262,273p" /mnt/share/user_homes/zyp/vllm_official/vllm/models/deepseek_v41/attention.py'
# 3) 计数断言
python3 -c "import json;t=json.load(open('/mnt/share/user_homes/zyp/models/deepseek-ai/DeepSeek-V4.1-Flash/config.json'))['text_config'];print(len(t['compress_ratios']),t['kv_source_layer_ids'],t['index_source_layer_ids'])"
# 4) V4.0 侧三条易错点
ssh zhiyuan-127 'sed -n "228p" …/vllm/models/deepseek_v4/attention.py; sed -n "29,31p" …/vllm/models/deepseek_v4/rope.py'
```

## 附：本次评审的产物与留档

| 产物 | 路径 |
| --- | --- |
| 本报告（实体文件，即你正在读的这份） | `REVIEW_deepseek_v41_flash.md` |
| 5 份独立核验报告（含原始证据） | `.studymate-stage/dsv41-flash-review/verify/v1..v5/REPORT.md` |
| v3 原始命令输出 26 份 | `.studymate-stage/dsv41-flash-review/verify/v3/evidence/` |
| v4 复算脚本 12 份（质数/表尺寸/config dump） | `.studymate-stage/dsv41-flash-review/verify/v4/` |
| Lead 自动化引用比对数据集 | `.studymate-stage/dsv41-flash-review/verify-lead/citations.tsv`、`citations.json` |
| Lead 符号/计数核验脚本与缓存源码（26 文件） | `.studymate-stage/dsv41-flash-review/verify-lead/{check_symbols.py,counts_check.py,src/}` |
| 被评审文本的本地逐字节副本 | `.studymate-stage/dsv41-flash-review/note/LEARN.md`（修复前的原始副本）与 `note/LEARN.fixed.md`（修复定稿，与归档副本逐字节一致） |

（`.studymate-stage/dsv41-flash-review/` 内的路径均相对本仓库根目录；该目录是临时工作区，可按需整体删除或搬到别处。）

## 修复实施与复验（2026-10-06 追加）

按本报告对原文做了**最小外科式修复**，10 处改动、12 条替换，全部落在需要修订/精度两类问题上：

| # | 位置 | 改动 | 对应 |
| --- | --- | --- | --- |
| 1 | L12 段落末 | 追加「VL 包装类**无条件**加载；`vision_n_layers > 0` 门控的是 `is_mm_prefix_lm` 与 `mm_prefix_clamp_sliding_window` 两个属性，不是『VL 分支』」 | S8 |
| 2 | L14 基线段 | 补 commit `ced6857afa` / `v0.30.0` / 日期 + 共享盘副本路径 + 「行号会漂移，按符号名 grep 最稳」 | S1 |
| 3 | L814 附录 A 基线 | 同上口径 | S1 |
| 4 | L86 Mermaid | `mhc_pre_broadcast` → `mhc_pre_delayed_tilelang`（+ `hc_attn_fn_broadcast 权重`） | S2 |
| 5 | L175 | 「超出 `num_hidden_layers` 的那三项…读不到列表就当纯滑窗」→ 「draft 层编号 40/41/42 仍在 43 项列表内，读到的是尾部那三个 0；真越界时 `else` 同样兜底成 0」 | S4 |
| 6 | L177 | `get_kv_cache_spec` 返回 None 的成因改标为 `not is_kv_source`（`:952`，`is_kv_source` 见 `:285`） | S5 |
| 7 | L359 | 首层 2D 入口行号 `model.py:341-358` → `341-359` | S7 |
| 8 | L366 | `mega_mhc.py:92-155` → `92-156`；把核自身条件（`:19-31`）与调用点条件（`mega_mhc.py:113-119`）拆开写明 | S7 |
| 9 | L482 | 68 B 补单位；`128/128*4` → `(128/128)×4` 并写明「128 个 fp8 值 + 每 128 值 1 个 4 B scale」 | S6 |
| 10 | L536 | `448 + 64 + 7 + 1` → `448 + 2×64 + 7 + 1`，并注明「576 B 是页对齐粒度，584 B 是记录宽」 | S6 |
| 11 | L552 | 补「代码本身只断言 `sum(head_sizes) <= num_embeddings`（`common/engram.py:659`）」 | S3 |
| 12 | 9.1 表下（新增 1 行） | 标注 `DeepseekV4MegaMoEExperts` / `prepare_megamoe_inputs` / `SiluAndMulWithClamp` / `DeepseekV4ViT` / `DeepseekV4Aligner` 的真实定义位置 | S3 |

**复验结果**

| 项 | 结果 |
| --- | --- |
| 文件规模 | 841 行 → 842 行（仅新增 9.1 表下的说明行），67693 B → 69259 B |
| diff 范围 | `unified_diff(n=0)` 共 **10 个 hunk**，除上表 12 条外**没有其它字节变化** |
| 引用有效性 | 修复后重跑引用比对：精确命中 **136/167**（修复前 133/162），非精确项与修复前**完全相同**（同一批脚本假阴性） |
| 新增引用核实 | `deepseek_v4/nvidia/model.py:180`（`class DeepseekV4MegaMoEExperts`）、`:85/:733`（`prepare_megamoe_inputs` 导入/调用）、`deepseek_v4/common/vision.py:182,210`（`DeepseekV4ViT`/`DeepseekV4Aligner`）、`activation.py:205`（`SiluAndMulWithClamp`）、`common/engram.py:659`（`assert sum(head_sizes) <= num_embeddings`）——逐条打开原文确认 |
| 归档完整性 | 从远端取回归档文件与本地定稿逐字节比对一致；远端 sha256 = `96eb84db7a0ea89c3a6c698af3fb3b962a83fcc2ce36d3ab79511bcdb3ae2cae` |

**归档位置**（远端 `zhiyuan-127`，2026-10-06 二次确认）：

```
/mnt/share/user_homes/zyp/backup_unpushed_docs_20261005/collab/docs/community_talks/
├─ README.md                                 家族索引（未改动）
├─ deepseek_v41_flash/
│   ├─ LEARN.md                              69259 B  sha256 96eb84db7a0e…（修复定稿）
│   └─ REVIEW_deepseek_v41_flash.md           23730 B  sha256 f032f18982db…（本报告）
└─ deepseek_v4_flash_0731/ glm52/ glm53_flash/ qwen38_flash_next/ qwen38_27b/   原有 8 个族文件哈希未变
```

`/workspace/vllm_iluvatar/docs/community_talks/`（共享盘副本 `/mnt/share/user_homes/zyp/vllm_iluvatar/docs/community_talks/`）**已整体删除**，该仓库不再有 `community_talks`；删除前逐文件核对，目录内只有本次这两个文件，其余 6 个模型目录均为空壳，因此无内容丢失。备份 `MANIFEST.md` 已同步更新：新增一条 2026-10-06（二次）记录，并把 `deepseek_v41_flash/LEARN.md` 行的 sha256/字节数刷新为定稿值、补上报告行。

未做的两件事（需要时再说）：① `docs/community_talks/` 此前一直是**未跟踪**目录，本轮同样**没有提交**任何 git 变更；② 仓库里其他文档（`docs/adaptation_records/` 等）未动。

---

## 一句话总结

**这份讲义可以直接发：417 条断言核验下来没有一处事实性错误，162 处源码引用抽样命中率 100%，符号零编造。评审发现的 1 条需修订项（基线未锁 commit，行号会漂移）与 8 条精度项已按最小 diff 修复（10 个 hunk / 12 处替换，其余字节零改动），修复定稿已归档到 `docs/community_talks/deepseek_v41_flash/`。**
