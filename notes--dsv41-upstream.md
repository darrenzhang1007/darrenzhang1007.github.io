# DeepSeek-V4.1-Flash：上游 vLLM（NVIDIA）实现

本文介绍 DeepSeek-V4.1-Flash 在**上游官方 vLLM `v0.32.0`（NVIDIA 路径）**上的实现。

**本文结构（自顶向下）**：第 0 章模型架构概览 → 第 1 章基于 `config.json` 梳理 40 层配置 → 第 2 章从上层走通一次完整 `forward`（各子模块先视作黑盒）→ 第 3 章展开 mHC / Attention / MoE / Engram / DSpark 模块内部实现 → 第 4、5 章介绍量化与算子（kernel）分派 → 第 6 章提供文件与符号速查表。定位算子位置，可直接查看 [6.1 节](#61-模块--文件) 和 [5.1 节](#51-组件--kernel-总表)。

## 基线

| 项 | 值 |
|---|---|
| 源码基线 | 官方 `vllm-project/vllm`，**`v0.32.0`**，commit `5df63a7203`（2026-10-17，分支 `releases/v0.32.0`） |
| 模型配置 | 官方 checkpoint [`deepseek-ai/DeepSeek-V4.1-Flash` 的 `config.json`](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash/file/view/master/config.json) |

本文中 `文件:行号` 的路径一律相对 `vllm/`，例如 `models/deepseek_v41/attention.py:659` 指的是 `vllm/models/deepseek_v41/attention.py` 第 659 行。第 2、3 章的走查正文为了简洁，在上下文已经点明文件后会只写 `attention.py:659` 这样的短形式。

---

## 0. 架构概览

DeepSeek-V4.1-Flash 是一个 **40 层全 MoE 模型**（无稠密 FFN 层，`hidden_size=5120`），核心由以下 7 个部分组成（具体层号、配置参数与代码实现见第 1～3 章）：

- **MLA（多头潜在注意力，压缩通道维）**：普通 Transformer 每个注意力头各自保存一份 K 和 V；MLA 将所有头的 KV 压缩为一个共享的 **512 维潜在向量（latent）**，由 64 个 Query 头共享同一份 KV（细节见 [3.2.5 节](#325-mla-的形状两段低秩--分组输出)）。
- **Sparse MLA（SWA + Compressor + Indexer，减少序列长度维的计算量）**：仅靠 MLA 时，长上下文仍需遍历全部历史 KV；Sparse MLA 同时维护**近处滑窗缓存（SWA，保留最近 128 个 token）**与**压缩缓存（`Compressor` 实时压缩全量历史，`Indexer` 打分检索出最相关的 512 条记录）**，使注意力每次最多只读 `128 + 512 = 640` 条记录（细节见 [3.2.1 节](#321-双轨缓存与-compress_ratios-的三档语义)）。
- **跨层共享与两级候选（`source` 层复用）**：L2–L39 共 38 层全部执行上述 Sparse MLA，但只有少数 `source` 层真正挂载 `Compressor`（4 层）和 `Indexer`（8 层）去刷新压缩缓存与 top-512 索引，其余层直接复用前序层的计算结果；L20 还额外输出粗筛候选块供后续层做二级检索（细节见 [3.2.2 节](#322-跨层复用谁真的持有模块三张-source-层表) 与 [3.2.3 节](#323-两级候选在-nv-路径怎么落地)）。
- **四流超连接残差（4 流 mHC + 延迟融合）**：将普通 Transformer 的单条残差流扩展为 **4 条并行残差流 `[T, 4, 5120]`**，在进出子层（Attention / MoE）时做「4 流 ↔ 1 流」加权转换（`pre` 与 `post`），并通过算子融合将上一子层的 `post` 推迟到下一子层的 `pre` 一并执行以减少显存读写（细节见 [2.2 节](#22-主干循环每层交还一个六元组) 与 [3.1 节](#31-mhc从上层入口到-kernel-的调用链与内部计算)）。
- **MoE（混合专家）**：全 40 层均为 MoE（每层 **384 个路由专家选 top-6 + 1 个共享专家**），统一通过门控打分加偏置选专家（`noaux_tc`），取消了 V4.0 前 3 层的词表哈希路由（细节见 [3.3 节](#33-moe一个类同时服务主干和-draft)）。
- **Engram（N-gram 哈希记忆查表，解耦记忆与计算）**：用超大规模的 2/3/4-gram 词组哈希嵌入表（可直接卸载到主机 CPU 内存，按址 $O(1)$ 查表）直接为残差流补充局部词组搭配与静态知识，替代了 V4.0 前 3 层的单 token 哈希 MoE，节省早期层的 Attention 与 FFN 算力（细节见 [3.4 节](#34-engramn-gram-哈希查表与门控注入)）。
- **DSpark（半自回归投机解码草稿模型）**：挂载在主干之后的 3 层独立轻量草稿模型，融合主干末三层的隐状态后单步并行预测 5 个草稿 token（`block_size=5`）交回主干验证，替代了逐 token 串行递推的经典 MTP（细节见 [3.5 节](#35-dspark3-层半自回归投机草稿模型)）。

和 V4 标准版的关键差异，这几条最容易讲错：

| 维度 | V4 标准版 | **V4.1-Flash** |
|---|---|---|
| 层数 / hidden | 43 / 4096 | **40 / 5120** |
| 专家规模 | 256 / top-6 | **384 / top-6** |
| MoE 路由 | 前 3 层 `tid2eid` 哈希 + 40 层 `noaux_tc` | **全 40 层 `noaux_tc`** |
| 投机解码 | 经典 MTP | **DSpark**（不通用） |

---

## 1. 先建立模型轮廓：config 怎么决定 40 层

### 1.1 config 怎样被读进来

字段都在 `text_config` 下，但 `DeepseekV41Config.__init__` **把 `text_config` 的字段全部摊平到顶层**，所以模型代码里是 `config.hidden_size` 直读。同一处还把 `rope_scaling` 改名成 `rope_parameters`（Transformers v5 里 `rope_scaling` 变成了属性），并把 `quantization_config.expert_dtype` 提到顶层。

`num_hash_layers` 不在 `config.json` 里，由 `nvidia/model.py` 构建 MoE 时硬编码为 `0`——这就是「全 40 层 `noaux_tc`」的源码依据（落点见 [3.3 节](#33-moe一个类同时服务主干和-draft)）。

### 1.2 关键字段

| 字段 | 值 | 含义 |
|---|---|---|
| `hidden_size` | 5120 | 单流宽度；mHC 四流总宽是 `4 × 5120 = 20480`，但张量始终保持 3 维 `[T,4,5120]`，**不展平** |
| `num_attention_heads` / `head_dim` | 64 / 512 | MLA 的 Q 头数与每头宽度（512 = 448 维内容 NoPE + 64 维位置 RoPE） |
| `num_key_value_heads` | 1 | KV 压成单头 latent，由 64 个 Q 头共享 |
| `q_lora_rank` / `o_lora_rank` / `o_groups` | 1280 / 1024 / 8 | Q 投影与输出 O 投影都做两段低秩分解（先压到低维再升回高维，省参数和算力），其中 O 投影分 8 组做 |
| `qk_rope_head_dim` | 64 | 带旋转位置编码（RoPE）的维度；其余 448 维不加位置编码（NoPE） |
| `sliding_window` | 128 | 近处滑窗（SWA）大小：最近 128 个 token 永远原样保留 |
| `compress_ratios` | 43 项（`0`×2, `2`×18, `1`×20, 末尾 `0`×3） | 每层压缩档位：主干 L0–L1 为 `0`（纯滑窗）、L2–L19 为 `2`（2:1 压缩）、L20–L39 为 `1`（1:1 压缩），末尾 3 个 `0` 对应 DSpark draft（详见 [3.2.1 节](#321-双轨缓存与-compress_ratios-的三档语义)） |
| `kv_source_layer_ids` | `[2, 8, 14, 20]` | 真正实例化 `Compressor` 并写入压缩 KV cache 的 4 个 `source` 层，其余层复用（详见 [3.2.2 节](#322-跨层复用谁真的持有模块三张-source-层表)） |
| `index_source_layer_ids` | `[2, 8, 14, 20, 24, 28, 32, 36]` | 真正实例化 `Indexer` 并刷新 top-512 索引的 8 个 `source` 层，其余层复用（详见 [3.2.2 节](#322-跨层复用谁真的持有模块三张-source-层表)） |
| `index_n_heads` / `index_head_dim` / `index_topk` | 32 / 128 / 512 | Indexer 检索器的打分头数、头宽，以及每轮从压缩缓存检索出的位置数（top-512） |
| `candidate_source_layer_id` / `candidate_topk_blocks` / `candidate_block_size` | 20 / 2048 / 8 | L20 先粗筛出 2048 个候选块（每块含 8 个压缩位置），供后面层的 Indexer 做二次精选（详见 [3.2.3 节](#323-两级候选在-nv-路径怎么落地)） |
| `hc_mult` / `hc_sinkhorn_iters` / `hc_eps` | 4 / 20 / 1e-6 | mHC 的残差流数（4 条）、Sinkhorn 归一化迭代次数（把 `4×4` 流混合矩阵反复行/列归一防止数值爆炸）、RMSNorm 的 eps |
| `n_routed_experts` / `num_experts_per_tok` / `n_shared_experts` | 384 / 6 / 1 | 每层 384 个路由专家里挑 top-6，外加 1 个固定激活的共享专家 |
| `moe_intermediate_size` / `swiglu_limit` | 2304 / 10.0 | 每个专家 FFN 的中间层宽度，以及 SwiGLU 激活值的截断上限 |
| `scoring_func` / `topk_method` / `routed_scaling_factor` | `sqrtsoftplus` / `noaux_tc` / 1.5 | 路由门控的打分激活函数、无辅助损失偏置选 top-k 方法、路由专家输出求和后的放大倍数 |
| `engram_layer_ids` | [1, 14] | 挂载 Engram 词组哈希查表模块的层号 |
| `engram_max_ngram_size` / `engram_n_heads` | 4 / 8 | 统计 2/3/4-gram 共 3 种词长 × 8 个哈希头 = 每个位置查 **24 个哈希列** |
| `num_nextn_predict_layers` / `dspark_block_size` | 3 / 5 | DSpark 投机草稿模型共 3 层，单步预测 5 个草稿 token |
| `dspark_n_routed_experts` / `dspark_num_experts_per_tok` | 128 / 3 | **draft 层的 MoE 规模小于主干**（128 个专家选 top-3） |
| `dspark_target_layer_ids` | [37, 38, 39] | DSpark 草稿模型从主干采集隐状态的层号 |

24 个哈希列的公式在 `common/engram.py`：`n_hash_cols = (max_ngram_size - 1) * n_heads`（`EngramLayout`），配 `4` 和 `8` 得 24。源码注释也写明「每个位置按 `max_ngram_size - 1` 个 n-gram 哈希（2-gram 到 max）」。

---

## 2. 先从上层走通一次 forward

本章按实际调用顺序梳理主干的一次完整 `forward` 流程，并给出对应的核心代码片段与行号。**mHC 算子、Attention、MoE 与 Engram 的内部实现均在第 3 章展开**，本章仅关注它们的输入输出与层间状态传递。

### 2.1 对象怎样嵌套，以及进第 0 层前先做了哪几步

先看从顶层入口到单层内部的完整模块树：

```text
DeepseekV41ForCausalLM                   nvidia/vl_model.py:117（checkpoint 的 architectures）
└─ language_model                        DeepseekV41LLMForCausalLM，nvidia/model.py:1537
   ├─ model                              DeepseekV4Model，nvidia/model.py:662
   │  ├─ embed_tokens                    VocabParallelEmbedding，vocab 129280 → [T,5120]（:726）
   │  ├─ engram_hash                     NgramHashState，整批算一次 24 列 n-gram 哈希（:859）
   │  ├─ engram_batch                    EngramBatch，进层前一次性把 L1/L14 的表项查好（:780）
   │  ├─ layers[0..39]                   DeepseekV4DecoderLayer，nvidia/model.py:283
   │  │  ├─ engram                       Engram（仅 L1、L14 不为 None，其余 38 层为 None，:305）
   │  │  ├─ hc_attn_fn / hc_ffn_fn       mHC 混合系数投影 + attn_norm / ffn_norm
   │  │  ├─ attn                         DeepseekV4FlashMLAAttention 等（:320）
   │  │  └─ ffn                          DeepseekV4MoE（:333）
   │  └─ norm                            末层 final RMSNorm(5120)（:865）
   └─ lm_head                            ParallelLMHead，5120 → 129280（:1574）
```

顶层 `forward` 的调用链为：`DeepseekV41ForCausalLM.forward`（`nvidia/vl_model.py:289`）→ `DeepseekV41LLMForCausalLM.forward`（`nvidia/model.py:1639`）→ **`DeepseekV4Model.forward`（`nvidia/model.py:912`）**。

在进入 40 层循环之前，`DeepseekV4Model.forward`（`nvidia/model.py:912-1005`）先完成以下预处理：

1. **查词嵌入（`embed_tokens`）**：`hidden_states = self.embed_input_ids(input_ids)`（`:924`），得到 `[T, 5120]` 的初始二维激活。
2. **计算全批 Engram n-gram 哈希（`engram_hash`）**：在进入 Decoder 层及 Sequence Parallel 切分之前，调用 `self.engram_hash(...)`（`:966`）根据 `input_ids` 和前文 `lookback_token_ids` 计算整批 token 的 24 列哈希 ID `engram_hashes` 与 `engram_mask`（多模态场景下图像占位 token 对应的 mask 为 `False`，用于阻断跨图像边界的 n-gram，`:954-955`）。
3. **批量预查 L1 / L14 的 Engram 表项（`engram_batch.prepare_embeddings`）**：调用 `self.engram_batch.prepare_embeddings(engram_hashes)`（`:983`），在进入第 0 层前一次性查出 L1 和 L14 所需的嵌入向量（若开启 CPU offload，则在 `engram_prefetch_stream` 上发起异步预取）。后续执行到 L1 和 L14 的 `layer.engram(...)` 时直接消费该结果做门控注入。
4. （可选）若开启 Sequence Parallel（SP），在 `:1003-1004` 将 `hidden_states` 和 `input_ids` 按 token 维切分（`sp_shard`），随后进入 40 层循环。

### 2.2 主干循环：每层交还一个六元组

完成上述预处理后，`DeepseekV4Model.forward` 在 `nvidia/model.py:1006-1024` 初始化 mHC 状态并调用 `self._run_layers`；而 `_run_layers` 的定义位于同文件 `nvidia/model.py:1093-1136`：

```python
# 1. 调用入口：DeepseekV4Model.forward（nvidia/model.py:1006-1024）
residual, post_mix, res_mix = None, None, None
pre_mix: torch.Tensor | None = None
if not get_pp_group().is_first_rank:
    pre_mix = intermediate_tensors["pre_mix"]
aux_hidden_by_layer: dict[int, torch.Tensor] = {}
hidden_states, residual, post_mix, res_mix, pre_mix = self._run_layers(
    range(self.start_layer, self.decoder_replay_start),
    hidden_states, positions, input_ids,
    pre_mix, post_mix, res_mix, residual,
    aux_hidden_by_layer, engram_hashes, engram_mask,
)

# 2. 循环定义：DeepseekV4Model._run_layers（nvidia/model.py:1093-1136）
for idx in layer_ids:
    hidden_states, residual, post_mix, res_mix, pre_mix, previous_aux = (
        self.layers[idx](
            hidden_states, positions, input_ids,
            pre_mix, post_mix, res_mix, residual,
            engram_hashes, engram_mask,
            capture_previous_aux=idx in self.aux_hidden_state_layers,
            mega_gate_metadata=mega_gate_metadata,
        )
    )
    if previous_aux is not None:
        # idx is the one-based id of the layer whose post this is.
        aux_hidden_by_layer[idx] = previous_aux
```

与普通 Transformer 循环（每层只传递 `hidden_states`）不同，这里每层进出都传递 **6 个张量**：前 5 个 `(hidden_states, residual, post_mix, res_mix, pre_mix)` 是 **mHC（四流超连接残差）** 的跨层接力状态，第 6 个 `previous_aux` 是供 DSpark 草稿模型使用的隐状态采集口。

从黑盒视角理解这 5 个跨层张量的作用，只需明确三点（从上层入口到底层 kernel 的调用链与数学公式见 [3.1 节](#31-mhc从上层入口到-kernel-的调用链与内部计算)）：

- **主干维护 4 条残差流，子层仅接收 1 条流**：`residual` 是 `[T, 4, 5120]` 的四流张量，而 Attention 和 MoE 的输入输出均为二维 `x [T, 5120]`。因此每次进出子层都需做「4 流 ↔ 1 流」转换：
  - **`pre`（进子层前，4 流 → 1 流）**：用权重 `pre_mix` 将 4 条残差流加权合并为 1 条 `x_in [T, 5120]` 并做 RMSNorm 送入子层；同时计算出本子层出栈要用的 `post_mix`、`res_mix` 以及下一子层要用的新 `pre_mix`。
  - **`post`（出子层后，1 流 → 4 流）**：子层输出单流 `x [T, 5120]` 后，用 `res_mix` 和 `post_mix` 将其合入 4 条残差流，更新出新的 `residual [T, 4, 5120]`。
- **为减少显存读写，将 `post` 推迟半步与下一子层的 `pre` 融合（Delayed mHC）**：若每个子层末尾都立即执行 `post` 将 `[T, 4, 5120]` 写回显存，下一子层开头的 `pre` 又需立即从显存读取该张量。因此 vLLM **不在子层末尾执行 `post`，而是将其推迟半步，在下一子层开头用融合算子 `mhc_shifted_post_pre` 一次性完成「上一子层的 post + 下一子层的 pre」**。
- **半步错位的结果**：每层末尾的 MoE 执行完毕后，其 `post` 尚未执行，因此需将 MoE 输出 `hidden_states` 连同待结清的 `residual / post_mix / res_mix / pre_mix` 一并返回主干，传入**下一层开头**完成更新。

下图对比了同一层内 **① 普通 Transformer（最左）**、**② 未延迟的正常 mHC（中间）** 与 **③ vLLM 实际执行的延迟 mHC（最右）** 的执行边界：

![第 i 层残差接力对比：普通 Transformer vs 正常 mHC vs 延迟 mHC](assets/png/dsv41_mhc_delayed_compare.png)

### 2.3 单层内部：四个入口分支与层末返回

进入 `self.layers[idx]`，即 `DeepseekV4DecoderLayer.forward`（`nvidia/model.py:455-625`）。

函数开头先确定本层是否启用 **mHC 辅助流 `mhc_stream`**（`nvidia/model.py:478-490`），随后按当前层的位置进入 **四条互斥的 mHC 入口分支**（`:494-586`）：

```python
# 0. 确定本层是否启用辅助流 mhc_stream（nvidia/model.py:478-490）
mhc_stream = self.mhc_stream
if mhc_stream is not None and (
    in_piecewise_cudagraph()
    or not 0 < positions.shape[0] <= MHC_OVERLAP_MAX_TOKENS
    or not torch.cuda.is_current_stream_capturing()
):
    mhc_stream = None
mhc_pre = (
    partial(mhc_pre_delayed_overlap, stream=mhc_stream)
    if mhc_stream is not None
    else mhc_pre_delayed_tilelang
)

# 1. 四条互斥的层入口分支（nvidia/model.py:494-586）
if residual is None:
    if x.dim() == 2:
        # 分支 A 首层（:496-514）：embedding 广播成四流
        residual = x.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()
        post_mix, res_mix, x, attn_pre = mhc_pre(
            residual, self.hc_attn_fn_broadcast, ...)
    else:
        # 分支 B PP 非首 rank（:515-530）：接到的已经是四流
        residual = x
        post_mix, res_mix, x, attn_pre = mhc_pre(residual, self.hc_attn_fn, ..., pre_mix=pre_mix)
elif self.engram is not None and engram_hashes is not None:
    # 分支 C Engram 层 L1/L14（:531-561）
    previous_post = mhc_post_tilelang(x, residual, post_mix, res_mix)
    residual = self.engram(previous_post, engram_hashes[:, self.engram.layer_hash_index], engram_mask)
    post_mix, res_mix, x, attn_pre = mhc_pre(residual, self.hc_attn_fn, ..., pre_mix=pre_mix)
else:
    # 分支 D 普通层（:562-586）：post 与 pre 融在一个核里
    residual, post_mix, res_mix, x, attn_pre, aux = mhc_shifted_post_pre(
        x, residual, post_mix, res_mix, self.hc_attn_fn, ...,
        stream=mhc_stream, reduce_results=self.fuse_mhc_all_reduce)
```

- **分支 A（第 0 层，`:496-514`）**：使用专用权重 `hc_attn_fn_broadcast`。第 0 层没有前一层遗留的 `post` 和 `pre_mix`，需先将词嵌入 `x [T, 5120]` 广播为 4 条相同的流 `[T, 4, 5120]`，再计算首组混合系数。
- **分支 B（流水线并行 PP 的非首卡，`:515-530`）**：前一张卡在层边界切分时已完成 `post` 并传入四流 `residual`，因此直接调用 `mhc_pre`。
- **分支 D（普通层，`:562-586`）**：调用融合算子 ① `mhc_shifted_post_pre`（权重为 `hc_attn_fn`），在单个 kernel 内先完成上一层 MoE 的 `post`（得到上图中的 `residual₀`），再完成本层 Attention 前的 `pre`（输出二维 `x` 与 `attn_pre`）。
- **分支 C（挂载 Engram 的 L1 和 L14，`:531-561`）是延迟融合的唯一例外**：Engram 的查表结果需注入到「上一层刚完成 `post` 的四流残差」上，使本层计算的混合系数能反映注入后的残差。因此这两层不能走融合算子，而是拆为三步：先调 `mhc_post_tilelang` 完成上一层 MoE 的 `post` → 调 `self.engram(...)` 注入词组向量 → 再调 `mhc_pre` 执行本层 Attention 前的 `pre`。

完成入口分支得到二维 `x [T, 5120]` 后，单层后半段（`nvidia/model.py:591-625`）依次执行 Attention、融合算子 ② 与 MoE：

```python
x = self.attn(positions, x, None)                      # 1. Attention 子层（:591，内部见 3.2）

if mhc_stream is not None:                             # 等待入口 mhc 在辅助流上的计算（:597-598）
    torch.cuda.current_stream().wait_stream(mhc_stream)

residual, post_mix, res_mix, x, ffn_pre, _ = mhc_shifted_post_pre(
    x, residual, post_mix, res_mix, self.hc_ffn_fn, ...,
    pre_mix=attn_pre, norm_weight=self.ffn_norm.weight,
    stream=mhc_stream, reduce_results=self.fuse_mhc_all_reduce,
)                                                      # 2. 合入 Attn 的 post，并用 attn_pre 做 MoE 的 pre（:599-617）

if self.fuse_mhc_all_reduce and self.ffn.defers_finalize(x.shape[0]):
    x = self.ffn.forward_unfinalized(x, input_ids)     # 3a. 延迟归约版 MoE（:618-620，官方配置不触发）
else:
    x = self.ffn(x, input_ids, mega_gate_metadata)     # 3b. 常规 MoE 子层（:621-622，内部见 3.3）

if mhc_stream is not None:                             # 等待第 2 步在辅助流上的计算（:623-624）
    torch.cuda.current_stream().wait_stream(mhc_stream)

return x, residual, post_mix, res_mix, ffn_pre, previous_aux  # （:625）
```

后半段代码中包含两处条件分支：

- **`if mhc_stream is not None`（`:478-490`、`:597-598`、`:623-624`）——mHC 辅助流重叠**：当启用完整 CUDA Graph（`not in_piecewise_cudagraph()` 且处于捕获态）且 token 数落在 `(0, MHC_OVERLAP_MAX_TOKENS]` 时，第 0 步会保留 `self.mhc_stream`，将 mHC 计算混合系数的小矩阵乘放到辅助流上与主流并行执行，并在 Attention 之后（`:598`）及层末返回前（`:624`）各执行一次 `wait_stream` 同步。
- **`if self.fuse_mhc_all_reduce and self.ffn.defers_finalize(x.shape[0])`（`:618-622`）——MoE 延迟归约**：满足条件时，MoE 调用 `forward_unfinalized` 返回未完成归约的 `MoEOutput`，将专家输出归约推迟到下一层入口的 `mhc_shifted_post_pre`（`:583` 的 `reduce_results=self.fuse_mhc_all_reduce`）中合并执行。但在**官方 checkpoint 配置下 `defers_finalize` 恒为 `False`，始终走 `else` 分支的常规 `self.ffn(...)`**（原因见 [3.3 节](#33-moe一个类同时服务主干和-draft)）。

对照 `:591-625`，层末 `:625` 返回给主干循环的 **6 个返回值**来源与用途如下——**前 5 个张量中，仅第 1 个 `x` 是本层 MoE（`self.ffn`）的直接输出，其余 4 个均在第 2 步「进 MoE 之前」的 `mhc_shifted_post_pre`（`:599`）中计算生成，跨过 MoE 原样传递至下一层**：

| 层末 `return` 项 | 下一层接收的形参名 | 形状 | 本层生成位置 | 传入下一层的用途 |
|---|---|---|---|---|
| **`x`** | `x`（主干中为 `hidden_states`） | `[T, 5120]` | **后半段第 3 步（`:622`）：本层 MoE（`self.ffn`）的直接输出** | 供下一层入口执行 MoE 的 `post`（合入 4 流） |
| **`residual`** | `residual` | `[T, 4, 5120]` | 后半段第 2 步（`:599`）：**进 MoE 前**的 `mhc_shifted_post_pre` | 本层合入 Attention 后的 4 流残差基座（即图中的 `residual₁`） |
| **`post_mix`** | `post_mix` | `[T, 4, 1]` | 后半段第 2 步（`:599`）：**进 MoE 前**的 `mhc_shifted_post_pre` | 下一层入口执行 `post` 时对 `x` 做 4 路加权缩放 |
| **`res_mix`** | `res_mix` | `[T, 4, 4]` | 后半段第 2 步（`:599`）：**进 MoE 前**的 `mhc_shifted_post_pre` | 下一层入口执行 `post` 时对 `residual` 的 4 条流做 $4\times 4$ 混合 |
| **`ffn_pre`** | `pre_mix`（层内层外命名不同） | `[T, 4]` | 后半段第 2 步（`:599`）：**进 MoE 前**的 `mhc_shifted_post_pre` | 下一层入口算完 `post` 得到 `residual₀` 后，用它将 4 流合并为 1 流送入下一层 Attention |
| **`previous_aux`** | （主干存入 `aux_hidden_by_layer[idx]`） | `[T, 5120]` | **前半段层入口（`:565` / `:542`）**：分支 D 或分支 C | 因第 `idx-1` 层末尾的 `post` 推迟至第 `idx` 层入口完成，此处取出的 `aux` 对应**上一层完整结束后的隐状态**（供 DSpark 采集 `dspark_target_layer_ids = [37, 38, 39]`） |

因此，**MoE 本身只是一个 `[T, 5120] → [T, 5120]` 的二维变换模块，不感知 4 条残差流与 `mix` 系数**。

### 2.4 后半段分流（`decoder_replay_layers`）与末层四流合并（`_collapse` / `hc_collapse_triton`）

回到 `DeepseekV4Model.forward`（`nvidia/model.py:1012-1077`），紧接在第一次 `self._run_layers` 之后，代码并不是直接调 `_collapse`，而是先经过一个 `if self.decoder_replay_layers is not None:` 分支（`:1026-1045`）：

```python
# 1. 外层入口：DeepseekV4Model.forward（nvidia/model.py:1012-1045）
#    先跑前半段 range(0, 20)，即 L0 ~ L19（若未启用 replay，则 decoder_replay_start=40，直接跑满 40 层）
hidden_states, residual, post_mix, res_mix, pre_mix = self._run_layers(
    range(self.start_layer, self.decoder_replay_start),   # range(0, 20) → L0 ~ L19
    hidden_states, positions, input_ids,
    pre_mix, post_mix, res_mix, residual,
    aux_hidden_by_layer, engram_hashes, engram_mask,
)
late_aux: list[torch.Tensor] = []
if self.decoder_replay_layers is not None:
    # 紧接着在这里进入 _run_replay_layers，继续跑完后半段 range(20, 40)，即 L20 ~ L39！
    hidden_states, pre_mix, *late_aux = self.decoder_replay_layers(
        hidden_states, positions, input_ids,
        pre_mix, post_mix, res_mix, residual,
    )
else:
    hidden_states = self._collapse(
        hidden_states, residual, post_mix, res_mix,
        aux_hidden_by_layer, full_num_tokens,
    )

# 2. self.decoder_replay_layers 内部调用的 _run_replay_layers（nvidia/model.py:1160-1191）
def _run_replay_layers(self, hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual):
    aux_hidden_by_layer: dict[int, torch.Tensor] = {}
    # 第二次调用 _run_layers：跑完后半段 range(20, 40)，即 L20 ~ L39
    hidden_states, residual, post_mix, res_mix, pre_mix = self._run_layers(
        range(self.decoder_replay_start, self.end_layer), # range(20, 40) → L20 ~ L39
        hidden_states, positions, input_ids,
        pre_mix, post_mix, res_mix, residual,
        aux_hidden_by_layer,
    )
    hidden_states = self._collapse(
        hidden_states, residual, post_mix, res_mix,
        aux_hidden_by_layer, positions.shape[0],
    )
    return hidden_states, pre_mix, ...
```

先看清 **`self.decoder_replay_start` 与 `self.decoder_replay_layers`（`:1026`）在干什么**（初始化见 `nvidia/model.py:800-839`，实现见 `models/deepseek_v41/decoder_replay_layers.py:40`）：

- **注意：不是只跑前 20 层！无论 Prefill 还是 Decode，都会跑满全部 40 层；区别在于长 Prefill 时，后 20 层（`L20–L39`）只对每个请求末尾的 `128` 个 token 执行计算。**
- **为什么长 Prefill（例如输入 `#0 ~ #9999` 共 10,000 个 token）在 L20 之后可以扔掉前 9,872 个 token？**
  1. **L21–L39 不再写压缩 KV cache**：在 `DeepseekV4Model.__init__`（`:805`）中，`cut = max(config.kv_source_layer_ids) = 20`——**L20 是全模型最后一个写压缩 KV cache 的层**，L21–L39 全部共用 L20 写好的压缩 KV（见 [3.2.2 节](#322-跨层复用谁真的持有模块三张-source-层表)）。只要 **L20** 把 `#0 ~ #9999` 全部 10,000 个 token 的压缩 KV 写进 cache，全量压缩历史就已经构建完毕。
  2. **L21–L39 的本层 SWA 滑窗只留最后 128 个 token**：L21–L39 自己只维护容量为 `sliding_window = 128` 的滑窗 cache。对于 10,000 个 token 的 Prompt，前 9,872 个 token（`#0 ~ #9871`）即使在 L21–L39 算出 SWA KV，也会滑出窗口被丢弃；而末尾的 `#9872 ~ #9999` 在 L21–L39 做注意力时，远处历史读的是 L20 写好的压缩 KV，近处滑窗也只互相关注 `#9872 ~ #9999` 这 128 个位置内部——**前 9,872 个 token 在 L20 之后的隐状态对最终输出（`#9999` 的 logits）及后续 decode 的 cache 均无任何影响**。
- **两段式执行流程（对应 `:830-838` 的日志说明）**：

```text
输入长 Prompt：10,000 个 token（#0 ~ #9999）
  │
  ▼
【第一段：L0 ~ L19（前 20 层）】self._run_layers(range(0, 20), ...)（:1012）
  • 全部 10,000 个 token（#0 ~ #9999）完整跑完 L0 ~ L19
  │
  ▼
【第二段：L20 ~ L39（后 20 层）】self.decoder_replay_layers(...)（:1026）
  ├─ 长 Prefill 步（启用裁剪子批 replay_batch）：
  │    1) 先调 self.layers[20].write_kv(...)（:627），替 L20 把全部 10,000 个 token 的压缩 KV、Indexer K 与 SWA KV 写入 cache
  │    2) 丢弃前 9,872 个 token（#0 ~ #9871），将 batch 裁剪为仅剩末尾 128 个 token（#9872 ~ #9999）
  │    3) 调 self._run_replay_layers（:1160）→ self._run_layers(range(20, 40), ...)，仅对这 128 个 token 跑完 L20 ~ L39
  │
  └─ 普通 Decode 步（或长度 ≤ 128 的短 Prefill，此时 replay_batch is None）：
       • 完全没有两阶段裁剪！uses_replay_batch() 为 False（decoder_replay_layers.py:73, 112），既不调 write_kv 也不裁 token，直接原样调 _run_layers(range(20, 40)) 接着跑完后 20 层，与一口气跑 range(0, 40) 完全等价。
```

- **什么条件控制 `self.decoder_replay_layers` 是否启用（即是不是 `None`）？**
  它由 `DeepseekV4Model.__init__`（`nvidia/model.py:806-810`）中的三个条件共同决定，**只要有一条不满足，`self.decoder_replay_layers` 就是 `None`（此时 `self.decoder_replay_start = 40`）**：
  1. **`self.layers[cut].attn.swa_cache_layer.bounded_replay` 为 `True`**（`attention.py:529-567`）：
     - 总开关是 `CacheConfig.swa_bounded_replay`（`config/cache.py:250`，**默认为 `True`**）；
     - 必须使用 **Model Runner V2**（`vllm_config.use_v2_model_runner` 为 `True`）；
     - 未开启 Prefill Context Parallel（`prefill_context_parallel_size == 1`）且非 ROCm 平台。
  2. **`self._decoder_replay_supported(vllm_config, cut)` 返回 `True`**（`nvidia/model.py:1202-1240`）：
     - **未开启 Sequence Parallel（SP）**（`not self.use_sequence_parallel`）、未开启 `prefill_context_parallel_size > 1`、未开启微批（`not use_ubatching`）；
     - **PP 切分未把 `L20–L39` 拆到不同卡上**（持有 L20 的 PP stage 必须同时持有 L20 之后的全部层：`self.start_layer <= 20` 且 `self.end_layer == 40`）；
     - `cut`（L20）及之后没有挂载 Engram 层（官方配置 Engram 在 L1 和 L14，满足）；
     - 若挂了投机草稿模型（DSpark），草稿模型的滑窗不能超过主干的 128 窗口（官方配置相等，满足）。
  3. **`cut < self.end_layer - 1`**：`cut = max(config.kv_source_layer_ids) = 20 < 39`，恒满足。
  - 此外，即使 `self.decoder_replay_layers` 不为 `None`，在每个 step 运行时是否真正裁剪（`uses_replay_batch()`），还由 `DeepseekV41ModelState`（`nvidia/model_state.py:220-276`）按当步请求动态决定：只有当某条 Prefill 请求的 token 数超过窗口 `128`、且该请求没有开启 `prompt_logprobs`（开启 `prompt_logprobs` 需要所有 prompt 位置的 logits，不能丢弃前文）时，才会构造 `replay_batch` 触发裁剪。

- **为什么上面代码里有两处 `self._collapse`（`:1037` 与 `:1184`）？它们是 `if / else` 互斥的，同一次 `forward` 只会执行其中一处**：
  - **当走 `:1026` 的 `if` 分支时**（默认启用 `decoder_replay_layers`）：`else` 分支不会执行。之所以把 `self._collapse` 放在 `_run_replay_layers` 内部（`:1184`）而不是提到 `if / else` 外面，是因为长 Prefill 裁剪时，`decoder_replay_layers._run`（`decoder_replay_layers.py:115-130`）会先把 10,000 行 `index_select` 裁剪成 **128 行**送入 `_run_replay_layers`，算完后再用 `index_copy_` 把结果填回 10,000 行的零张量——在填回 10,000 行**之前**先对这 128 行调 `_collapse`，既只需对 128 行算 `mhc_post_tilelang`，又当场消费掉了 `post_mix / res_mix`，无需再把它们填回 10,000 行往外传。
  - **当走 `:1036` 的 `else` 分支时**（如开启了 SP 时 `decoder_replay_layers is None`，`decoder_replay_start = 40`）：不会调用 `_run_replay_layers`；`:1012` 一次跑完 `range(0, 40)` 后，直接在 `:1037` 调那一次 `self._collapse`。

最后看第 39 层跑完后的**末层收尾（把 4 条流加权合并回 1 条流）**。

主干循环内部一直维护着 4 条并行的残差流 `[T, 4, 5120]`，但最后的 `self.norm`（`RMSNorm(5120)`）和 `self.lm_head`（`5120 → 129280`）只接收标准的二维单流张量 `[T, 5120]`；因此在跑完第 39 层后，必须把 `[T, 4, 5120]` 的 4 条流加权求和合并成 1 条 `[T, 5120]`（源码函数名里用 `collapse` 指代这次四流合并）。

注意代码里有两个名字都带 `collapse` 的函数，它们前后脚配合完成最后三步收尾（`nvidia/model.py:1058-1077`、`:1138-1158`、`:1609-1614`）：

```python
# 第 1 步：self._collapse（nvidia/model.py:1138-1158）
#   注意：虽然函数名叫 _collapse，但它只负责替最后一层（L39）补做欠着的 MoE post，
#   返回的 hidden_states 仍然是 4 条流 [T, 4, 5120]！
hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)  # [T, 4, 5120]
if self.end_layer in self.aux_hidden_state_layers:
    final_aux = hidden_states.mean(dim=1)
    aux_hidden_by_layer[self.end_layer] = final_aux

# 第 2 步：hc_collapse_triton + self.norm（nvidia/model.py:1070-1071）
#   这一步才真正把 4 条流 [T, 4, 5120] 按末层的 pre_mix [T, 4] 加权求和合并成 1 条流 [T, 5120]，
#   等价于：(hidden_states.float() * pre_mix.unsqueeze(-1)).sum(dim=1).to(bfloat16)
hidden_states = hc_collapse_triton(hidden_states, pre_mix)                     # [T, 4, 5120] → [T, 5120]
hidden_states = self.norm(hidden_states)                                       # [T, 5120]

# 第 3 步：DeepseekV41LLMForCausalLM.compute_logits（nvidia/model.py:1609-1614）：投影到词表
logits = self.logits_processor(self.lm_head, hidden_states)                    # [n_logits, 129280]
```

**`self.norm` 与 `self.lm_head` 看到的维度始终是 5120，不是 20480**——四条流只在 mHC 内部存在，并且末端没有额外可学习的 `hc_head` 权重，四流合并算子 `hc_collapse_triton`（`model_executor/kernels/mhc/triton.py:108`）就是直接拿第 39 层留下的 `pre_mix` 对 4 条流做一次 FP32 加权求和：

$$\text{out}[t, :] = \sum_{m=0}^{3} \text{pre\_mix}[t, m] \cdot \text{hidden\_states}[t, m, :]$$

（若开启了投机解码，在 `:1059-1064` 执行 `hc_collapse_triton` 合并之前，还会先把未合并的四流状态 `[T, 4, 5120]` 展平为 `[T, 20480]` 拷入 `self._mtp_hidden_buffer`。）

---

## 3. 深入模块内部：mHC、Attention、MoE、Engram 与 DSpark

第 2 章把各子模块当黑盒走通了主干。这一章平级展开它们的内部实现。

### 3.1 mHC：从上层入口到 kernel 的调用链与内部计算

#### 3.1.1 上层入口与调用链：从 `DecoderLayer` 怎么一路调到底层 kernel

先看完整的三级调用链——`DeepseekV4DecoderLayer.forward`（`nvidia/model.py`）并不直接调底层 CUDA/TileLang 算子，而是先调 **`models/deepseek_v41/nvidia/ops/mhc.py`** 里的上层分派入口，再由它分发到 `mega_mhc.py` 或 `model_executor/kernels/mhc/tilelang.py`：

```text
DeepseekV4DecoderLayer.forward（models/deepseek_v41/nvidia/model.py:455-625）
├─ 1. 普通层入口（:565）与进 MoE 前（:599）：调 mhc_shifted_post_pre（nvidia/ops/mhc.py:191）
│     ├─ [可选前置] 若开启 fuse_mhc_all_reduce（:220-255）：
│     │     └─ _all_reduce_mhc（nvidia/ops/cute_dsl.py:AllReduceMHC，把 TP all-reduce 与 post+pre_mix 四流合并融为一核）
│     ├─ 分支 ① 若传了辅助流 stream is not None（FULL cudagraph + token 数 ≤ 16，:256-281）：
│     │     ├─ mhc_post_tilelang（kernels/mhc/tilelang.py:791）→ _MHC_POST_TILELANG_KERNEL
│     │     └─ mhc_pre_delayed_overlap（nvidia/ops/mhc.py:98）
│     │           ├─ 主流：MHC_PRE_NORM_KERNEL(split_mode="input") 先合并出当前子层输入 x
│     │           └─ 辅助流 mhc_stream：tf32_hc_prenorm_gemm + MHC_PRE_NORM_KERNEL(split_mode="stats") 算混合系数
│     ├─ 分支 ② 若 can_use_mega_mhc(...) 为真（SM100 + DeepGEMM + not capture_aux，:284-303）：
│     │     └─ mhc_shifted_post_pre_deep_gemm（nvidia/ops/mega_mhc.py:47）
│     │           └─ deep_gemm.mega_mhc（utils/deep_gemm.py，单核包揽 post + 24维投影 + Sinkhorn + 四流合并 + RMSNorm）
│     └─ 分支 ③ 其余情况兜底（:305-323）：
│           └─ mhc_fused_post_pre_delayed_tilelang（kernels/mhc/tilelang.py:269）
│                 ├─ 小 batch（:386-397）：_MHC_FUSED_TILELANG_KERNEL（寄存器内连做 post + 24维投影 GEMM）
│                 ├─ 大 batch（:398-432）：_MHC_POST_TILELANG_KERNEL + tf32_hc_prenorm_gemm
│                 └─ 尾部（:442-497）：MHC_PRE_NORM_KERNEL（Triton 核：Sinkhorn + pre_mix 四流合并 + RMSNorm）
│
├─ 2. 首层（:501）/ PP 非首卡（:517）/ Engram 层（:548）单独做 pre：调 mhc_pre（:486-490 绑定）
│     ├─ 有 mhc_stream：mhc_pre_delayed_overlap（nvidia/ops/mhc.py:98，同上）
│     └─ 无 mhc_stream：mhc_pre_delayed_tilelang（kernels/mhc/tilelang.py:74）
│           └─ tf32_hc_prenorm_gemm（或 _HC_PRENORM_GEMM_TILELANG_KERNEL）+ MHC_PRE_NORM_KERNEL
│
└─ 3. Engram 层入口（:540）与末层 _collapse（:1141）单独做 post：
      └─ mhc_post_tilelang（kernels/mhc/tilelang.py:791）→ _MHC_POST_TILELANG_KERNEL
```

顺着这张调用树，直接看上层主入口 **`mhc_shifted_post_pre`（`nvidia/ops/mhc.py:191-323`）** 的分派代码就很清楚了：

```python
# models/deepseek_v41/nvidia/ops/mhc.py:191-323
def mhc_shifted_post_pre(x, residual, post_layer_mix, comb_res_mix, fn, ...,
                         pre_mix=None, norm_weight=None, capture_aux=False,
                         stream=None, reduce_results=False):
    # 前置：若开启 fuse_mhc_all_reduce（TP=4 + NVLink 多播），用 CuTe DSL 融合核顺带完成归约与 post（:220-255）
    ...
    # 分支 ①：传了辅助流 mhc_stream（FULL cudagraph 且 token 数 ≤ 16），拆成 post + 辅助流 overlap pre（:256-281）
    if stream is not None:
        if layer_input is None:
            residual = mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix)
        aux = residual.mean(dim=1) if capture_aux else residual.new_empty(0, residual.shape[-1])
        pre_outputs = mhc_pre_delayed_overlap(residual, fn, ..., pre_mix=pre_mix,
                                              norm_weight=norm_weight, stream=stream, layer_input=layer_input)
        return residual, *pre_outputs, aux

    # 分支 ②：满足 Mega-mHC 条件时，走 DeepGEMM 单核全融合（:284-303）
    if can_use_mega_mhc(x, residual, pre_mix, norm_weight, capture_aux):
        outputs = mhc_shifted_post_pre_deep_gemm(x, residual, pre_mix, post_layer_mix, comb_res_mix, fn, ...)
        return *outputs, x.new_empty(0, x.shape[1])

    # 分支 ③：兜底走 TileLang 融合路径（:305-323）
    return mhc_fused_post_pre_delayed_tilelang(x, residual, post_layer_mix, comb_res_mix, fn, ...,
                                               pre_mix=pre_mix, norm_weight=norm_weight, capture_aux=capture_aux)
```

其中决定走不走分支 ②（`mhc_shifted_post_pre_deep_gemm`）的判定函数在 `nvidia/ops/mega_mhc.py:17-45`：

```python
# models/deepseek_v41/nvidia/ops/mega_mhc.py:17-45
def is_mega_mhc_supported(hidden_size, hc_mult) -> bool:
    if not (is_deep_gemm_supported()
            and current_platform.is_device_capability_family(100)
            and hidden_size > 0 and hidden_size % 1024 == 0
            and hc_mult == 4):
        return False
    deep_gemm = _import_deep_gemm()
    return deep_gemm is not None and callable(getattr(deep_gemm, "mega_mhc", None))

def can_use_mega_mhc(x, residual, pre_mix, norm_weight, capture_aux) -> bool:
    return (pre_mix is not None and norm_weight is not None
            and not capture_aux
            and x.shape[0] <= 1 << 20
            and is_mega_mhc_supported(x.shape[1], residual.shape[1]))
```

- **静态条件官方配置全部满足**：`hidden_size = 5120` 是 1024 的 5 倍，`hc_mult = 4`，因此在 SM100 家族 + DeepGEMM 可用时满足尺寸约束。（注意两个参数的维度：`x` 为二维 `[T,5120]`，故 `x.shape[1]` 是 `hidden_size`；`residual` 为三维 `[T,4,5120]`，故 `residual.shape[1]` 是 `hc_mult`。）
- **动态条件 `not capture_aux` 会使采集隐状态的层回退到分支 ③（TileLang）**：需要为 DSpark 采集 `previous_aux` 的层边界（`dspark_target_layer_ids = [37, 38, 39]`）不满足 `not capture_aux`，会自动回退到分支 ③ 的 `mhc_fused_post_pre_delayed_tilelang`。因此单次 forward 中 40 层未必全部执行同一个底层 kernel。

#### 3.1.2 算子内部算了什么：`post` 与 `pre` 的数学逻辑

无论分派到 DeepGEMM 还是 TileLang，底层 kernel 的数学逻辑均等价于 `model_executor/kernels/mhc/torch.py` 中的纯 PyTorch 参考实现：

- **`post` 如何将单流输出 `x [T, 5120]` 合入 4 条流 `residual [T, 4, 5120]`**（对应 `_MHC_POST_TILELANG_KERNEL`；参考实现 `mhc_post_torch` 位于 `kernels/mhc/torch.py:135-147`）：
  ```python
  # 1. 4 条旧残差流先按 4×4 矩阵 res_mix 交叉加权（每条新流都是 4 条旧流的线性组合）
  mixed_residual = torch.einsum("...ij,...ih->...jh", res_mix, residual)     # [T, 4, 5120]
  # 2. 单流输出 x 按 4 个权重 post_mix 缩放为 4 份，分别加到上面 4 条流上
  residual_new = mixed_residual + post_mix * x.unsqueeze(-2)                 # [T, 4, 5120]
  ```
  第 $j$ 条残差流（$j = 0, 1, 2, 3$）的更新公式为：

  $$\text{residual}_{\text{new}}[:, j, :] = \underbrace{\sum_{i=0}^{3} \text{res\_mix}[:, i, j] \cdot \text{residual}[:, i, :]}_{\text{① 4 条旧流按 } 4\times 4 \text{ 矩阵做线性组合}} \;+\; \underbrace{\text{post\_mix}[:, j] \cdot x}_{\text{② 单流输出 } x \text{ 乘第 } j \text{ 个权重后相加}}$$

- **`pre` 如何合并出单流 `x_in` 并计算三个 `mix` 系数**（对应 `tf32_hc_prenorm_gemm` + `MHC_PRE_NORM_KERNEL`；参考实现 `mhc_pre_delayed_torch` 位于 `kernels/mhc/torch.py:6-45`）：
  1. **投影 GEMM（`tf32_hc_prenorm_gemm`）**：将更新后的 4 条流展平为 `[T, 20480]`，乘以本子层的投影权重 `fn`（形状 `[24, 20480]`，即 `hc_attn_fn` 或 `hc_ffn_fn`），同时计算每行的平方和 `sqrsum` 用于 RMS 归一化，每个 token 输出 **24 个数**（`4 + 4 + 16`）。
  2. **后处理与四流合并（`MHC_PRE_NORM_KERNEL`）**：
     - 前 **4 个数**经 `sigmoid` 与缩放偏置，生成留给**下一子层**使用的新 `pre_mix [T, 4]`；
     - 中间 **4 个数**经 `sigmoid` 并乘以 `hc_post_alpha=2.0`，生成本子层出栈使用的 `post_mix [T, 4, 1]`；
     - 最后 **16 个数**重排为 `4×4` 矩阵，先做 `softmax` 再执行 **20 轮 Sinkhorn 行列交替归一化**（使每行、每列之和接近 1，保证多流混合数值稳定），生成 `res_mix [T, 4, 4]`；
     - 同时用**上一子层传入的旧 `pre_mix [T, 4]`** 对 4 条流做加权求和，并乘以子层入口的 `norm_weight`（`attn_norm` 或 `ffn_norm`）完成 RMSNorm：
       ```python
       x_in = RMSNorm((pre_mix.unsqueeze(-1) * residual.float()).sum(dim=1), norm_weight)  # [T, 5120]
       ```

### 3.2 Attention：Sparse MLA 的组成、跨层复用与 forward 流程

进入 `DeepseekV4Attention`（`attention.py:176`）之前，先明确组成 Sparse MLA 的四个部分在代码中的职责——特别是 **SWA（近处滑窗），它没有独立权重，也不单独执行注意力计算，`forward` 为空函数**：

| 组成部分 | 代码属性（`attention.py`） | 是否有独立权重 / 是否计算注意力 |
|---|---|---|
| **1. `SWA`（近处滑窗）** | `self.swa_cache_layer`（`DeepseekV4SWACache`，`:557`；定义于 `v1/attention/backends/mla/sparse_swa.py:74`） | **无权重，不单独计算注意力**（`sparse_swa.py:142` 的 `def forward(self): ...` 为空）。仅作为向 vLLM 申请 128 窗口显存的缓存容器：`kv` 向量来自 MLA 主投影 `fused_wqa_wkv`，由 `_fused_qnorm_rope_kv_insert`（`:1011`）完成 RoPE 与量化后写入。 |
| **2. `Compressor`（压缩器）** | `self.compressor`（`DeepseekCompressor`，`:577`，仅 4 个 `kv_source` 层非 `None`） | **有权重（`fused_wkv_wgate` + `RMSNorm`）**。将全量历史 token 压缩为 512 维 latent，写入跨层共享的压缩 KV cache。 |
| **3. `Indexer`（检索打分器）** | `self.indexer`（`DeepseekV4Indexer`，`:462`，仅 8 个 `index_source` 层非 `None`） | **有权重（`wq_b` / `weights_proj` / `wk` / `k_norm`）**。对压缩区历史打分，选出 top-512 下标写入 `topk_indices_buffer`。 |
| **4. `MLA`（主注意力）** | `fused_wqa_wkv / wq_b / wo_a / wo_b` + `flash_mla_with_kvcache` / `flash_mla_sparse_fwd` | **唯一执行注意力计算的模块**。将 **SWA 的 128 条 KV** 与 **Indexer 选出的 512 条压缩 KV** 一并送入**同一个 MLA 算子**，单次完成统一的 softmax 注意力计算。 |

#### 3.2.1 双轨缓存与 `compress_ratios` 的三档语义

以生成第 10,000 个 token（历史下标 `#0 ~ #9999`，`pos=9999`）为例，先说明上面 SWA 与 Compressor 双轨缓存最容易误解的一点：

> **问：既然最近 128 个 token（`#9872 ~ #9999`）已经在 SWA 滑窗里了，那 `Compressor` 处理的只是滑窗外的 `#0 ~ #9871` 吗？**
>
> **答：不是。`Compressor` 处理的是从 `#0` 到 `#9999` 的全量 10,000 个 token，原因有三点：**
> 1. **必须在生成当步实时压缩**：每个 token 只有在经过模型的当前步才有 `hidden_states`；若等 128 步滑出窗口后再压缩，对应的 `hidden_states` 已不存在。因此每个 token 在生成当步会**同时**写入本层的 SWA cache 和（`kv_source` 层的）压缩 KV cache。
> 2. **检索范围也是全量 `(pos + 1) // compress_ratio`**：由 `v1/attention/backends/mla/indexer.py:1490`、`:1726`（`seq_lens // compress_ratio`）和 `common/ops/cache_utils.py:1093`（`topk_len = min((pos + 1) // COMPRESS_RATIO, TOP_K)`）可见，压缩 KV cache 保存了 `#0 ~ #9999` 生成的全部 `10000 // compress_ratio` 条记录，`Indexer` 也在全部记录中打分选出 512 条。
> 3. **已有全量压缩区，为何仍保留 128 的本层 SWA 滑窗？**
>    - 在 **`2:1` 压缩层（L2–L19）**：压缩区将 2 个相邻 token 有损合并为 1 条记录（且总长度为奇数时，末尾未配对的 token 尚不写入压缩 cache，`compressor_utils.py:83`）；SWA 滑窗保证最近 128 个 token 拥有未合并的逐 token 原始 KV。
>    - 在 **`1:1` 压缩层（L20–L39）**：压缩 KV cache 在 L20–L39 共 20 层之间**共享 L20 的同一份数据**（L21–L39 不写入压缩 KV），而 SWA 滑窗 cache 由**各层独立计算并维护**——因此 L21–L39 通过共享的 L20 压缩区检索远文（512 条），通过本层独立的 SWA 滑窗保留近处上下文（128 条）。

明确双轨关系后，`config.json` 的 `compress_ratios` 数组中 `0 / 2 / 1` 三档的具体行为如下（注意 **`0` 和 `1` 语义不同**）：

| 档位 | 层 | 生成第 10,000 个 token（`#0 ~ #9999`）时的行为（详见 `compressor.py:175-242`） |
|---|---|---|
| **`0`** | L0–L1（2 层） | **纯滑窗（不启用压缩区，共读 `128` 条）**：<br>• **本层 SWA 滑窗**：读取最近 128 个 token（`#9872 ~ #9999`）<br>• **压缩区**：不使用；`#0 ~ #9871` 不参与计算，不读写压缩 KV cache，也不运行 `Indexer` |
| **`2`** | L2–L19（18 层） | **2:1 压缩 + 检索 top-512（共读 `128 + 512 = 640` 条）**：<br>• **本层 SWA 滑窗**：保留最近 128 个 token（`#9872 ~ #9999`）的原始 KV<br>• **压缩区**：`Compressor` 将 `#0 ~ #9999` 每 **2 个相邻 token**（`(#0,#1)`、`(#2,#3)`…）经门控 `wgate` 加权合并为 **1 条压缩记录**，共 **5,000 条**；`Indexer` 从中检索出 **512 条**，与滑窗的 128 条合并计算注意力 |
| **`1`** | L20–L39（20 层） | **1:1 压缩 + 检索 top-512（共读 `128 + 512 = 640` 条，并非纯滑窗）**：<br>• **本层 SWA 滑窗**：保留最近 128 个 token（`#9872 ~ #9999`）的本层原始 KV<br>• **压缩区**：`Compressor` 不做两两合并（无 `wgate`），`#0 ~ #9999` 每 **1 个 token** 经 `wkv + RMSNorm` 存为 **1 条压缩记录**，共 **10,000 条**；`Indexer` 从这 10,000 条中检索出 **512 条**，与本层滑窗的 128 条合并计算注意力 |

`config.json` 里的 `compress_ratios` 实际有 **43 项**：前 40 项对应主干，末尾 3 个 `0` 对应 3 层 DSpark draft，即 draft 层都是纯滑窗。

#### 3.2.2 跨层复用：谁真的持有模块（三张 source 层表）

先明确一个前提：**40 层里没有任何一层是跑全量历史的普通 Dense MLA——L0–L1 是只读 128 条的纯 SWA 层，而 L2–L39 这 38 层全部都是读 `128 条本层 SWA + 512 条压缩 KV` 的 Sparse MLA 层。**

既然 L2–L39 这 38 层都要读「压缩 KV cache」和「top-512 位置索引」来算 Sparse MLA，如果 38 层每层都各自算一遍压缩、各自打分挑一遍 top-512，显存与计算开销过大。因此在 `DeepseekV4Attention.__init__`（`attention.py:431-482`、`:575-585`）中，**只在少数几层创建 `self.compressor` 和 `self.indexer` 去刷新缓存与索引，其余层的这两个属性直接设为 `None`（跳过压缩与检索步骤，算 Sparse MLA 注意力时直接读取前面层刷新的结果）**：

```python
# attention.py:431-482 —— 仅 8 个层创建 self.indexer，其余 32 层为 None
self.indexer = None
if self.is_index_source:
    ...
    self.indexer = self._indexer_cls(index_k_cache)(...)

# attention.py:575-585 —— 仅 4 个层创建 self.compressor，其余 36 层为 None
self.compressor = None
if self.is_kv_source:
    self.compressor = DeepseekCompressor(...)
```

配置与代码中，将 **`self.compressor` / `self.indexer` 不为 `None`、负责计算并将结果写入共享缓存的层**称为 **`source` 层**（对应 `kv_source_layer_ids`、`index_source_layer_ids`），将属性为 `None`、直接复用前序结果的层称为 `consumer`（`attention.py:306-309`）：

| 共享项 | 负责计算的 `source` 层（子模块不为 `None`） | 直接复用的后续层（子模块为 `None`） |
|---|---|---|
| **压缩 KV cache**<br>（`kv_source_layer_ids`） | **L2、L8、L14、L20**（共 4 层）<br>仅这 4 层创建 `self.compressor` 并写入压缩 KV cache | L3–L7 复用 L2；L9–L13 复用 L8；L15–L19 复用 L14；**L21–L39 全部复用 L20**（因 L20 是最后一个写入压缩 KV 的层，`nvidia/model.py:805` 用 `cut = max(config.kv_source_layer_ids)` 定位 L20，使 L21–L39 在 eager prefill 时只需重放各请求末尾滑窗内的 token） |
| **top-512 位置索引**<br>（`index_source_layer_ids`） | **L2、L8、L14、L20、L24、L28、L32、L36**（共 8 层）<br>仅这 8 层创建 `self.indexer`，重新检索 top-512 并写入共享的 `topk_indices_buffer` | 每层后续紧邻的若干层直接复用该索引（如 L25–L27 复用 L24 选出的 top-512） |
| **粗筛出的 2048 个候选块**<br>（`candidate_source_layer_id`） | **L20**（1 层）<br>L20 的 `self.indexer` 选出 top-512 的同时，将得分最高的 2048 个块（每块 8 个压缩位置）写入 `candidate_block_buffer` | **L24、L28、L32、L36** 的 `self.indexer` 不再遍历全量历史，仅在这 2048 个候选块内做二次检索 |

以相邻的 **L2（`source` 层）** 与 **L3（复用层，`compressor=None, indexer=None`）** 对比，即可看清复用层为何仍然是 Sparse MLA 层（只是省去了准备步骤）：

| 单层 Attention 内的步骤 | L2（`source` 层：持有 `Compressor` 与 `Indexer`） | L3（复用层：`compressor=None, indexer=None`） |
|---|---|---|
| **1. 本层 Q 与 SWA 缓存** | 计算 L2 的 `Q_2`（64 头），将最近 128 个 token 写入 `L2.swa_cache` | 计算 L3 的 `Q_3`（64 头），将最近 128 个 token 写入 `L3.swa_cache` |
| **2. 压缩 KV cache** | **执行 `L2.compressor`**，将压缩记录写入 `L2.kv_cache` | **跳过计算**，通过 `compressed_cache_prefix` 直接指向 `L2.kv_cache` |
| **3. 检索 top-512 索引** | **执行 `L2.indexer`**，选出 512 个下标写入共享的 `topk_indices_buffer` | **跳过计算**，直接读取 `topk_indices_buffer` 中 L2 刚写好的 512 个下标 |
| **4. 执行 MLA 注意力核** | **运行 Sparse MLA（`128 + 512`）**：<br>用 `Q_2` 读取 `L2.swa_cache`（128 条）+ `L2.kv_cache` 中的 512 条 | **同样运行 Sparse MLA（`128 + 512`）**：<br>用 `Q_3` 读取 `L3.swa_cache`（128 条）+ 同一个 `L2.kv_cache` 中的同 512 条 |

结合上表，代码中的三处关键逻辑如下：

- **各层如何确定复用的 `source` 层？** 向下取小于等于当前层号的最大 `source` 层号（`attention.py:343-348`）：
  ```python
  self.kv_source_layer_id = max(s for s in self.kv_source_layers if s <= layer_id)
  self.index_source_layer_id = max(s for s in self.index_source_layers if s <= layer_id)
  ```
  例如 L27 向下查找到最近的 `kv_source` 为 L20、最近的 `index_source` 为 L24，因此读取 L20 的压缩 KV、使用 L24 的 top-512 索引。L0–L1 为纯滑窗（`compress_ratio == 0`），这两个字段均为 `None`；DSpark 的 3 层 draft 同理（`:332-334` 通过 `layer_id < config.num_hidden_layers` 排除 draft 层）。
- **为何 8 个 `Indexer` 层中，仅前 4 个（L2/L8/L14/L20）写入 Indexer K cache？** V4.1 的 `Indexer` 不从 `hidden_states` 投影 key，而是直接将本层 `self.compressor` 输出的 latent 经 `wk + k_norm` 变换为 key（`attention.py:1260-1262`）。后 4 个 `Indexer`（L24/L28/L32/L36）所在层的 `self.compressor` 为 `None`，无 latent 可用，因此不创建 `wk / k_norm` 权重，直接复用 L20 写入的 Indexer K cache 打分，对应代码中的 `owns_k=self.is_kv_source`（`attention.py:472`）。
- **`self.compressor is None` 不代表不读取压缩 KV。** 它仅表示该层不自行计算和写入压缩 KV；只要 `compress_ratio > 0`（L2–L39 均满足），计算注意力时都会通过 `self.compressed_cache_prefix`（`:590`）读取所属 `source` 层的压缩 KV。

#### 3.2.3 两级候选在 NV 路径怎么落地

上表第三行（`candidate_source_layer_id = 20`）在代码中的具体落点如下：

1. 父层按层号算出两个布尔量（`attention.py:460-461`）——`is_candidate_source = (layer_id == candidate_source_layer)`、`uses_candidates = (0 <= candidate_source_layer < layer_id)`，只有这两者之一成立时才把共享的 `candidate_block_buffer` 传给 indexer（`:475-481`），并置 `candidate_write=is_candidate_source`。三个字段本身在 `:328-330` 从 config 读入。
2. `Indexer` 再按自己属不属于 L24+ 来选打分算子（`attention.py:1359-1398`）：
   - **L2 / L8 / L14 / L20**：走 `attn_cls`（`SparseAttnIndexer`，`model_executor/layers/sparse_attn_indexer.py:944`），对全量历史做 **dense 打分**；其中 L20（`candidate_write=True`）同时将得分最高的 2048 个块写入 `candidate_block_buffer`。
   - **L24 / L28 / L32 / L36**：在开启 `attention_config.indexer_sparse_logits` 时走 `mqa_cls`（`SparseMQAIndexer`，`model_executor/layers/sparse_mqa_indexer.py:75`），只对那 2048 个候选块打分。这条路**要求 MXFP4 indexer cache**（`indexer_kv_dtype='mxfp4'`），否则在 `:1367` 直接报错；没开这个开关时这 4 层仍走 dense 的 `SparseAttnIndexer`。

注意这里和其他平台不一样：**NV 路径没有 `_select_candidate_blocks` / `_apply_candidate_mask` 这两个函数**，候选的写入与过滤都在 indexer op 内部完成；那两个名字属于 ROCm 的 `v1/attention/ops/rocm_aiter_mla_sparse.py`。

#### 3.2.4 forward 的四步

`DeepseekV4Attention.forward`（`attention.py:659`）只有四步，但每一步都藏着设计意图：

```python
attn_out = self._alloc_attn_out(hidden_states.shape[0], hidden_states)
qr_kv, kv_score, indexer_weights = self._run_parallel_input_projections(hidden_states)
qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)
self._prepare_and_attn_fn(hidden_states, qr, kv, qr_scale, kv_score,
                          indexer_weights, positions, attn_out)
return self._o_proj(attn_out, positions)
```

**第一步先把输出 buffer 预分配好。** 注释写明原因：后面的注意力核跑在 CUDA Graph 的分段断开区（`breakable_cudagraph`）里，要求输出张量地址固定、原地写入（in-place），所以 MLA 核不返回新张量，而是直接往调用方预先分配好的 `[num_tokens, padded_heads, head_dim]` 里写。

**第二步是三路并行的输入投影**（`attention.py:932`）。`execute_in_parallel` 把最重的 `fused_wqa_wkv` 留在默认流，另外两个较轻的 GEMM 放到辅助流 0 和 1：

```python
if self.compressor is not None:
    def compressor_kv_score():
        return torch.mm(hidden_states, compressor.fused_wkv_wgate.weight.T,
                        out_dtype=torch.float32)
    aux_fns[0] = compressor_kv_score
if self.indexer is not None:
    def indexer_weights_proj():
        weights, _ = indexer.weights_proj(hidden_states)
        return weights
    aux_fns[1] = indexer_weights_proj

qr_kv, (kv_score, indexer_weights) = execute_in_parallel(
    lambda: self._fused_wqa_wkv_gemm(hidden_states), aux_fns,
    self.ln_events[0], self.ln_events[1:3], aux_streams,
    enable=hidden_states.shape[0] <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD)
```

三件事值得注意。一是**两个辅助分支都带 `is not None` 守卫**——不是每层都有 compressor 和 indexer（见 [3.2.2 节](#322-跨层复用谁真的持有模块三张-source-层表)），没有的层这一路为空。二是**并行只在小 batch 下开启**，受 `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` 阈值控制，大 batch 下单核已能占满 SM。三是这里有一条 **V4.1 相对 V4.0 的结构性差异**，源码注释直接点明：

> The v4.1 indexer derives K from the kv-source compressor's latent, so unlike v4.0 there is no indexer K GEMM over hidden_states here.

也就是说 **V4.1 的 indexer K 是从 kv-source compressor 的 latent 导出的，不再单独对 `hidden_states` 做一次 K 投影**。V4.0 这里有三个 GEMM，V4.1 只剩两个。

**第三步 `_split_qkv_and_norm`（`attention.py:773`）把融合投影劈开再各自归一化**，并且在这里决定要不要把量化也融进去：

```python
qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
if self.q_lora_rank % 32 == 0 and self._can_fuse_query_quant:
    qr_quant, kv = fused_q_kv_rmsnorm_quant(qr, kv, self.q_norm.weight.data,
                                            self.kv_norm.weight.data, self.eps)
    return qr_quant, None, kv          # 量化后的 Q + scale，直接传给 wq_b
qr, kv = fused_q_kv_rmsnorm(qr, kv, ...)   # 否则只归一化
```

`q_lora_rank=1280` 能被 32 整除，所以官方配置下只要投影后端支持就会走上面那条融合量化路径，Q 的量化结果和 scale 由 `wq_b` 与 indexer 的 `wq_b` 共用，省掉一次重复量化。

**第四步 `_prepare_and_attn_fn`（`attention.py:833-915`）把第三步的低秩张量变成完整的 Q/KV、写好全部缓存，最后在 CUDA Graph 的分段断开点（eager break）里完成稀疏注意力。**

先看它在进入断开区之前依次准备了哪三样东西（`:856-903`）：

```python
# 4.1 主 Q 升维 + 写本层 SWA 滑窗 cache（attention.py:856-860）
def project_query_and_cache_kv():
    q = self._wq_b_proj(qr, qr_scale).view(-1, self.n_local_heads, self.head_dim)  # 1280 → 64×512
    return self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)       # Q 加 RoPE + kv 写入 SWA cache

# 4.2 若本层有 Compressor（L2/8/14/20），在辅助流上并行算出压缩 latent（:868-879）
if compressor is not None:
    q, latent = maybe_execute_in_parallel(
        project_query_and_cache_kv, lambda: compressor(kv_score, positions), ...)
else:
    q = project_query_and_cache_kv()

# 4.3 若本层有 Indexer（8 个 index_source 层），准备 index_q 并写入 Indexer K / 压缩 KV cache（:881-903）
#     indexer(...) 内部（:1447）：若 owns_k 为真先拿 latent 写 Indexer K cache（_produce_k，:1405），再算 index_q
if compressor is not None:
    indexer_result, _ = maybe_execute_in_parallel(
        prepare_indexer, lambda: compressor.insert_cache(latent, positions, self.rotary_emb), ...)
else:
    indexer_result = prepare_indexer()
index_q, index_q_scale, index_weights_out = indexer_result
```

这三步准备完后，所有的输入投影和 cache 写入都已经就绪，最后才调 `@eager_break_during_capture` 装饰的 `_sparse_indexer_and_attn`（`attention.py:986`）。所谓分段捕获（piecewise capture），就是把上面形状固定的线性投影录进 CUDA Graph 静态图以减少 CPU 发射开销，而在遇到动态的稀疏检索和 MLA 主核时暂时跳出图、退回普通逐算子执行（eager 模式）。`_prepare_and_attn_fn` 在 `__init__`（`attention.py:484`）里按 runner 版本二选一绑定：

```python
self._prepare_and_attn_fn = self._prepare_and_attn
if not vllm_config.use_v2_model_runner:
    # MRV1's piecewise capture only tolerates the wide eager region
    self._prepare_and_attn_fn = self._prepare_and_attn_eager
```

`_sparse_indexer_and_attn`（`attention.py:986`）里面只做最后两件事——先让 `Indexer` 打分挑出 top-512 写进共享的 `topk_indices_buffer`，再调 `forward_mqa` 让 MLA 主核读取 SWA cache 和压缩 cache、把结果写进第一步预分配的 `out` buffer：

```python
if self.indexer is not None and index_q is not None:
    q_quant = (index_q, index_q_scale) if index_q_scale is not None else index_q
    self.indexer.indexer_op(hidden_states, q_quant, None, index_weights)
self.forward_mqa(q, kv, positions, out)
```

**边界划在这里是有讲究的**：输入投影和 cache 写入留在被捕获的图里，只有稀疏检索和 MLA 跑在 eager 区。源码注释还记了一个坑——MRV1 的 piecewise capture 只能接受「宽」的 eager 区，用窄的那个会让输入投影留在图里从而产出垃圾结果（issue #51430）。

#### 3.2.5 MLA 的形状：两段低秩 + 分组输出

形状定义在 `DeepseekV4Attention.__init__`，`q_lora_rank` / `o_lora_rank` / `head_dim` 都从 config 直读：

```text
Q 路：hidden 5120
  → fused_wqa_wkv（MergedColumnParallelLinear [1280 | 512]，disable_tp）
     ├─ q_lora 1280 → q_norm RMSNorm(1280) → wq_b：1280 → 64×512
     └─ kv latent 512 → kv_norm RMSNorm(512)
KV：单头（num_key_value_heads=1），head_dim 512 = 448 NoPE + 64 RoPE

O 路：attn out 64×512
  → wo_a：(64×512/8 = 4096) → 8×1024，按 8 组批矩阵乘
  → wo_b：8×1024 → 5120（RowParallel）
```

`o_lora_rank=1024` + `o_groups=8` 的含义：输出投影不是 `64×512→5120` 的大矩阵，而是先按 8 组压到 `8×1024` 的低秩空间再升回 5120。两段低秩（Q 侧 1280、O 侧 1024）与 V4.0 同形态，只是 `q_lora_rank` 从 1024 变成 1280。

**Q 头数要 padding，且带可学习的 `attn_sink`**：
- `attn_sink`（`attention.py:359` 的 `nn.Parameter`）是每个 Q 头独享的一个可学习标量偏置（Attention Sink），参与注意力 softmax 的分母计算——当某个头在当前位置不需要关注任何历史 token 时，可以把多余的注意力概率分给这个 sink 槽。
- FlashMLA 的 FP8 decode 核只支持 `h_q ∈ {64, 128}`（TP 切分后若不足 64 头会 pad 到 64）。因此 `attn_sink` 按 **padded 头数**分配、初值填 `-inf`，加载权重时只填前 `n_local_heads` 个真实头——多出来的 padding 头保持 `-inf`，在 softmax 里自然权重为 0。

### 3.3 MoE：一个类同时服务主干和 draft

`DeepseekV4MoE`（`nvidia/model.py:125`）本身很薄，它继承 V4.0 的同名类，`__init__` 里只做一件有意思的事——**按 prefix 里的层号切换专家规模**：

```python
n_routed_experts = config.n_routed_experts          # 384
n_activated_experts = config.num_experts_per_tok    # 6
if extract_layer_index(prefix) >= config.num_hidden_layers:
    n_routed_experts = getattr(config, "dspark_n_routed_experts", 0) or n_routed_experts
    n_activated_experts = getattr(config, "dspark_num_experts_per_tok", 0) or n_activated_experts
super().__init__(..., n_routed_experts=n_routed_experts,
                 n_activated_experts=n_activated_experts,
                 num_hash_layers=0, image_sentinel_lo=IMAGE_SENTINEL_BASE_ID)
```

这就是 [1.2 节](#12-关键字段)「draft 层 MoE 规模与主干不同」的落地方式：**同一个类，层号 `>= 40` 就自动变成 128/top-3**。DSpark 的 3 层 draft 在构图时编号是 `layers.40/41/42`，正好落进这个分支。`num_hash_layers=0` 也是在这里写死的——这是「全 40 层 `noaux_tc`、没有哈希路由」这条结论的源码出处。

真正的专家执行不在这个文件，而在它继承的 V4.0 基类 `DeepseekV4MoE`（`models/deepseek_v4/nvidia/model.py:661`）与 `DeepseekV4MegaMoEExperts`（同文件 `:212`）。

**正常情况下 `self.ffn(hidden_states, input_ids, mega_gate_metadata)` 进去干了哪三步**（`models/deepseek_v4/nvidia/model.py:898-994`）：

```python
# models/deepseek_v4/nvidia/model.py:898-994
# 1. 门控打分选出 top-6 专家 ID 与归一化权重（小 batch 走 GateLinear + fused_topk_bias，大 batch 走 bf16_mega_gate）
if not mega_backend.supports_bf16_mega_gate() or hidden_states.shape[0] <= gate_threshold:
    router_logits, _ = self.gate(hidden_states)                                  # [T, 384] FP32
    topk_weights, topk_ids = fused_topk_bias(
        hidden_states=hidden_states, gating_output=router_logits,
        scoring_func=self.scoring_func,                                          # "sqrtsoftplus"
        e_score_correction_bias=self.gate.e_score_correction_bias.data,          # 文本 token 的路由偏置
        topk=self.n_activated_experts, renormalize=self.renormalize,             # top-6 归一化
        input_tokens=input_ids, routed_scaling_factor=self.routed_scaling_factor,# 乘上 1.5
        bias_vl=bias_vl.data if bias_vl is not None else None,                   # 图像 token 的路由偏置
        image_sentinel_lo=self.image_sentinel_lo if bias_vl is not None else 0,
    )
else:
    topk_weights, topk_ids = mega_backend.bf16_mega_gate(...)                    # 融合门控核（:962-976）

# 2. 路由专家计算（DeepseekV4MegaMoEExperts，:980-985）
final_hidden_states = self.experts(
    hidden_states, topk_weights, topk_ids, activation_clamp=float(self.swiglu_limit)
)

# 3. 加上共享专家输出（若未融进上面的 MegaMoE 核内，则单独算完相加，:987-992）
if self.shared_experts is not None and not self.experts.has_fused_shared_experts:
    final_hidden_states += self.shared_experts(hidden_states)
return final_hidden_states
```

这段代码还解释了两个常见问题：
- **为什么调 `self.ffn` 还要传 `input_ids` 进去？** 不是为了哈希路由（V4.1 的 `tid2eid` 已经是 `None`），而是为了**多模态图像 token 的路由偏置 `bias_vl`**（`:767-774`）：门控选 top-6 时要靠 `input_ids`（或预构建的 `mega_gate_metadata.image_token_mask`）识别哪些位置是图像占位 token——普通文本 token 加 `e_score_correction_bias` 选专家，图像 token 改加 `bias_vl` 选专家。
- **共享专家（`shared_experts`）什么时候融进 `self.experts`、什么时候单独算？** 在 `__init__`（`:830-845`）里，只要开了 Sequence Parallel 或 `TP == 1`（每张 EP 卡都持有完整的共享专家权重），`fuse_shared_experts` 就为 `True`，共享专家会直接塞进 `DeepseekV4MegaMoEExperts` 的 `deep_gemm.fp8_fp4_mega_moe` 核里跟 384 个路由专家一起算完；只有在未开 SP 的多卡 TP 下，才退回第 3 步单独调 `self.shared_experts(hidden_states)` 再相加。

**另一条容易误以为默认开启的分支：MoE 延迟归约（`defer_finalize`）。** 在 `DecoderLayer.forward` 调 MoE 时（`nvidia/model.py:618-622`），代码写了一个分支：如果 `self.fuse_mhc_all_reduce and self.ffn.defers_finalize(x.shape[0])` 为真，就不走普通的 `self.ffn(...)`，而是走 `self.ffn.forward_unfinalized(x, input_ids)`——故意不在本层末尾做专家输出的归约（finalize / all-reduce），而是返回一个半成品 `MoEOutput`（所以层间 `x` 的类型标注是 `torch.Tensor | MoEOutput`），推迟到**下一层开头的第一个 mHC 边界**跟那里的通信合并做（Engram 层的前一层 L0/L13 因下一层入口要先独立跑 `mhc_post_tilelang`，在 `:797` 被排除）。

但看 `defer_finalize`（`nvidia/model.py:154`，判定入口 `defers_finalize` 在 `:176`，未归约执行入口 `forward_unfinalized` 在 `:182`）的启用条件，它比看上去严格得多：

```python
experts_cls = getattr(self.experts.routed_experts.quant_method, "experts_cls", None)
if (experts_cls is TrtLlmMxfp4ExpertsModular
        and self.experts.routed_scaling_factor == 1.0):
    moe_config.defer_moe_finalize(MHC_OVERLAP_MAX_TOKENS)
```

两个条件都要满足：专家实现必须是 `TrtLlmMxfp4ExpertsModular`（注释解释了原因——只有这个模块化实现能在 GEMM2 之后停下来），而且 **`routed_scaling_factor` 必须等于 1.0**。

官方 checkpoint 的 `routed_scaling_factor` 是 **1.5**（`models/deepseek_v4/nvidia/model.py:694` 直接 `getattr(config, "routed_scaling_factor", 1.0)` 读进来），所以**这个条件不成立，官方配置下延迟 finalize 不会被打开**，`defers_finalize()` 恒为 `False`，`forward_unfinalized` 分支走不到。讲这条优化时要带上这个前提，否则容易让人以为它是默认开启的。

### 3.4 Engram：N-gram 哈希查表与门控注入

**Engram 的设计目的（解耦记忆与计算）**：在标准 Transformer 中，浅层网络需要消耗大量 Attention 与 FFN 算力去拼合局部子词（subword n-grams，如固定词组、专有名词、代码常见片段）并记忆高频静态搭配。V4.0 曾尝试把前 3 层 MoE 改为按单 token ID 查表路由（`tid2eid`），但只能看单个 token 且仍需执行专家矩阵乘；V4.1 取消了 `tid2eid`，改在 **L1 和 L14** 挂载 **Engram** 模块（与 Qwen 新架构中的 PLE / N-gram Embedding、LongCat-Flash 的 N-gram Embedding 同属「大表查表扩充容量、零算力开销」的设计）：

- **参数量极大，但每步计算量极小**：`config.json` 中 L1 和 L14 的 Engram 嵌入表分别有 **3.84 亿行**（`engram_num_embeddings = [384006168, 384016682]`，`engram_head_dim = 256`，单表约 983 亿参数）；但每个 token 每次只需按哈希下标直接取出 **24 行**（`24 × 256 = 6144` 维），且整张大表支持直接放在主机 CPU 内存中（`cpu_offload`），通过异步流预取隐藏延迟。

实现全部位于 `models/deepseek_v41/common/engram.py`，分三步完成：

1. **词表归一与 24 列质数桶哈希（`NgramHashState.forward`，`engram.py:520` → `_hash_ids_kernel`，`:288`）**
   - 先通过 `token_map`（`:14-16`）将 129,280 的原始 token ID 映射为 99,092 的规范化 ID（`engram_compressed_vocab_size=99092`，合并大小写与前导空白等价词）。
   - 对每个位置 $p$，结合前 3 个位置 $(p-1, p-2, p-3)$ 的 token 组成 **2-gram、3-gram、4-gram**（共 `max_ngram_size - 1 = 3` 种长度）；每种长度用 **8 个哈希头**（`engram_n_heads=8`）分别做奇整数乘法哈希并对互不相同的质数取模（`EngramLayout`，`:208-243`，每个桶约 1600 万行，24 个桶互不重叠拼成 3.84 亿行），产出每个位置 **24 个表行下标** `engram_hashes [T, 24]`。
2. **进层前批量查表与异步预取（`EngramBatch.prepare_embeddings`，`engram.py:1455` → `_engram_lookup_kernel`，`:639`）**
   - 在进入第 0 层之前（`nvidia/model.py:983`），一次性将 L1 和 L14 的 24 列嵌入向量取出（`[T, 24, 256]` 展平为 `[T, 6144]`）。若开启 `cpu_offload`，则在 `engram_prefetch_stream` 上通过 UVA 从锁页主机内存异步拷贝到 GPU 暂存区 `staged_rows`。
3. **投影与四流门控注入（`Engram.forward`，`engram.py:1676` → `_fused_engram_post_wkv_kernel`，`:1186-1275`）**
   - 跑到 L1 和 L14 入口时，先将查出的 `[T, 6144]` 向量过线性层 `self.wkv`（`:1555`，`6144 → (4 + 1) × 5120`），切分为 **4 个流的 `key [T, 4, 5120]`** 和 **1 个共享的 `value [T, 5120]`**。
   - 随后在 `_fused_engram_post_wkv_kernel`（`:1247-1274`）中，将每条残差流 `hidden[:, h, :]` 与对应的 `key[:, h, :]` 做带可学习权重 `q_weight / k_weight` 的 RMS 归一化点积，经 $\text{signed\_sqrt}(\text{dot})$ 与 $\text{sigmoid}$ 得到门控标量 `gate`，按 `hidden + gate * value` 注入四条残差流（图像占位 token 的 `token_mask` 为 `False`，`gate` 强制置 0）。

### 3.5 DSpark：3 层半自回归投机草稿模型

实现位于 `models/deepseek_v41/nvidia/dspark.py`（复用 `model_executor/models/qwen3_dspark.py` 的基础头组件）。与前代逐 token 串行递推的经典 MTP 不同，DSpark 是一个**单步并行提案 `dspark_block_size = 5` 个 token 的半自回归草稿模型**：

1. **融合主干末三层隐状态（`combine_hidden_states`，`dspark.py:161-166`）**
   - 主干 forward 时采集 `dspark_target_layer_ids = [37, 38, 39]` 三层的四流均值隐状态（见 2.2 / 2.3 节的 `previous_aux`），拼接为 `[T, 3 × 5120]` 后经 `main_proj`（`15360 → 5120`）与 `main_norm` 融合为 `main_x [T, 5120]`（`:102-110`）。
   - 同时由 `precompute_and_store_context_kv`（`:169-199`）直接用 `main_x` 算出 3 个草稿层的上下文 SWA KV 并写入各草稿层的滑窗缓存。
2. **3 层轻量化 Decoder 循环（`DSparkDeepseekV4Model.forward`，`dspark.py:201-251`）**
   - 共享主干的 `embed_tokens` 与 `lm_head`（`has_own_embed_tokens = False`，`:283-284`），包含 3 层 `DeepseekV4DecoderLayer`（构图编号 `layers.40/41/42`，checkpoint 权重前缀 `mtp.{0,1,2}.*`）。
   - 每层均为**纯滑窗注意力**（`compress_ratios` 末尾三项为 `0`，且块内 5 个草稿位置之间非因果可见）+ **小号 MoE**（128 个路由专家选 top-3，见 [3.3 节](#33-moe一个类同时服务主干和-draft)）。
3. **Markov 头与置信度头输出（`dspark.py:145-156`）**
   - 3 层循环结束后同样经 `mhc_post_tilelang` + `hc_collapse_triton` 加权合并回单流 `[T, 5120]`（`:242-248`），接 `DSparkMarkovHead`（`dspark_markov_rank = 256`）与 `DSparkConfidenceHead` 生成 5 个草稿 token 提案，交由 40 层主干在下一 step 一次验证。

---

## 4. 量化：MXFP8 线性 + MXFP4 专家

先解释一下这两个格式名：**MXFP8** 和 **MXFP4** 是 NVIDIA Blackwell（SM100）硬件原生支持的**微缩放（Microscaling）低比特格式**。前代 DeepSeek-V3/V4.0 常用的是 `[128, 128]` 大块共享一个 FP32 缩放因子（block-FP8）；而 MX 格式把粒度缩到了**每 32 个元素**（或 `[32, 32]` 块）共享一个 1 字节的 **`ue8m0` 缩放因子**（无符号、8 位指数、0 位尾数，只存 $2^n$ 幂次缩放）。

实现全部在 `models/deepseek_v41/quant_config.py` 的 `DeepseekV4FP8Config`（`:37`，继承通用 `Fp8Config`）。

官方 `quantization_config` 是 `quant_method="fp8"` + `weight_block_size=[32,32]` + `scale_fmt="ue8m0"` + `expert_dtype="fp4"`。

分派逻辑：`is_scale_e8m0 = (expert_dtype == "fp4")`；当 `weight_block_size == [32,32]` 且 `is_scale_e8m0` 时，线性层拿到 `ModelOptLinearMethod(QuantSpec(weight=kMxfp8Static, activation=kMxfp8Dynamic))`——也就是 **MXFP8**，而不是 V4.0 那套 `[128,128]` block-FP8。专家侧用 `Mxfp4MoEMethod`。

这个分派还决定线性层在 vLLM 内部注册的 scale 属性名叫什么（加载权重时，`nvidia/model.py:1548` 的 `hf_to_vllm_mapper` 要靠这个属性名把 checkpoint 里的 scale 张量映射进对应的层）：

| 路径 | 线性层方法 | scale 参数名 |
|---|---|---|
| MXFP8（`[32,32]` + fp4 experts） | `ModelOptLinearMethod` | `weight_scale` |
| block-FP8 | — | `weight_scale_inv` |
| 专家侧 MXFP4 | `Mxfp4MoEMethod` | `w{1,2,3}_weight_scale`（无 `_inv`） |

### 4.1 fp4 专家按层再分 NVFP4 / MXFP4

`expert_dtype == "fp4"` 之后，专家层还会按层判断走 NVFP4 还是原生 MXFP4——`_is_nvfp4_expert_layer`（`quant_config.py:111`），配合 `_modelopt_ignore`（`:61` 初始化）与 `_EXPERTS_LAYER_RE`（`:29`）；调用点在 `get_quant_method`（`:221`）：

```python
def _is_nvfp4_expert_layer(self, prefix: str) -> bool:
    if self.moe_quant_algo != "NVFP4":
        return False
    match = _EXPERTS_LAYER_RE.search(prefix)      # layers.{i}.ffn.experts
    if match is None or not self._modelopt_ignore:
        return True
    ...
    return not any(fnmatch.fnmatchcase(name, p) for p in self._modelopt_ignore)
```

- **官方 checkpoint 不触发这条分支**：官方 `quantization_config` 没有 `moe_quant_algo` 字段，函数第一行直接返回 `False`，**全部专家走原生 MXFP4**。
- **NVFP4 只针对 ModelOpt 重新导出的 checkpoint**：ModelOpt 的 NVFP4 导出会把 `ignore` 列表中的专家保持为原生 MXFP4，典型就是随 checkpoint 打包的 draft 层。所以这类 checkpoint 里可能**两种专家量化并存**。
- draft 层在构图时叫 `layers.{num_hidden_layers + k}`，但 checkpoint 里命名是 `mtp.{k}`，所以判断时要先把层号换算回 `mtp.*` 再去匹配 ignore 模式。

讲「MXFP4 专家」时可以补一句限定：**这是官方 checkpoint 的格式；是否走 NVFP4 取决于 checkpoint 的量化元数据，不是模型结构决定的**。

### 4.2 其余配套约束

- **Quark checkpoint 不由本 config 认领**：`override_quantization_method`（`quant_config.py:164`）遇到 `quant_method == "quark"` 直接返回 `None`，交给 `QuarkConfig` 按层分派，以保留 Quark 的 per-layer scheme。
- **Engram 的 fp8 表有单独重路由规则**：`embed.weight → embed_tokens.weight` 的后缀规则会改表名但不改 scale 名，所以 mapper 显式把 `engram.embed.scale` 路由到 `engram.embed_tokens.weight_scale_inv`。
- **MegaMoE 的硬门槛**（继承 V4.0）：`expert_dtype` 必须 fp4、必须开 EP、`scoring_func` 必须 `sqrtsoftplus`，否则 `DeepseekV4Model.__init__` 直接报错。
- **block-FP8 的 shared expert 要补 padding**：把 intermediate 轴按 `TP × block` 向上取整补零，让标准 TP loader 能均匀切（尾部 rank 落在零 pad 上）；SP 或未量化时不需要。

---

## 5. Prefill / Decode 与 kernel 速查

本章描述缺省的 **FlashMLA 路径**（`nvidia/flashmla.py` 的 `DeepseekV4FlashMLAAttention`）。其余三个后端见 [6.2 节](#62-四个-attention-后端类)。

### 5.1 组件 → kernel 总表

「Python 入口」一列给的是可直接跳转的 `文件:行号`，路径相对 `vllm/`。

| 组件 | Python 入口 | Kernel / Op | Prefill | Decode |
|---|---|---|---|---|
| mHC 首层 | `DeepseekV4DecoderLayer.forward`<br>`models/deepseek_v41/nvidia/model.py:455` | `mhc_pre_delayed_tilelang`（2D 用 broadcast 权重 / 3D 用原权重） | 同 | 同 |
| mHC 层间 | 同上 | `mhc_shifted_post_pre_deep_gemm`（`nvidia/ops/mega_mhc.py:47`，SM100 + `hidden%1024==0`）否则 `mhc_fused_post_pre_delayed_tilelang` | 同 | 同 |
| mHC 末层 | `DeepseekV4Model._collapse`<br>`models/deepseek_v41/nvidia/model.py:1138` | `mhc_post_tilelang` + `hc_collapse_triton(hidden, pre_mix)` | 同 | 同 |
| Engram 查表 | `EngramBatch.prepare_embeddings`<br>`models/deepseek_v41/common/engram.py:1455` | `_engram_lookup_kernel`（`engram.py:639`），CPU offload 时走独立 prefetch stream | 同 | 同 |
| Engram 注入 | `Engram.forward`<br>`models/deepseek_v41/common/engram.py:1676` | `_fused_engram_post_wkv_kernel`（`engram.py:1186`） | 同 | 同 |
| n-gram 哈希 | `NgramHashState.forward`<br>`models/deepseek_v41/common/engram.py:520` | `_hash_ids_kernel`（`:288`）/ `_write_hash_cache_kernel`（`:253`） | 全批算一次 | 同 |
| Compressor | `DeepseekCompressor.forward`<br>`models/deepseek_v41/compressor.py:250` | `fused_save_compress_norm`（`compressor.py:283` 调用，存状态+池化+RMSNorm 一个 Triton 核） | 写压缩 KV | 滑窗增量 |
| 压缩 cache 插入 | `DeepseekCompressor.insert_cache`<br>`models/deepseek_v41/compressor.py:297` | `rope_quant_insert`（`common/ops/fused_compress_quant_cache.py:224`） | 同 | 同 |
| SWA cache 插入 + Q RoPE | `_fused_qnorm_rope_kv_insert`<br>`models/deepseek_v41/attention.py:1011` | `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`(uint8) / `..._full_cache_bf16_insert` / `..._full_cache_fp8_insert` | 同 | 同 |
| Indexer K | `DeepseekV4Indexer._produce_k`<br>`models/deepseek_v41/attention.py:1405` | `indexer_k_norm_rope_store`（`common/ops/indexer_k_store.py:29`） | 同 | 同 |
| Indexer Q | `DeepseekV4Indexer.forward_q`<br>`models/deepseek_v41/attention.py:1439` | `fused_indexer_q_rope_quant` | 同 | 同 |
| Indexer 打分 | `SparseAttnIndexer`（`model_executor/layers/sparse_attn_indexer.py:944`）或 `SparseMQAIndexer`（`sparse_mqa_indexer.py:75`） | DeepGEMM `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits` | dense | paged |
| 候选块 | `DeepseekV4Indexer.__init__` 按 `candidate_write` 分派<br>`models/deepseek_v41/attention.py:1359-1398` | 由 indexer op 内部处理（见 [3.2.3 节](#323-两级候选在-nv-路径怎么落地)），NV 路径没有独立的选择/掩码函数 | 分 chunk | 整批 |
| 短上下文 top-k | `DeepseekV4Indexer.forward`<br>`models/deepseek_v41/attention.py:1470` | `_fill_short_context_topk_indices`（`attention.py:104`，全选） | 同 | 同 |
| 主注意力 decode | `DeepseekV4FlashMLAAttention._forward_decode`<br>`models/deepseek_v41/nvidia/flashmla.py:167` | `flash_mla_with_kvcache`（`:230` 调用；tile_sched_swaonly / c1a / c2a） | — | ✓ |
| 主注意力 prefill | `..._forward_prefill`<br>`models/deepseek_v41/nvidia/flashmla.py:248` | `dequantize_and_gather_k_cache`（`common/ops/cache_utils.py:772`）→ `combine_topk_swa_indices`（`:925`）→ `flash_mla_sparse_fwd` | ✓（`PREFILL_CHUNK_SIZE=4` 分块） | — |
| O 投影 | `_o_proj`<br>`models/deepseek_v41/nvidia/flashmla.py:63`（基类 `attention.py:248`） | `dsv41_o_proj`（`nvidia/ops/o_proj.py:63`）/ `deep_gemm_fp8_o_proj`（逆 RoPE + wo_a + wo_b 融合 FP8 einsum） | 同 | 同 |
| MoE 专家 | `DeepseekV4MegaMoEExperts`<br>`models/deepseek_v4/nvidia/model.py:212` | `deep_gemm.fp8_fp4_mega_moe` + `prepare_megamoe_inputs`（`deepseek_v4/nvidia/ops/prepare_megamoe.py:206`） | 同 | 同 |
| 路由 | `fused_topk_bias`<br>`model_executor/layers/fused_moe/router/fused_topk_bias_router.py:146` | `_moe_C.topk_softplus_sqrt` | 同 | 同 |

### 5.2 FlashMLA 的 Prefill / Decode 形态差

**Decode 单核完成计算。** 注意参数分工：主 `k_cache` 传入 **SWA cache**，压缩 cache 通过 `extra_k_cache` 传入，输出通过 `out=` 原地写入（`nvidia/flashmla.py:230`）：

```python
out, _ = flash_mla_with_kvcache(
    q=q,
    k_cache=swa_cache,                                 # 滑窗
    block_table=None,
    head_dim_v=512,
    tile_scheduler_metadata=tile_metadata,             # 按 compress_ratio 选
    cache_seqlens=None,
    is_fp8_kvcache=True,
    indices=swa_indices,
    topk_length=swa_lens,
    softmax_scale=self.scale,
    attn_sink=self.attn_sink,
    extra_k_cache=kv_cache if not swa_only else None,  # 压缩 KV（可能是别人的）
    extra_indices_in_kvcache=topk_indices,
    extra_topk_length=topk_lens,
    out=output.unsqueeze(1),
)
```

tile scheduler 按本层 `compress_ratio` 三选一，同类型的层在一个 decode step 内共享一份 `FlashMLASchedMeta`——第一次调用触发核内 planner，后续同类型层看到 `have_initialized=True` 就跳过。这也是为什么 `sparse_mla.py` 的 `DeepseekV41SparseSWAMetadataBuilder` 要按 V4.1 的层表重算 plan 集合：共享的 builder 默认按 V4.0 的 1/4/128 分类，V4.1 的 0/1/2 对不上。

非 SWA-only 层还要把 indexer 写的**本地**索引转成全局（`compute_global_topk_indices_and_lens`），压缩页的 block size 是 `block_size // compress_ratio`。

**Prefill 是「gather 后 sparse」。** 按 `PREFILL_CHUNK_SIZE=4`（`attention.py:200` 的 ClassVar）分块，每块：

1. 在 workspace 里把选中的压缩 KV **反量化** gather 成稠密 bf16（`dequantize_and_gather_k_cache`，offset 0）；
2. 再把 SWA KV gather 到同一块 workspace 的 `offset=chunk_N` 之后；
3. `combine_topk_swa_indices` 合并 top-k 索引与 SWA 索引。VL 场景还要带上 `left_visible` / `right_visible` 和 `max_image_tokens`——图像 span 内部是双向可见的，会加宽索引行；
4. `flash_mla_sparse_fwd`。

workspace 宽度 `combined_topk = round_up(top_k + window_size + max_image_tokens, 128)`。注意 `has_compressed = not swa_only`：V4.1 里 **ratio-1 层也要 gather 全长压缩区**，只有 ratio-0 层是纯 SWA。

### 5.3 cache 记录布局

定义在 `common/ops/cache_utils.py` 顶部注释与常量，写入分派在 `common/ops/fused_compress_quant_cache.py`（按字节宽选 kernel）：

| 记录 | 字节 | 构成 |
|---|---|---|
| V4（旧） | 584 B（页 576 B） | 448 fp8 e4m3 NoPE + 64 bf16 RoPE，再 7 个 64 维 UE8M0 scale + 1 字节 pad |
| **V4.1** | **528 B**（页对齐 512 B） | 512 维全部 fp8 e4m3（含 RoPE），再 16 个 UE8M0 scale |
| V4.1 NVFP4 | 288 B | 仅用于压缩区（`nvidia/flash_mla_mega_attn.py`） |

`attention.py:513` 的 `swa_bytes_per_token = 528 if self.kv_mxfp8 else 584` 是这条分派的开关。

Indexer K cache 的每位置字节数由 `attention.py` 的 `_indexer_k_cache_head_dim`（`:92`）按 `use_fp4_kv` 分支给出——MXFP4 变体是 `index_head_dim//2 + index_head_dim//MXFP4_BLOCK_SIZE` = **68 B**（64 packed 值 + 4 UE8M0 scale），FP8 路径是 `index_head_dim + index_head_dim//128*4` = **132 B**（128 fp8 值 + 4 fp32 scale）。

### 5.4 后端与开关

| 开关 | 作用 |
|---|---|
| attention backend | 见 [6.2 节](#62-四个-attention-后端类)的四个类与缺省规则；入口 `_select_dsv4_attn_cls` |
| `--kv-cache-dtype` | `fp8_ds_mla`（UE8M0 uint8，见 5.3 节）vs `auto`/`bf16`/`fp8` plain row |
| `--kernel-config moe_backend` | `deep_gemm_mega_moe` / `flashinfer_moe_ep_mega_*` / 其他 → FusedMoE |
| `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` | token 数 ≤ 阈值才开多流输入 GEMM |
| `VLLM_MOE_SKIP_PADDING` | padding token 的 topk 置 -1 跳过路由 |
| `attention_config.indexer_sparse_logits` | 候选消费层是否只对候选块打分（见 [3.2.3 节](#323-两级候选在-nv-路径怎么落地)）。开启要求 `indexer_kv_dtype='mxfp4'` |
| engram config | `cpu_offload`（pinned host + UVA）、`dp_shared_memory`（节点内 DP 共享一份表，要求 `cpu_offload=True` 且 engram DP size > 1）；见 `common/engram.py` 的 `can_share_engram_tables` 与 `DPSharedEngramStorage` |
| 计算能力 | sparse-MLA backend 只支持 **SM 9 / 10**（`supports_compute_capability`，`sparse_mla.py:124`，内部 `capability.major in [9, 10]`）；mega attention 核仅 SM100 有实例化（`sparse_mla.py:243` 注释，判定在 `:262`）；`mega_mhc` 只在 SM100 家族（`nvidia/ops/mega_mhc.py:17`） |
| kernel block size | `get_supported_kernel_block_sizes`（`sparse_mla.py:89`，mega attention 后端另有一份在 `:258`）返回 `64 if is_device_capability_family(90) else 128` |

---

## 6. 代码地图（查阅用）

**查找实现时需先注意两点目录位置：**

1. **模型代码不在 `vllm/model_executor/models/`**，而在 **`vllm/models/deepseek_v41/`**，按平台分 `common/`（跨平台）、`nvidia/`、`amd/` 三个子目录。本文只讲 `common/` + `nvidia/`。
2. **有三类实现不在 `deepseek_v41/` 目录里**：MoE 专家主体在 `deepseek_v4/`（V4.1 直接继承 V4.0），mHC 的计算核在 `model_executor/kernels/mhc/`，indexer 打分 op 在 `model_executor/layers/`。去 `deepseek_v41/` 里找这三样会找不到。

### 6.1 模块 → 文件

| 结构模块 | 文件 | 主要符号 |
|---|---|---|
| 配置类 | `transformers_utils/configs/deepseek_v41.py` | `DeepseekV41Config`（`:8`，基类 `PreTrainedConfig`） |
| 多模态入口 | `models/deepseek_v41/nvidia/vl_model.py` | `DeepseekV41ForCausalLM`（`:117`，即 checkpoint 的 `architectures`） |
| 文本塔入口 | `models/deepseek_v41/nvidia/model.py` | `DeepseekV41LLMForCausalLM`（`:1537`） |
| 主干 | 同上 | `DeepseekV4Model`（`:662`）、`_run_layers`（`:1093`）、`_collapse`（`:1138`）、`_run_replay_layers`（`:1160`） |
| 后半段窗口重放 | `models/deepseek_v41/decoder_replay_layers.py` | `DecoderReplayLayers`（`:40`）、`ReplayBatch`（`:31`） |
| 单层 | `models/deepseek_v41/nvidia/model.py` | `DeepseekV4DecoderLayer`（`:283`）、`write_kv`（`:627`） |
| MoE 装配 | 同上 | `DeepseekV4MoE`（`:125`），继承 V4.0 |
| **MoE 专家执行** | **`models/deepseek_v4/nvidia/model.py`** | `DeepseekV4MegaMoEExperts`（`:212`）、`DeepseekV4MoE`（`:661`） |
| MegaMoE 输入组织 | `models/deepseek_v4/nvidia/ops/prepare_megamoe.py` | `prepare_megamoe_inputs` |
| Attention 抽象基类 | `models/deepseek_v41/attention.py` | `DeepseekV4Attention`（`:176`，ABC）、`_o_proj`（`:248`） |
| Indexer | 同上 | `DeepseekV4Indexer`（`:1255`）、`DeepseekV4IndexerCache`（`:1175`）、`_produce_k`（`:1405`） |
| **Indexer 打分 op** | **`model_executor/layers/sparse_attn_indexer.py`**、**`sparse_mqa_indexer.py`** | `SparseAttnIndexer`、`SparseMQAIndexer`（由 `DeepseekV4Indexer` 的 `attn_cls` / `mqa_cls` 两个 ClassVar 指向，是平台替换点） |
| Compressor | `models/deepseek_v41/compressor.py` | `DeepseekCompressor`（`:175`） |
| Engram | `models/deepseek_v41/common/engram.py` | `Engram`（`:1513`）、`EngramLayout`（`:208`）、`NgramHashState`（`:423`）、`EngramBatch`（`:1391`）、`ParallelEngramEmbedding`（`:942`） |
| DSpark | `models/deepseek_v41/nvidia/dspark.py` | `DSparkDeepseekV4Model`（`:72`）、`DSparkDeepseekV4ForCausalLM`（`:280`） |
| 量化配置 | `models/deepseek_v41/quant_config.py` | `DeepseekV4FP8Config`（`:37`） |
| 后端与 metadata | `models/deepseek_v41/sparse_mla.py` | `DeepseekV4SparseMLABackend`（`:71`）、`DeepseekV41SparseSWAMetadataBuilder`（`:53`） |
| **mHC 计算核** | **`model_executor/kernels/mhc/`** | `tilelang.py`（`mhc_pre_delayed_tilelang` / `mhc_post_tilelang` / `mhc_fused_post_pre_delayed_tilelang`）、`triton.py`（`hc_collapse_triton`） |
| mHC 的 NV 变体 | `models/deepseek_v41/nvidia/ops/mhc.py`、`mega_mhc.py` | `mhc.py`：`supports_mhc_overlap`（`:44`）、`mhc_pre_delayed_overlap`（`:98`）、`mhc_shifted_post_pre`（`:191`）；`mega_mhc.py`：`is_mega_mhc_supported`（`:17`）、`can_use_mega_mhc`（`:31`）、`mhc_shifted_post_pre_deep_gemm`（`:47`） |
| cache 布局 | `models/deepseek_v41/common/ops/cache_utils.py` | `V4_BYTES_PER_TOKEN`（`:48`）、`V41_BYTES_PER_TOKEN`（`:49`）、`dequantize_and_gather_k_cache`（`:772`）、`compute_global_topk_indices_and_lens`（`:832`）、`combine_topk_swa_indices`（`:925`） |
| 压缩 cache 写入 | `models/deepseek_v41/common/ops/fused_compress_quant_cache.py` | `rope_quant_insert`（`:224`）及其按字节宽分派的 kernel 表（`:256-257`） |
| Indexer K 写入 | `models/deepseek_v41/common/ops/indexer_k_store.py` | `indexer_k_norm_rope_store`（`:29`） |
| Q/KV 归一化融合 | `models/deepseek_v41/common/ops/query_quant.py` | `fused_q_kv_rmsnorm_quant`（`:90`）、`can_fuse_query_quant`（`:135`） |
| O 投影 | `models/deepseek_v41/nvidia/ops/o_proj.py`、`fused_wo_a.py` | `dsv41_o_proj`（`:63`） |
| RoPE 构造 | `models/deepseek_v41/common/rope.py` | `build_deepseek_v4_rope` |

FlashMLA 本体不在模型目录：`flash_mla_with_kvcache` / `flash_mla_sparse_fwd` 由 `vllm/v1/attention/ops/flashmla.py` 导入，再往下是外部 FlashMLA 库。

### 6.2 四个 Attention 后端类

这是 NV 路径的实现分叉点，**不是只有 FlashMLA 一条路**。选择入口是 `nvidia/model.py` 的 `_select_dsv4_attn_cls`（`:197`），它把通用的 sparse-MLA 后端枚举映射到 V4.1 专用类：

| 后端枚举 | 实现类 | 文件 |
|---|---|---|
| `FLASHMLA_SPARSE_DSV41`（及 `_DSV4` / 通用） | `DeepseekV4FlashMLAAttention` | `nvidia/flashmla.py:50` |
| `FLASHMLA_MEGA_ATTN_DSV41` | `DeepseekV4MegaAttnAttention` | `nvidia/flash_mla_mega_attn.py:145`（继承上一行） |
| `FLASHINFER_MLA_SPARSE_DSV41` | `DeepseekV4FlashInferMLAAttention` | `nvidia/flashinfer_sparse.py:246` |
| 同上，但 SM12 设备 | `DeepseekV4FlashInferSM120Attention` | `nvidia/flashinfer_sparse.py:602` |

不指定后端时的缺省：**SM12 走 FlashInfer，SM100 在拓扑允许时走 mega attention，其余走 FlashMLA**。V3.2 的 `FLASHINFER_MLA_SPARSE` / `_SM120` 会被显式报错拒绝，必须用带 `_DSV41` 后缀的那个。

四个类都继承 `attention.py` 的抽象基类 `DeepseekV4Attention`，各自实现 `_forward_decode` / `_forward_prefill` / `_o_proj`。**第 5 章描述的是 `nvidia/flashmla.py` 这条缺省路径。**

---

## 7. 核验与维护

### 7.1 本文的核验记录

| 步骤 | 做法 | 结果 |
|---|---|---|
| 锁定树身份 | `git rev-parse HEAD` + `git describe --tags` + `git remote -v` | 官方仓库，HEAD `5df63a7203` 即 `v0.32.0` tag |
| 确认目录布局 | `git ls-tree -r --name-only HEAD` 过滤 `^vllm/.*deepseek_v4` | 模型实现位于 `vllm/models/deepseek_v41/` |
| 符号存在性 | 提取 277 个候选符号，一次性读入 `vllm/` 全部源码做子串匹配 | **277/277 命中** |
| 实现路径定位 | 按类名/函数名定位到文件与行，核对继承链与后端选择分支 | 已写入第 6 章代码地图 |
| 计数类断言 | 对照官方 `config.json` 复算 `compress_ratios`、各 source 层表、候选参数、Engram 与 DSpark 字段 | **全部一致**（2 / 18 / 20 三档；4 个 kv-source；8 个 index-source；候选 L20 / 2048 块 / 每块 8；24 个哈希列） |

### 7.2 升级到更新的上游版本时

1. **锁定新树身份**：版本号、commit，确认 `vllm/models/deepseek_v41/` 目录没有重组。
2. **重放符号核验**：一次性全量读入源码做子串匹配，比逐符号 grep 快两个数量级。
3. **复算计数类断言**：对照当时的官方 `config.json`，不要沿用旧值。
4. **重新定位代码地图**：第 6 章的表是行号密集区，跨目录复用关系（`deepseek_v4/`、`model_executor/kernels/mhc/`、`model_executor/layers/`）尤其要确认。
5. **重点复查量化分派与后端选择**：第 4 章与 [6.2 节](#62-四个-attention-后端类)是上游改动最活跃的区域。

完成上述步骤前，顶部基线表不要提前改写。
