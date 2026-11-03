# DeepSeek-V4.1-Flash 文档与资源索引

> **阅读导航**：**[全局索引（本文）](INDEX.md)** ｜ ① [本仓设备侧实现 (README，含全部架构图)](README.md) ｜ [上游 vLLM 实现（NVIDIA，对外口径）](上游vLLM实现.md) → ② [实服部署指南](部署指南.md) ｜ [分步交互查看器](assets/html/dsv41_dynamic_architecture.html) ｜ [原图集文档（已并入 README，保留备查）](架构图集与算子映射.md)

本目录汇总 DeepSeek-V4.1-Flash 的架构说明、实服部署指南、上游讲义以及全套图表资源。核心文档按阅读阶梯平铺在根目录，图表资源按文件类型归入 `assets/`。

## 1. 目录结构总览

```text
dsv41_flash/
├── INDEX.md                                   # 全局文档与资源导航索引（本文件）
├── README.md                                  # 第 1 站：本仓设备侧实现，基线 vLLM 0.30.0（全部架构图 + 从模型结构到一次推理 + 38 个算子与计算量）
├── 上游vLLM实现.md                             # 对外口径：上游官方 vLLM 的 NVIDIA 实现（基线 v0.32.0）
├── 部署指南.md                                 # 第 2 站：实服部署、编译安装与性能/精度验收指南
├── 架构图集与算子映射.md                        # 已整体并入 README.md，原件保留备查（勿再单独维护）
├── reference/                                 # 上游参考讲义与评审文档
│   ├── LEARN.md                               # 上游 vLLM (NVIDIA) 实现学习讲义
│   └── REVIEW_deepseek_v41_flash.md           # LEARN.md 核验与评审报告
└── assets/                                    # 图表资源（按 html / svg / png / dot 分类）
    ├── html/
    │   ├── dsv41_dynamic_architecture.html                        # 9 页签分步交互查看器（主版）
    │   ├── dsv41_dynamic_architecture_brief.html                  # 调用链速览动态查看器（Archify）
    │   ├── dsv41_dynamic_architecture_brief.architecture.json     # 速览查看器的节点配置
    │   └── dsv41_dynamic_architecture_brief.visual-check.json     # 速览查看器的生成检查记录
    ├── svg/
    │   ├── dsv41_call_architecture.svg        # 架构总板（五步阅读流 + 七块接口面板）
    │   ├── dsv41_call_architecture_brief.svg  # 调用链速览图
    │   ├── dsv41_model_architecture.svg       # 七分区模型架构总图
    │   ├── dsv41_decoder_layer.svg            # DecoderLayer 延迟 mHC 状态机详图
    │   ├── dsv41_mla.svg                      # MLA / Compressor / Indexer / O 投影详图
    │   └── dsv41_moe.svg                      # MoE 路由专家与共享专家详图
    ├── png/                                   # 由 README.md 文内折叠的 Mermaid 图源导出
    │   ├── dsv41_model_architecture_overview.png
    │   ├── dsv41_model_architecture.png
    │   ├── fig2_decoder_layer_internals.png
    │   ├── fig3_mla_indexer_internals.png
    │   ├── fig4_moe_internals.png
    │   └── dsv41_inference_pipeline.png
    └── dot/
        ├── dsv41_call_architecture.dot        # 总图骨架 DOT 源码
        ├── dsv41_call_architecture_brief.dot  # 调用链速览图 DOT 源码
        ├── dsv41_model_architecture.dot       # 七分区模型架构总图 DOT 源码
        ├── dsv41_decoder_layer.dot            # DecoderLayer 局部图 DOT 源码
        ├── dsv41_mla.dot                      # MLA 局部图 DOT 源码
        └── dsv41_moe.dot                      # MoE 局部图 DOT 源码
```

## 2. 核心阅读阶梯（根目录平铺文档）

| 顺序 | 文档 | 定位与主要内容 |
|---|---|---|
| **① 架构全书（对内）** | **[README.md](README.md)** | 本仓设备侧实现，图文已合一：第 0 章一页纸速览 → 第 1 章参数与对象轮廓（含七分区全景、模型总览）→ 第 2 章上层走通一次端到端推理（含 DecoderLayer 图、推理数据流与算子映射图）→ 第 3 章深入 mHC / Attention / MoE（含 MLA、MoE 内部图）→ 第 4～8 章状态、分支、Iluvatar 实现、框架接入与计算负载 → 第 9 章图索引、M1–M12 张量契约总表、结构—算子对照与历史验证记录。 |
| **② 上游口径（对外）** | **[上游vLLM实现.md](上游vLLM实现.md)** | 上游官方 vLLM 的 NVIDIA 实现，不含任何设备侧内容：40 层层表与 `compress_ratios` 三档、MLA 两段低秩、MXFP8 线性 + MXFP4 专家量化、FlashMLA 的 Prefill/Decode 形态差、组件→kernel 总表与后端开关，只描述上游官方实现。**基线上游 `v0.32.0` / `5df63a7203`**，层表数字已对照官方 `config.json` 复核。 |
| **③ 实操落地** | **[部署指南.md](部署指南.md)** | DeepSeek-V4.1-Flash-W4A8 部署指南：含 CoreX 5.0 环境、开发者模式编译安装、INT8 Indexer 启动配置与实测性能/精度验收。 |

## 3. 图表资源（`assets/`）

### 3.1 交互式查看器（`assets/html/`）
- **[dsv41_dynamic_architecture.html](assets/html/dsv41_dynamic_architecture.html)**：主版分步查看器（①全模型、②一层Decoder、③Attention、④Engram、⑤DSpark、全部接口详图、DecoderLayer、MLA、MoE 共 9 个页签）。
- **[dsv41_dynamic_architecture_brief.html](assets/html/dsv41_dynamic_architecture_brief.html)**：调用链速览动态查看器（调用链动画、节点聚焦、路径追踪），配置见 [architecture.json](assets/html/dsv41_dynamic_architecture_brief.architecture.json)。

### 3.2 矢量架构图（`assets/svg/`）
- **[dsv41_call_architecture.svg](assets/svg/dsv41_call_architecture.svg)**：模块数据流与接口详图总板（M1–M12，配合 `README.md`）。
- **[dsv41_decoder_layer.svg](assets/svg/dsv41_decoder_layer.svg)**：DecoderLayer 延迟 mHC 与五元组状态机（M3 / M4，嵌于 `README.md` 2.3 节）。
- **[dsv41_mla.svg](assets/svg/dsv41_mla.svg)**：MLA 并发投影、两级缓存、检索与 O 投影（M5 / M6 / M7 / M9，嵌于 `README.md` 3.2 节）。
- **[dsv41_moe.svg](assets/svg/dsv41_moe.svg)**：MoE 路由专家与共享专家并行汇合（M8，嵌于 `README.md` 3.3 节）。
- **[dsv41_call_architecture_brief.svg](assets/svg/dsv41_call_architecture_brief.svg)**：调用链速览图（嵌于 `README.md` 第 0 章）。
- **[dsv41_model_architecture.svg](assets/svg/dsv41_model_architecture.svg)**：七分区模型架构总图（嵌于 `README.md` 1.3 节）。

### 3.3 位图（`assets/png/`）
全部嵌于 [`README.md`](README.md) 正文，由同一小节 `<details>` 折叠的 Mermaid 代码块导出：
[dsv41_model_architecture_overview.png](assets/png/dsv41_model_architecture_overview.png)（图 0，0.4 节）、[dsv41_model_architecture.png](assets/png/dsv41_model_architecture.png)（图 1，1.3 节）、[fig2_decoder_layer_internals.png](assets/png/fig2_decoder_layer_internals.png)（图 2，2.3 节）、[fig3_mla_indexer_internals.png](assets/png/fig3_mla_indexer_internals.png)（图 3，3.2 节）、[fig4_moe_internals.png](assets/png/fig4_moe_internals.png)（图 4，3.3 节）、[dsv41_inference_pipeline.png](assets/png/dsv41_inference_pipeline.png)（推理数据流与算子映射图，2.5 节）。

> 结构图（图 0～图 4、七分区全景）画的是**模型结构**，上游与设备侧共通，对外讲解可直接复用；**推理数据流与算子映射图画的是设备侧算子，不要用于对外**；对外的上游口径见 [`上游vLLM实现.md`](上游vLLM实现.md)。

### 3.4 Graphviz DOT 图源（`assets/dot/`）
每个 SVG 都有同名 DOT 图源，修改图时先改 DOT 再重新渲染：
[dsv41_call_architecture.dot](assets/dot/dsv41_call_architecture.dot)、[dsv41_call_architecture_brief.dot](assets/dot/dsv41_call_architecture_brief.dot)、[dsv41_model_architecture.dot](assets/dot/dsv41_model_architecture.dot)、[dsv41_decoder_layer.dot](assets/dot/dsv41_decoder_layer.dot)、[dsv41_mla.dot](assets/dot/dsv41_mla.dot)、[dsv41_moe.dot](assets/dot/dsv41_moe.dot)。

## 4. 上游参考与评审（`reference/`）

- **[reference/LEARN.md](reference/LEARN.md)**：上游 vLLM（NVIDIA 实现）学习讲义，9 章 + 源码速查附录，是 [`上游vLLM实现.md`](上游vLLM实现.md) 的深度底稿。基线是旧的 `ced6857afa`（0.30 线），**行号不适用于 v0.32.0**，且其中 `nvidia/engram.py` 已在 0.32 删除（并入 `common/engram.py`）；按符号名查即可，符号 277/277 仍在。
- **[reference/REVIEW_deepseek_v41_flash.md](reference/REVIEW_deepseek_v41_flash.md)**：上游学习讲义的逐行核验与评审报告。

## 5. 外部原工程仓库记录索引

以下记录位于原工程仓库，保留原路径供回到源码工程查阅：

- 适配记录：`../../adaptation_records/deepseek_v41_flash/`
- 性能记录：`../../adaptation_records/deepseek_v41_performance/`
- 架构图口径记录：`../../adaptation_records/deepseek_v41_architecture/vllm_0.26.0_fix_dsv41_architecture_dspark_layer_count.md`
- 长上下文 eager 排查：`../../adaptation_records/dsv41_flash_eager_long_context_garbage.md`
- 性能速查：`../../user/performance_cheatsheet.md#deepseek-v41-flash-w4a8-w4a8`
- 精度速查：`../../user/evaluation_cheatsheet.md#deepseek-v41-flash-w4a8`
