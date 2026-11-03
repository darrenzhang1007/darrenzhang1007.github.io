# DeepSeek-V4.1-Flash-W4A8 部署指南

> **阅读导航**：[全局索引 (INDEX)](INDEX.md) ｜ ① [架构图集与算子映射](架构图集与算子映射.md) → ② [核心架构长文 (README)](README.md) → **③ [实服部署指南（本文）](部署指南.md)** ｜ [分步交互查看器](assets/html/dsv41_dynamic_architecture.html)

本指南对应127机器、16个Iluvatar BI-V150可见设备、CoreX 5.0及现有容器 `ds_v4.1_perf`。当前推荐配置为 **TP8 / PP1 / DP2 / EP16、INT8 indexer、DSpark K5 greedy、CUDA Graph**，支持262144上下文、每请求最多5张图片。

2026-10-21已在 `ds_v4.1_perf` 内以开发者模式完整编译安装当前插件，正式主扩展直接提供INT8接口，CPU检查、重启和简单请求验证均通过，**16个worker直接使用新主扩展的INT8接口，临时bridge不再加载**。保持 `VLLM_ILUVATAR_INDEXER_INT8=1` 和原服务全部模型参数；新runtime不再预载镜像旧主扩展或临时bridge。历史主扩展、bridge、性能与精度日志均保留。

## 1. 版本与产物

记录日期：2026-10-21。SDK版本取自本轮产物的构建记录，不以app源码目录之后的HEAD替代。

| 组件 | 当前版本 / 标识 |
|---|---|
| 容器 | `ds_v4.1_perf`，x86_64；GPU运行均在此容器内 |
| 镜像 | `registry.iluvatar.com.cn:10443/vllm_plugin/corex_vllm_plugin:0.30.0-corex-5.0.0-ubuntu24.04-py3.12-x86_64-release` |
| 本地镜像ID | `sha256:4c40065f76a22b798c3325e9ac944821c220637a92e88dcfc0c4b05f7491a579` |
| CoreX / Python | CoreX 5.0.0；Python 3.12.11 |
| PyTorch | `2.10.0+corex.5.0.0`，C++ ABI=0 |
| vLLM / vllm_iluvatar | 0.30.0 / 0.30.0 |
| 插件源码 | 分支 `dsv41-pp-dspark-defer-perf`；本轮完整安装起点 `12788ccc`，生产源码与 `8b8187b7b083553be614f50778d8bff941bb1579` 一致 |
| 安装方式 / 主扩展 | `pip install -e /workspace/vllm_iluvatar --no-deps --no-build-isolation -v`；源码目录正式 `_C.so`，构建、CPU检查及实服冒烟均通过 |
| ixformer | `08a47e1f2fc7e1b47035b9b60423bd14de015c57`，从app组源码用Python3.12/CoreX5.0重编译 |
| ixinfer | `aca6f3547c5731f6716e64149d24b2da74556175`，从app组源码用CoreX5.0重编译，`libcuinfer.so.7` |
| GPU架构 | `ivcore11` |
| 权重 | 宿主机 `/mnt/share/user_homes/zyp/models/deepseek-ai/DeepSeek-V4.1-Flash-W4A8`；容器 `/workspace/models/deepseek-ai/DeepSeek-V4.1-Flash-W4A8` |

SDK源码根目录为宿主机 `/mnt/share/user_homes/zyp/app_perf_infer/sdk`，容器对应 `/workspace/app_perf_infer/sdk`。该目录原有Python3.11/CoreX4.4.4产物不是本轮使用的产物。

构建时SDK子模块并非全部处于父仓库记录的位置：ixformer的 `third_party/ixmma` 实际为 `bad35df9ea5299f8be56a84212dfa97d46c59a98`；ixinfer的 `3rdparty/msgpack`、`3rdparty/spdlog` 存在工作区差异。**只checkout上表两个HEAD不足以完整复现源码状态。** 完整子模块列表、源码差异、构建命令与10个产物的SHA256均保留在下述构建记录中：

```text
/workspace/vllm_iluvatar/logs/perf/v0.30.0/dsv41_functional_merge_20261020_142112/
  sdk_source_manifest.json
  sdk_ixformer_source_diff.patch
  sdk_ixinfer_source_diff.patch
  sdk_build_commands.json
  sdk_compiled_manifest.json
  sdk_build_status.json
```

`sdk_source_manifest.json` 的SHA256为 `6677dd85fb85ad073da263555e77186b9977ea14a04cff5c3e80df7dcd928141`。这是源码定位与产物哈希记录，不承诺两次编译的二进制逐字节一致。

## 2. 当前部署所需目录

宿主机 `/mnt/share/user_homes/zyp/` 映射为容器 `/workspace/`。

| 用途 | 容器路径 |
|---|---|
| 插件Python源码 | `/workspace/vllm_iluvatar` |
| IXFORMER_ROOT | `/workspace/vllm_iluvatar/logs/functional/v0.30.0/dsv41_editable_native_20261021_104955/runtime` |
| ixformer真实编译库 | `/workspace/vllm_iluvatar/logs/perf/v0.30.0/dsv41_functional_merge_20261020_142112/sdk_corex50/ixformer/lib` |
| IXINFER_LIB | `/workspace/vllm_iluvatar/logs/perf/v0.30.0/dsv41_functional_merge_20261020_142112/sdk_corex50/ixinfer` |
| ixformer Python包和扩展链接 | `/workspace/vllm_iluvatar/logs/perf/v0.30.0/dsv41_functional_merge_20261020_142112/sdk_corex50/runtime/ixformer_runtime` |
| 当前正式主扩展 | `/workspace/vllm_iluvatar/vllm_iluvatar/_C.cpython-312-x86_64-linux-gnu.so` |
| 历史INT8临时bridge（保留、不再加载） | `/workspace/vllm_iluvatar/logs/perf/v0.30.0/dsv41_merged_sdk_indexer_int8_perf_after_20261020_152607/native_int8_bridge/build/_indexer_int8_native.cpython-312-x86_64-linux-gnu.so` |

`IXFORMER_ROOT/ixformer_runtime` 是指向上述SDK runtime的链接，ixformer没有安装成独立pip包。新runtime的 `sitecustomize.py` 从开发者安装正常导入新 `_C`，仅在首次真实INT8调用成功后记录worker身份、输入dtype及主扩展路径，随后恢复原生函数；不执行旧主扩展预载或bridge补挂。runtime和研发SDK仍为启动依赖，迁移时需要保留链接目标并调整绝对路径。

正式主扩展SHA256为 `d081133c4d0ae4c5d37d93eddfc68fcb2f9acc73864b381f4f3980e26c2b6149`。旧镜像主扩展SHA256 `8edd76fe7cf488b526feaf64014e1ff0825393755312494b9e68a6b5cb9a4bbb`、bridge SHA256 `e16bfe22907b3e2e218db1b7cee139404b1076808de0b96301248de64119deb8` 只用于历史证据和回退备份；安装前的包和主扩展已备份在本次结果目录 `backup/`，见 `backup/manifest.json`。旧目录不删除，但当前启动不依赖bridge。研发ixinfer仍须先于系统库加载；错误绑定系统cuinfer曾导致缺符号或启动hang。

## 3. 启动命令

进入现有容器：

```bash
docker exec -it ds_v4.1_perf bash
```

启动前用 `ixsmi` 确认16个设备空闲，重启只处理已确认属于本次服务的进程。当前正式主扩展服务已在8000端口通过验证并保留运行，本轮仅清理启动配置，未重启服务。以下命令保持 `dsv41_editable_native_20261021_104955/launch_command_vllm.sh` 的实际模型配置，省略当前INT8路径无效或已有默认值的环境设置；`max-num-seqs=128` 和 `max-num-batched-tokens=2048` 采用当前默认，不显式传入：

```bash
export IXFORMER_ROOT=/workspace/vllm_iluvatar/logs/functional/v0.30.0/dsv41_editable_native_20261021_104955/runtime
export IXINFER_LIB=/workspace/vllm_iluvatar/logs/perf/v0.30.0/dsv41_functional_merge_20261020_142112/sdk_corex50/ixinfer
export VLLM_ILUVATAR_ROOT=/workspace/vllm_iluvatar
export PYTHONPATH="${IXFORMER_ROOT}:${IXFORMER_ROOT}/ixformer_runtime:${VLLM_ILUVATAR_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${IXINFER_LIB}:${IXFORMER_ROOT}/ixformer_runtime/ixformer:/usr/local/corex/lib64:/usr/local/openmpi/lib:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VLLM_ILUVATAR_INDEXER_INT8=1
export VLLM_ILUVATAR_IXFD_COMM=1
export VLLM_ILUVATAR_DSPARK_SPLIT_SAMPLE=1
export VLLM_ILUVATAR_DSV41_DISABLE_AUX_STREAM=1
export VLLM_ILUVATAR_W4A8_FILTER_EP=1
export NCCL_ALGO=allreduce:tree
export TORCH_NCCL_BLOCKING_WAIT=1
unset CUDA_LAUNCH_BLOCKING VLLM_PP_LAYER_PARTITION

vllm serve /workspace/models/deepseek-ai/DeepSeek-V4.1-Flash-W4A8 \
  --served-model-name DeepSeek-V4.1-Flash-W4A8 \
  -tp 8 -pp 1 -dp 2 --enable-expert-parallel \
  --tokenizer-mode deepseek_v41 \
  --mm-encoder-tp-mode weights \
  --limit-mm-per-prompt '{"image":5}' \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 \
  --no-enable-prefix-caching \
  --max-model-len 262144 \
  --max-cudagraph-capture-size 256 \
  --gpu-memory-utilization 0.83 \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy"}' \
  --port 8000
```

省略的4项及其依据如下，默认行为以当前安装的vLLM 0.30.0和本插件为准：

| 省略项 | 当前配置的行为 |
|---|---|
| `VLLM_ILUVATAR_DSA_PERSISTENT_PAGES` | 仅选择FP8 paged scorer实现；INT8评分提前返回，不经过该实现 |
| `VLLM_USE_V2_MODEL_RUNNER` | 未设置时自动选择；当前DSpark、多模态和TP8/PP1/DP2配置满足V2条件 |
| `VLLM_ENGINE_READY_TIMEOUT_S` | 插件在未设置时默认补为3600秒 |
| `NCCL_PROTO` | 当前MoE且DP>1，插件在未设置时默认补为Simple |

若外部已经设置过V2 runner、启动超时或NCCL协议，相应覆盖仍会生效。复用旧shell时，可先执行 `unset VLLM_USE_V2_MODEL_RUNNER VLLM_ENGINE_READY_TIMEOUT_S NCCL_PROTO` 恢复默认选择。INT8、IXFD、DSpark split sampling、关闭aux stream、decode EP过滤、NCCL Tree及blocking wait保留；DSpark K5 greedy由 `--speculative-config` 设置。历史性能表中的环境字段和原始启动日志保留原样，不将本次精简标为重测。

当前32GiB设备上的 `vllm serve` 默认batch token预算为2048；已开启chunked prefill，没有启用throughput模式，单图1024 tokens也不会触发多模态预算上调。该默认不能泛化到其他显存容量、运行模式或多模态配置。新权重脚本和本节命令均省略此参数，旧实验的显式2048及原始日志保留。

也可使用仓库入口 `examples/deepseek_v41_flash/serve_w4a8_mixed_dp2.sh`，先设置同一组研发runtime/库环境，并显式设置 `DSV41_MULTIMODAL=1`。脚本默认开启INT8，保留显式 `VLLM_ILUVATAR_INDEXER_INT8=0` 的FP8兼容选项；默认仍为纯文本入口，启用多模态后使用本节的图片数、上下文及batch参数。

## 4. 完整开发者模式编译安装与INT8接口

**正常编译当前仓库不会单独生成 `_indexer_int8_native.so`；它将 `paged_mqa_logits_int8` 接口直接编入主扩展 `vllm_iluvatar._C`，INT8 indexer继续保留。** 本次已用当前源码、CoreX5.0及同一批研发SDK完成全插件开发者安装，构建退出码0、耗时580.1秒；`cpu_preflight.json` 确认editable路径为 `file:///workspace/vllm_iluvatar`，正常导入的是第2节的新主扩展，INT8接口可调用，临时bridge未加载，9个研发库哈希不变，IXFD的 `all_reduce` 和 `init_comm_with_store` 可用。随后新主扩展已通过起服和简单请求验证，结果见第4.3与第5节。

- `CMakeLists.txt`（源码／外部材料：`../../../CMakeLists.txt`） 收集 `csrc` 下的C++/CUDA文件并构建 `_C`。
- `csrc/pybind_kernel.cc`（源码／外部材料：`../../../csrc/pybind_kernel.cc`） 在 `ENABLE_CUINFER` 下绑定 `cuinfer.paged_mqa_logits_int8`。
- `setup.py`（源码／外部材料：`../../../setup.py`） 在ivcore11默认启用cuinfer；本次显式设置 `VLLM_ILUVATAR_ENABLE_CUINFER=1`。
- 旧bridge只补充镜像旧主扩展缺少的接口，其源码、构建命令和历史产物继续保留；切换后的runtime不再加载它。

### 4.1 开发者安装命令与原始记录

在现有 `ds_v4.1_perf` 内执行，当前服务停止并确认资源释放后再安装。安装输出为源码目录主扩展，Python包以editable方式注册；没有重装依赖或修改研发SDK。编译进程临时清除INT8运行开关只是为了不执行服务runtime，服务启动必须按第3节重新设置为1。

根CMake不会自动将 `IXINFER_LIB` 用作编译期include/link配置，本次通过 `sdk_paths.cmake` 显式传入已验收SDK头文件和库路径。头文件配置沿用原始 `sdk_paths.cmake`；以下安装命令在代码清理后删除了输出模式环境变量，其余参数与已执行成功的 `install_editable.sh` 一致。当前构建统一继承stdout/stderr。原始脚本和CMake记录保持原样，后续新构建使用新结果目录。

```bash
mkdir -p /workspace/vllm_iluvatar/logs/functional/v0.30.0/dsv41_editable_native_20261021_104955
cat > /workspace/vllm_iluvatar/logs/functional/v0.30.0/dsv41_editable_native_20261021_104955/sdk_paths.cmake <<'CMAKE'
include_directories(BEFORE "/workspace/app_perf_infer/sdk/ixinfer/include")
link_directories(BEFORE "/workspace/vllm_iluvatar/logs/perf/v0.30.0/dsv41_functional_merge_20261020_142112/sdk_corex50/ixinfer" "/usr/local/corex/lib64")
CMAKE
```

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm_iluvatar
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=/usr/local/lib/python3.12/site-packages
export LD_LIBRARY_PATH=/workspace/vllm_iluvatar/logs/perf/v0.30.0/dsv41_functional_merge_20261020_142112/sdk_corex50/ixinfe[本地路径]
export VLLM_ILUVATAR_CUDA_ARCHITECTURES=ivcore11
export CMAKE_CUDA_ARCHITECTURES=ivcore11
export TORCH_CUDA_ARCH_LIST=ivcore11
export VLLM_ILUVATAR_ENABLE_CUINFER=1
export VLLM_ILUVATAR_MAX_JOBS=8
export CMAKE_BUILD_TYPE=Release
export VLLM_ILUVATAR_CMAKE_BUILD_DIR=/workspace/vllm_iluvatar/logs/functional/v0.30.0/dsv41_editable_native_20261021_104955/cmake
export CMAKE_ARGS="-DCMAKE_PROJECT_INCLUDE=/workspace/vllm_iluvatar/logs/functional/v0.30.0/dsv41_editable_native_20261021_104955/sdk_paths.cmake -DCMAKE_CUDA_COMPILER=/usr/local/corex/bin/clang++ -DCMAKE_OBJCOPY=/usr/bin/objcopy"
unset VLLM_ILUVATAR_ENABLE_STARTUP_PATCH VLLM_ILUVATAR_INDEXER_INT8 CUDA_LAUNCH_BLOCKING
python3 -m pip install -e /workspace/vllm_iluvatar --no-deps --no-build-isolation -v
```

本次 `build_install.log` 只保留140行，停在Ninja17/101，应视为部分stdout/stderr，缺失部分没有补造。`cmake/.ninja_log` 独立记录全部101步及最终链接，`build_status.json` 记录退出0及耗时，CPU与实服记录确认新产物和editable安装成功；`preflight.json` 记录构建起点、工作区状态和旧服务信息，安装前备份位于 `backup/`。仓库其他未提交的分析文件未被清理。

### 4.2 不加载旧runtime或bridge的独立CPU检查

本次正常editable导入检查已通过，详情在 `cpu_preflight.py`、`cpu_preflight.sh` 和 `cpu_preflight.json`。如需额外确认确切文件，可使用下面的 `-I -S` 检查，避免系统 `.pth` 或 `sitecustomize` 触发其他预载。它只导入主扩展，不执行GPU算子，也不代替实际请求验证。先设置第3节的 `LD_LIBRARY_PATH`，再执行：

```bash
CUDA_VISIBLE_DEVICES='' python3 -I -S - /workspace/vllm_iluvatar/vllm_iluvatar <<'PY'
import hashlib
import importlib.util
import json
import site
import sys
import sysconfig
from pathlib import Path

assert "sitecustomize" not in sys.modules
assert "vllm_iluvatar" not in sys.modules
# -S下import site不会执行site.main；不用会处理.pth的site.addsitedir。
sys.path.extend(site.getsitepackages())
import torch
target = (Path(sys.argv[1]) / ("_C" + sysconfig.get_config_var("EXT_SUFFIX"))).resolve(strict=True)
spec = importlib.util.spec_from_file_location("_C", target)
module = importlib.util.module_from_spec(spec)
sys.modules["_C"] = module
spec.loader.exec_module(module)
assert Path(module.__file__).resolve() == target
assert callable(getattr(getattr(module, "cuinfer", None), "paged_mqa_logits_int8", None))
assert "sitecustomize" not in sys.modules
assert "vllm_iluvatar" not in sys.modules
assert "_indexer_int8_native" not in sys.modules
print(json.dumps({"native_file": str(target), "native_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                  "torch_version": torch.__version__, "int8_api": True, "temporary_bridge_loaded": False}, indent=2))
PY
```

### 4.3 新主扩展实服验收状态

第二次使用与原服务完全相同的模型参数起服成功，READY耗时555.86秒，launcher PID3973096，服务保留在8000。`retained_service_verified.json` 核实16个worker的真实INT8输入为 `torch.int8`，实际主扩展均为第2节的源码目录新 `_C.so`，未加载旧bridge；9个研发库路径/哈希不变，16个Engram共享inode各由两个DP worker映射，CUDA Graph、DSpark K5 greedy及原启动参数均保持。单图、双图、五图、文本reasoning、tool-call和六图超限拒绝共8/8通过，7次正常生成累计322实际输出tokens，另1次六图返回预期HTTP400；最终health200、队列为空。

首次同参数启动在48/48权重分片加载后，主机可用内存降至23.51GiB，触发24GiB保护停止，未进入请求验收。控制进程同时记录到 `ixsmi pmon` 被SIGKILL；现有证据表明是watchdog终止所属后代，没有新增内核OOM证据，不能表述为已观察到物理OOM。安全清理后，第二次同参数重启成功，可用主存最低采样51.80GiB后回升。两次结果说明启动峰值仍需排查，没有据此认定pin memory或偶发hang根因已修复。首次日志保存在本次目录 `attempt1/`，保护记录为 `attempt1/memory_guard.json`，摘要为 `startup_attempts.json`，资源采样为 `resources.jsonl`。

本次只验证完整开发者安装、正式主扩展及实际请求，不重跑完整性能矩阵或EvalScope；第5节历史精度和第6节历史性能继续保留其原产物口径。

## 5. 当前进度、历史验收与可核对证据

本次完整开发者安装、CPU检查、正式主扩展重启和简单请求验证均通过。当前配置为INT8 indexer、TP8/PP1/DP2/EP16、DSpark K5 greedy、Graph、上下文262144、5图、batch2048，无显式 `--max-num-seqs`，实际默认128。新主扩展的实际请求证据如下；后续列出的全量文本精度仍来自历史“旧镜像主扩展＋bridge”，本次未重跑。

| 本次检查 | 实际结果 |
|---|---|
| 单图A / B | 分别识别4827、红圆/蓝正方形；9153、绿三角形/橙长方形 |
| 双图AB / BA | 两种顺序下数字与颜色形状均按图正确返回 |
| 文本reasoning | 17×19返回323，reasoning字段存在 |
| tool-call | 返回 `get_weather`、city=北京的结构化调用，未执行外部工具 |
| 5张1302×1302图片 | 4827、9153、6308、2746、8062及各图颜色形状均正确；实际输入5004 tokens |
| 6张图片 | HTTP400，提示每请求最多5图，属于预期拒绝 |
| 总计 | 8/8通过；7次正常生成、1次预期拒绝；实际输出322 tokens，health200，队列为空 |

以下为历史功能与精度记录：

历史配置移除显式maxseq64后，默认128起服成功；单图、双图、五图、超限拒绝、reasoning和tool-call功能检查8/8。此前maxseq64阶段还验证过261260实际输入tokens的图文检索，不能将该长请求误写成在默认128阶段重测。

历史文本精度使用旧主扩展＋bridge的INT8/DSpark/Graph服务，EvalScope1.12.0、并发16、三个数据集全量各一次。评测输出上限245760，预留上下文输入空间，所有回答均自然结束。

| 数据集 | 完成数 | 官方score |
|---|---:|---:|
| GPQA Diamond | 198/198 | 89.39% |
| C-Eval | 1346/1346 | 95.17% |
| MATH-500 | 500/500 | 98.00% |

合计2044/2044、3,938,796实际输出tokens；API错误、空答案和长度截断均为0。这是文本任务精度，视觉仅完成上述功能验证。INT8 indexer的9点纯文本性能与FP8对照见第6节，完整记录另见性能速查（源码／外部材料：`../../user/performance_cheatsheet.md#deepseek-v41-flash-w4a8-w4a8`）。

2026-10-21 10:31只读复核确认原launcher PID3519071及16个worker身份一致，16个真实INT8调用记录有效，主扩展/bridge/9个SDK库哈希不变，health200。实际请求返回323，图片返回4827、红圆、蓝正方形。此记录属于完整安装前的历史复核；本次已按进程身份停止旧服务并保留清理证据，随后启动新正式主扩展。

| 证据 | 宿主机路径 |
|---|---|
| 本次完整安装及CPU检查 | `/mnt/share/user_homes/zyp/vllm_iluvatar/logs/functional/v0.30.0/dsv41_editable_native_20261021_104955/` 内 build_install.log（源码／外部材料：`../../../logs/functional/v0.30.0/dsv41_editable_native_20261021_104955/build_install.log`）、build_status.json（源码／外部材料：`../../../logs/functional/v0.30.0/dsv41_editable_native_20261021_104955/build_status.json`）、cpu_preflight.json（源码／外部材料：`../../../logs/functional/v0.30.0/dsv41_editable_native_20261021_104955/cpu_preflight.json`） |
| 本次服务日志、完整启动命令 | 同目录 server.log（源码／外部材料：`../../../logs/functional/v0.30.0/dsv41_editable_native_20261021_104955/server.log`）、launch_command_vllm.sh（源码／外部材料：`../../../logs/functional/v0.30.0/dsv41_editable_native_20261021_104955/launch_command_vllm.sh`）、retained_service.json（源码／外部材料：`../../../logs/functional/v0.30.0/dsv41_editable_native_20261021_104955/retained_service.json`） |
| 新主扩展、16worker真实INT8与SDK核查 | 同目录 retained_service_verified.json（源码／外部材料：`../../../logs/functional/v0.30.0/dsv41_editable_native_20261021_104955/retained_service_verified.json`）、`indexer_execution/` |
| 本次简单请求与实际响应 | 同目录 request_checks.json（源码／外部材料：`../../../logs/functional/v0.30.0/dsv41_editable_native_20261021_104955/request_checks.json`）、additional_checks.json（源码／外部材料：`../../../logs/functional/v0.30.0/dsv41_editable_native_20261021_104955/additional_checks.json`），原始请求/响应在 `requests/`、`additional_requests/`，图片在 `test_images/`、`test_images_large/` |
| 首次保护停止与第二次资源记录 | 同目录 startup_attempts.json（源码／外部材料：`../../../logs/functional/v0.30.0/dsv41_editable_native_20261021_104955/startup_attempts.json`）、首次保护记录（源码／外部材料：`../../../logs/functional/v0.30.0/dsv41_editable_native_20261021_104955/attempt1/memory_guard.json`）、`attempt1/server.log`、`resources.jsonl`、`process_resources.jsonl`、`host_peak_details.jsonl` |
| 历史服务日志、完整启动命令、功能报告 | `/mnt/share/user_homes/zyp/vllm_iluvatar/logs/functional/v0.30.0/dsv41_multimodal_262k_defaultseq_20261020_210629/` 内的 `server.log`、`launch_command_vllm.sh`、`report.md` |
| 历史INT8及请求复核 | 上述历史目录 `service_rechecks/20261021_103104/audit.json`、`worker_int8_evidence.json` 和请求/响应JSON |
| 历史旧主扩展＋bridge的正式文本精度 | `/mnt/share/user_homes/zyp/vllm_iluvatar/accuracy_results/DeepSeek-V4.1-Flash-W4A8-multimodal_20261020_210818/`，含配置、原始预测/判分/报告、日志、`prediction_audit.json` |
| 适配、失败过程与并行限制 | 合并验收记录（源码／外部材料：`../../adaptation_records/deepseek_v41_flash/vllm_0.30.0_dsv41_mixed_marlin_merge_20261020.md`） |
| 精度参数、历史参考与原始分数 | 精度速查（源码／外部材料：`../../user/evaluation_cheatsheet.md#deepseek-v41-flash-w4a8`） |

只读API检查可在服务所在宿主机执行：

```bash
curl --fail --silent --show-error http://localhost/health
curl --fail --silent --show-error http://localhost/v1/models
```

确认INT8不能只看启动开关或 `/proc/<pid>/environ`。此前已复现setproctitle导致该环境采样缺项，实际执行证据要结合worker PID/启动时间、`torch.int8`输入记录和加载库路径。历史核查保存在上表对应目录；本次新runtime记录的首次真实INT8调用及16个worker身份已在 `retained_service_verified.json` 核对通过。

## 6. 已测性能：INT8自身与FP8对照

以下为2026-10-20完成的历史纯文本测试，使用本指南第1节的SDK、新W4A8权重与 **TP8 / PP1 / DP2 / EP16、DSpark K5 greedy、CUDA Graph**。当时主扩展为镜像旧 `_C.so`，INT8接口由临时bridge补齐；图片上限显式为0。**这些数据不代表当前五图多模态模式或本次开发者模式完整编译安装后的主扩展性能。** 本次安装仅做重启和简单请求验证，不重跑性能矩阵或精度。

### 6.1 测试配置与统计口径

| 项目 | 实测配置 |
|---|---|
| 硬件 / 软件 | 16×Iluvatar BI-V150；CoreX/Driver 5.0.0；Python 3.12.11；Torch 2.10.0；vLLM / 插件0.30.0 |
| 权重 / 代码 | `DeepSeek-V4.1-Flash-W4A8`；生产源码与 `8b8187b7` 一致；研发ixformer `08a47e1f`、ixinfer `aca6f354` |
| 并行 / Indexer | TP8 / PP1 / DP2 / EP16；`VLLM_ILUVATAR_INDEXER_INT8=1` |
| Graph / 上下文 | FULL_DECODE_ONLY，capture cap256；上下文262144；batch tokens2048；`gpu-memory-utilization=0.83` |
| Engram / 请求入口 | CPU offload、DP共享内存；关闭prefix cache；`--limit-mm-per-prompt '{"image":0}'`；纯文本 `/v1/completions` |
| DSpark / 通信 | K5 greedy；IXFD_COMM=1；NCCL allreduce:tree / Simple；`TORCH_NCCL_BLOCKING_WAIT=1` |
| 请求参数 | random数据；每点输出2048 tokens；`--ignore-eos --temperature 0.0 --seed 1000`；正式请求数为并发×2 |
| 轮数 / 完整性 | 每点各一轮，共9/9点、76/76请求、155,648实际输出tokens；每个请求均完整输出2048 tokens；缺失、无效、失败均为0 |

历史性能测试的其余固定开关为 `VLLM_ILUVATAR_DSPARK_SPLIT_SAMPLE=1`、`VLLM_ILUVATAR_DSV41_DISABLE_AUX_STREAM=1`、`VLLM_ILUVATAR_DSA_PERSISTENT_PAGES=1`、`VLLM_ILUVATAR_W4A8_FILTER_EP=1`、`VLLM_ILUVATAR_W4A8_FILTER_EP_PREFILL=0`。

输出tok/s是16卡服务在每个完整测试点期间的平均输出吞吐，不是瞬时峰值。TTFT/TPOT列为逐请求均值与P99，E2E为逐请求端到端耗时均值；每点仅2–16个请求，P99只描述本轮样本。原4点与补测5点复用同一服务，未在中间重启。

### 6.2 INT8 indexer自身完整性能（9点）

| 输入 / 输出 tokens | 并发 | 成功/请求 | 实际输出 tokens | 输出 tok/s | TTFT 均值/P99（秒） | TPOT 均值/P99（ms） | 平均E2E（秒） |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192/2048 | 1 | 2/2 | 4,096 | 79.71 | 5.41 / 5.43 | 9.91 / 10.02 | 25.69 |
| 8192/2048 | 4 | 8/8 | 16,384 | 158.22 | 9.67 / 17.35 | 18.09 / 27.78 | 46.70 |
| 8192/2048 | 8 | 16/16 | 32,768 | 204.78 | 15.94 / 31.61 | 26.39 / 34.65 | 69.97 |
| 16384/2048 | 1 | 2/2 | 4,096 | 64.29 | 10.55 / 10.59 | 10.41 / 10.44 | 31.86 |
| 16384/2048 | 4 | 8/8 | 16,384 | 129.98 | 20.38 / 34.79 | 20.43 / 35.07 | 62.20 |
| 16384/2048 | 8 | 16/16 | 32,768 | 145.11 | 27.05 / 63.52 | 38.26 / 84.07 | 105.38 |
| 32768/2048 | 2 | 4/4 | 8,192 | 64.30 | 29.81 / 42.77 | 16.45 / 21.53 | 63.49 |
| 32768/2048 | 8 | 16/16 | 32,768 | 90.71 | 53.97 / 130.63 | 60.41 / 109.60 | 177.63 |
| 60000/2048 | 2 | 4/4 | 8,192 | 38.76 | 54.79 / 82.57 | 24.66 / 32.60 | 105.27 |

### 6.3 FP8 / INT8 indexer对比（9个对应点）

FP8/INT8指Indexer Q/K量化与scoring路径，两组模型均使用同一份新W4A8权重。FP8来自此前15个有效点中的对应9点，生产代码、研发SDK、并行、DSpark、Graph及请求参数保持一致；9点完整argv除结果路径外一致。箭头为FP8→INT8，TTFT/TPOT均为请求均值；每点各一轮，变化百分比仅描述本次观察值。

| 输入 / 输出 tokens | 并发 | 输出吞吐 tok/s（FP8→INT8） | 吞吐变化 | 平均TTFT 秒（FP8→INT8） | 平均TPOT ms（FP8→INT8） |
|---:|---:|---:|---:|---:|---:|
| 8192/2048 | 1 | 59.97 → 79.71 | +32.9% | 5.88 → 5.41 | 13.81 → 9.91 |
| 8192/2048 | 4 | 118.74 → 158.22 | +33.3% | 12.33 → 9.67 | 27.55 → 18.09 |
| 8192/2048 | 8 | 139.58 → 204.78 | +46.7% | 16.44 → 15.94 | 48.47 → 26.39 |
| 16384/2048 | 1 | 33.09 → 64.29 | +94.3% | 12.23 → 10.55 | 24.26 → 10.41 |
| 16384/2048 | 4 | 74.08 → 129.98 | +75.5% | 20.13 → 20.38 | 43.32 → 20.43 |
| 16384/2048 | 8 | 75.75 → 145.11 | +91.6% | 43.33 → 27.05 | 82.17 → 38.26 |
| 32768/2048 | 2 | 45.05 → 64.30 | +42.7% | 37.62 → 29.81 | 25.87 → 16.45 |
| 32768/2048 | 8 | 41.07 → 90.71 | +120.9% | 96.83 → 53.97 | 145.54 → 60.41 |
| 60000/2048 | 2 | 18.45 → 38.76 | +110.0% | 86.57 → 54.79 | 66.04 → 24.66 |

9个对应点的输出吞吐观察值提高32.9%–120.9%，平均TPOT均降低；16384输入、并发4的平均TTFT由20.13变为20.38秒，单轮看基本持平，不作统计显著性判断。DSpark接受行为也有变化，总体收益不能全部归因于单个Indexer kernel；这些数据也没有重新比较其他TP/PP/DP拓扑，不能据此推断未覆盖负载的收益。

### 6.4 原始命令、日志与数据

下表路径为宿主机相对仓库根目录 `/mnt/share/user_homes/zyp/vllm_iluvatar/` 的位置；容器根目录对应 `/workspace/vllm_iluvatar/`。原始目录保留，不覆盖已有记录。

| 内容 | 原始路径 |
|---|---|
| INT8完整9点独立表、FP8/INT8对照表 | `logs/perf/v0.30.0/dsv41_int8_supplement_20261020_1848/combined_audited_measurements.csv`、`combined_comparison_to_fp8_baseline.csv` |
| 请求/token完整性、逐点可比性、18份来源JSON哈希 | 同目录 `combined_audit.json`、`combined_comparability.json`、`combined_source_manifest.json` |
| INT8原4点、原始启动命令和完整服务日志 | `logs/perf/v0.30.0/dsv41_merged_sdk_indexer_int8_perf_after_20261020_152607/` 内 `serve.sh`、`launch_command.txt`、`server.log`、`round1/` |
| INT8补测5点、累计报告 | `logs/perf/v0.30.0/dsv41_int8_supplement_20261020_1848/` 内 `runner.log`、`performance_report.md`、`round1/`；其 `server.log` 链接到上述同一服务日志 |
| FP8原始15个有效点（本节取对应9点） | `logs/perf/v0.30.0/dsv41_merged_sdk_perf_20261020_152607/` 内 `server.log`、`audit.json`、`round1/` |

各点详细结果位于 `round1/concurrency<C>_prompts<2C>_input<I>_output2048.json`；完整客户端命令与stdout/stderr分别为 `round1/logs/` 下的同名 `.argv.json` 和 `.log`。历史纯文本启动命令以原始 `launch_command.txt` 为准，当前多模态部署使用第3节命令。

## 7. 已知限制

- 清理后的 DeepSeek-V4.1 DSpark 仅支持 PP1；PP>1 + DSpark 在配置阶段明确报错。通用元数据、Graph捕获输入和采样兼容修复已独立为 `patch_dspark.py`，普通不带DSpark的PP功能保留。历史PP2+DSpark结果仅代表当时版本。

- `Failed to pin system memory` 及偶发启动hang的完整根因尚未确认；当前成功运行及精度完成不代表该问题已修复。
- batch2048是当前已验收配置；之前改成4096明显减少可用KV。只把图片上限设为5，并不会自动把当时的4图encoder预热降为1图。
- TP16受模型 `o_groups=8` 切分限制；其他TP/PP/DP候选的实际资源限制见合并验收记录。这里提供已测试的推荐配置，不宣称对所有负载最优。
- 当前部署仍依赖研发SDK和ixformer runtime链接，已不依赖临时bridge。完整开发者安装、CPU检查与正式主扩展实服冒烟均通过；本次没有重跑该新二进制的完整性能或精度，历史数据不替代这些测量。

本次PP支持范围收缩及代码清理仅完成CPU验证，按要求没有重启当前服务。运行中的INT8、DSpark K5服务继续使用重启前已加载的代码与原始参数，不能把已有GPU验收重标为清理后GPU验收。历史性能与精度记录保留原产物口径。
