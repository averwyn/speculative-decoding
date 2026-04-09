# feasibleDraft

本文围绕单层异构 speculative decoding 展开实验。当前代码同时支持三类实验设计：

- Qwen 系列实验：`Qwen/Qwen2.5-0.5B -> Qwen/Qwen2.5-1.5B`
- GPT-2 系列实验：`distilgpt2 -> gpt2`、`gpt2 -> gpt2-medium`
- Pythia 系列实验：以 `EleutherAI/pythia-1.4b` 为目标模型，逐步比较不同规模草稿模型

代码默认以 Qwen 配对作为默认预设，但现在已经支持通过模型对预设快速切换，不需要每次手动填写 `--draft` 和 `--target`。

## 可用模型对预设

- `qwen_main`: `Qwen/Qwen2.5-0.5B -> Qwen/Qwen2.5-1.5B`
- `distilgpt2_gpt2`: `distilgpt2 -> gpt2`
- `gpt2_gpt2_medium`: `gpt2 -> gpt2-medium`
- `pythia_410m_1.4b`: `EleutherAI/pythia-410m -> EleutherAI/pythia-1.4b`
- `pythia_160m_1.4b`: `EleutherAI/pythia-160m -> EleutherAI/pythia-1.4b`
- `pythia_70m_1.4b`: `EleutherAI/pythia-70m -> EleutherAI/pythia-1.4b`
- `pythia_31m_1.4b`: `EleutherAI/pythia-31m -> EleutherAI/pythia-1.4b`
- `pythia_14m_1.4b`: `EleutherAI/pythia-14m -> EleutherAI/pythia-1.4b`

## Qwen 系列配对示例


```powershell
python run_bench.py --model_pair qwen_main --prompt_preset analysis --warmup 1 --repeats 3 --out_dir results/qwen_main
```

## GPT-2 系列配对示例

```powershell
python run_bench.py --model_pair distilgpt2_gpt2 --prompt_preset analysis --warmup 1 --repeats 3 --out_dir results/distilgpt2_gpt2
python run_bench.py --model_pair gpt2_gpt2_medium --prompt_preset analysis --warmup 1 --repeats 3 --out_dir results/gpt2_gpt2_medium
```

显式展示草稿模型与目标模型的写法如下：

```powershell
python run_bench.py --draft distilgpt2 --target gpt2
python run_bench.py --draft gpt2 --target gpt2-medium
```

## Pythia 系列配对示例

```powershell
python run_bench.py --model_pair pythia_410m_1.4b --prompt_preset analysis --warmup 1 --repeats 3 --cache_dir D:\hf_cache --local_files_only --out_dir results/pythia_410m_1.4b
python run_bench.py --model_pair pythia_160m_1.4b --prompt_preset analysis --warmup 1 --repeats 3 --cache_dir D:\hf_cache --local_files_only --out_dir results/pythia_160m_1.4b
python run_bench.py --model_pair pythia_70m_1.4b --prompt_preset analysis --warmup 1 --repeats 3 --cache_dir D:\hf_cache --local_files_only --out_dir results/pythia_70m_1.4b
python run_bench.py --model_pair pythia_31m_1.4b --prompt_preset analysis --warmup 1 --repeats 3 --cache_dir D:\hf_cache --local_files_only --out_dir results/pythia_31m_1.4b
python run_bench.py --model_pair pythia_14m_1.4b --prompt_preset analysis --warmup 1 --repeats 3 --cache_dir D:\hf_cache --local_files_only --out_dir results/pythia_14m_1.4b
```

## Pythia 阶段性结果

目前已经完成 4 组 Pythia 配对实验，目标模型固定为 `EleutherAI/pythia-1.4b`，草稿模型分别为 `14m`、`31m`、`70m`、`160m`。

整体观察如下：

- `pythia-14m -> pythia-1.4b`：整体 speculative 结果明显低于 greedy baseline
- `pythia-31m -> pythia-1.4b`：当前四组中表现最好，最接近整体打平 baseline
- `pythia-70m -> pythia-1.4b`：未继续优于 `31m`，说明草稿模型增大后生成开销开始吞噬收益
- `pythia-160m -> pythia-1.4b`：进一步退化，说明在该 target 下更大的 draft 模型并不划算

按 `manual + greedy + ALL` 口径汇总如下：

| pair | baseline tokens/s | best speculative tokens/s | best speculative setting |
| --- | ---: | ---: | --- |
| `pythia-14m -> pythia-1.4b` | `35.24` | `28.67` | `greedy, k=4` |
| `pythia-31m -> pythia-1.4b` | `33.99` | `31.66` | `greedy, k=6` |
| `pythia-70m -> pythia-1.4b` | `35.28` | `29.94` | `greedy, k=6` |
| `pythia-160m -> pythia-1.4b` | `34.81` | `26.44` | `greedy, k=4` |

如果看局部最优而不是总体平均，四组都在 `draft_target_interaction` 这个 prompt 上出现最明显加速：

- `14m -> 1.4b`：最好达到 `44.80 tok/s`，配置为 `greedy, k=8`
- `31m -> 1.4b`：最好达到 `51.15 tok/s`，配置为 `greedy, k=8`
- `70m -> 1.4b`：最好达到 `39.72 tok/s`，配置为 `greedy, k=6`
- `160m -> 1.4b`：最好达到 `36.84 tok/s`，配置为 `greedy, k=8`

当前阶段的结论是：

- `greedy` 仍然是最适合 speculative decoding 的策略
- `top_k` 与 `top_p` 在这四组 Pythia 配对上都没有稳定超过 baseline
- 草稿模型不是越大越好，`31m` 目前是速度与接受率之间最平衡的候选
- 从 `31m -> 70m -> 160m` 的退化趋势看，当前瓶颈已经明显转向草稿模型生成开销
- 如果继续做 Pythia 系列探索，优先值得围绕 `pythia-31m -> pythia-1.4b` 继续细调 `k`

额外尝试：

- 已为代码加入 draft-only 量化入口：`--draft_quantization {none,8bit,4bit}`
- 在当前环境（Windows + RTX 3060 Laptop GPU 6GB + `bitsandbytes`）下，`pythia-31m -> pythia-1.4b` 的 `8bit` smoke test 没有观察到正收益，draft 侧耗时反而上升
- `4bit` smoke test 相比 `8bit` 略有改善，draft 侧耗时也更低，但整体仍未显示出足够明显的加速收益
- 因此，当前阶段不建议把 `bitsandbytes` 量化作为这一配对的主要提速方向

这些预设同时可用于：

- `run_bench.py`
- `run_experiment.py`
- `baseline_generate.py`

## 实验说明

当前实现假设 draft 模型与 target 模型可以共享 target 侧 tokenizer 家族。这个假设对当前几类实验都成立：

- Qwen2.5 系列共享同族 tokenizer
- Gpt2 系列共享 GPT-2 tokenizer
- Pythia 系列共享同族 tokenizer

正式 benchmark 默认会遍历：

- 解码策略：`greedy`、`top_k`、`top_p`
- 候选长度：`k = 1,2,4,6,8`
- prompt 集：`analysis`
- baseline 实现：`hf`、`manual`

输出文件为：

- 原始结果：`results/.../bench_raw.csv`
- 汇总结果：`results/.../bench_summary.csv`

当前结果文件中保留的核心指标包括：

- 吞吐：`tokens_per_s`
- 接受率：`acceptance_rate`
- 平均接受前缀长度：`avg_accepted_prefix_length`
- 阶段耗时拆解：`draft_time`、`verify_time`、`rebuild_time`

说明：

- 新生成的 benchmark 结果已经移除 `output_exact_match`、`output_prefix_match_len`、`output_prefix_match_ratio`
- 旧实验目录中的历史 CSV 可能仍保留这些字段，它们仅代表旧口径结果，不建议继续作为当前版本的核心分析指标

其中：

- `hf` baseline 表示 Hugging Face `generate()` 快路径
- `manual` baseline 表示手写自回归解码 baseline

当前 `run_bench.py` 默认会同时运行这两种 baseline，便于分别观察：

- 推测解码与工程优化后的 `generate()` 快路径之间的差距
- 推测解码与手写自回归实现之间的公平对比结果

## 优化总结

本轮实现工作主要围绕“降低推测解码额外工程开销”展开，核心改动如下：

1. 统一并修正实验口径

- benchmark 结果以吞吐、接受率、平均接受前缀长度和阶段耗时拆解为主，不再把输出逐 token 一致性作为通用统计指标
- `rejection_events` 表示实际发生拒绝的轮次数，而不是被拒绝的 token 数
- `local_files_only` 改为仅在显式传参时启用，避免默认离线模式影响实验
- `prompt_preset` 改为显式校验，避免拼写错误时静默回退到默认 prompt 集
- 新增 tokenizer 家族一致性保护，防止跨 tokenizer 家族模型组合被误用于当前实现

2. 压缩推测解码主流程开销

- 去掉验证阶段不必要的 CPU/GPU 来回搬运
- 去掉循环中频繁的整段 `input_ids` 拼接，仅在确有需要时临时重建
- 为 `greedy` 路径增加专门快路径，避免重复 softmax

3. 降低拒绝后的状态恢复成本

- target 模型拒绝后不再整段重建 cache，而是裁剪验证得到的 cache 到已接受前缀，再接入纠正 token 继续推进
- draft 模型拒绝后也不再整段重建，而是裁剪已有草稿 cache 后继续推进
- 当某一轮在加入已接受 token 或纠正 token 后已经达到 `max_new_tokens` 时，直接结束生成，不再额外执行仅用于下一轮准备的末轮 rebuild

4. 优化采样路径

- `top_k` 路径改为局部 softmax 版本，只在 top-k 支持集上执行概率归一化和采样
- `top_p` 路径去掉一次不必要的全词表 softmax，直接复用 nucleus 截断后的概率重新归一化

5. 增加开销拆解能力

- benchmark 输出中新增 `draft_time`、`verify_time`、`rebuild_time`
- 同时新增三者相对于总时间的占比字段，便于后续论文中做瓶颈分析
- 在 GPU 环境下，对分段前向调用采用同步计时口径，使 `draft_time`、`verify_time`、`rebuild_time` 的统计结果更接近真实阶段耗时

## 当前结论

结合 GPT-2 与 Pythia 两组补充实验，当前实现的主要结论如下：

- 经过多轮优化后，`greedy` 配置下的一级推测解码已经能够明显接近 baseline，且在部分 prompt 和候选长度下可局部超过 baseline
- `top_k` 和 `top_p` 路径经过针对性优化后也有提升，但整体仍更容易受到接受率下降和草稿侧采样成本的影响
- 当前最主要的性能瓶颈已经不是目标模型验证阶段，而是草稿模型候选生成阶段，即 `draft_time`
- 拒绝后的 cache 恢复开销虽然仍存在，但相较初始实现已经显著下降；在当前实现中，`rebuild_time` 主要反映前缀裁剪后的必要状态推进，而不再包含整段 cache 重建
- 对 `pythia-1.4b` 而言，草稿模型存在明显的“过大反而变慢”现象，目前 `pythia-31m` 是已测试配对中最平衡的候选
- 已尝试为 draft 模型接入 `bitsandbytes` 量化；其中 `4bit` 比 `8bit` 更有希望，但在当前硬件与环境组合下，两者都尚未带来决定性正收益

这说明：一级推测解码能否带来明显加速，不仅取决于算法流程本身，也强烈依赖模型配对关系与工程实现质量。若目标模型本身不够慢，或者草稿模型提案成本仍然较高，则推测解码的理论收益会被额外开销抵消。

## 推荐运行方式

如果想复现实验，推荐显式使用当前 torch 环境。下面给出一个 GPT-2 示例和一个 Pythia 示例：

```powershell
D:\MiniConda\envs\torch\python.exe run_bench.py --model_pair distilgpt2_gpt2 --prompt_preset analysis --warmup 1 --repeats 3 --cache_dir D:\hf_cache --local_files_only --out_dir results/distilgpt2_gpt2
D:\MiniConda\envs\torch\python.exe run_bench.py --model_pair pythia_31m_1.4b --prompt_preset analysis --warmup 1 --repeats 3 --cache_dir D:\hf_cache --local_files_only --out_dir results/pythia_31m_1.4b
D:\MiniConda\envs\torch\python.exe run_bench.py --model_pair pythia_31m_1.4b --draft_quantization 8bit --prompt_preset analysis --warmup 1 --repeats 3 --cache_dir D:\hf_cache --local_files_only --out_dir results/pythia_31m_1.4b_8bit
```

如果只想快速检查 baseline 差异，也可以单独运行：

```powershell
D:\MiniConda\envs\torch\python.exe baseline_generate.py --model_pair distilgpt2_gpt2 --impl hf
D:\MiniConda\envs\torch\python.exe baseline_generate.py --model_pair distilgpt2_gpt2 --impl manual
```

## 论文表述

1. Qwen 配对 `Qwen/Qwen2.5-0.5B -> Qwen/Qwen2.5-1.5B` 用于分析单层异构 speculative decoding 在同系列不同参数规模模型之间的表现。
2. GPT-2 配对 `distilgpt2 -> gpt2` 与 `gpt2 -> gpt2-medium` 用于验证方法在经典小规模自回归模型上的可迁移性，并观察草稿模型与目标模型规模差异缩小时的表现变化。
3. Pythia 配对以 `EleutherAI/pythia-1.4b` 为统一目标模型，系统比较 `14m / 31m / 70m / 160m` 草稿模型，用于观察草稿模型规模、接受率与吞吐之间的非单调关系。

## 当前 prompt 集

（1）概念解释类

1. `Explain speculative decoding in simple terms.`

（2）对比分析类

2. `Compare greedy decoding, top-k sampling, and top-p sampling in one paragraph.`

（3）机制说明类

3. `Explain how draft models and target models interact during speculative decoding.`

（4）原因分析类

4. `Why can a low acceptance rate reduce the speedup of speculative decoding?`
