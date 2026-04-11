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

补充说明：

- 上表中的吞吐结论仍然可以用于比较不同 Pythia 配对的整体表现。
- 但其中旧结果里的 `draft_target_latency_ratio` 采用的是较早的近似口径，不再建议把它当作论文里 `c` 的主分析依据。
- 当前版本已经把 `draft_target_latency_ratio` 重新实现为严格单步 decode latency 比：`draft_decode_step_time / target_decode_step_time`。

基于新口径做的 Pythia strict smoke test（`greedy + k=4 + max_new_tokens=32 + repeats=1`）结果如下：

| pair | target decode step time (s) | draft decode step time (s) | strict `draft_target_latency_ratio` |
| --- | ---: | ---: | ---: |
| `pythia-14m -> pythia-1.4b` | `0.03178` | `0.00852` | `0.268` |
| `pythia-31m -> pythia-1.4b` | `0.03333` | `0.00862` | `0.259` |
| `pythia-70m -> pythia-1.4b` | `0.03455` | `0.00888` | `0.257` |
| `pythia-160m -> pythia-1.4b` | `0.03282` | `0.01647` | `0.502` |

这组 strict smoke test 说明：

- `14m / 31m / 70m` 的单步 draft decode latency 在当前机器上确实非常接近，并不是旧统计口径单独造成的假象。
- `31m` 与 `70m` 的 strict latency ratio 略低于 `14m`，但差距不大，说明小模型区间的参数差异没有明显转化为单步延迟差异。
- 真正明显恶化的是 `160m`，其 draft 单步 decode 时间几乎翻倍，导致 `draft_target_latency_ratio` 跳到 `0.502`。
- 因此，在当前硬件与实现下，Pythia 系列更像是存在一个“小模型延迟平台区”，而不是参数越小、单步 decode 就线性越快。

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

两类结果文件的区别：

- `bench_raw.csv`：每一行对应一次实际运行结果
- `bench_summary.csv`：对相同配置的多次运行做聚合统计；数值字段会展开为 `*_mean` 与 `*_std`

其中：

- `*_mean` 表示该配置下多次重复运行的平均值
- `*_std` 表示该配置下多次重复运行的标准差

## 统计字段说明

### 1. 配置字段

- `prompt_id`：prompt 标识符，例如 `explain_sd_simple`、`sampling_diff`
- `mode`：运行模式；`baseline` 表示普通解码基线，`speculative` 表示推测解码
- `baseline_impl`：baseline 实现方式；`hf` 表示 Hugging Face `generate()` 快路径，`manual` 表示手写自回归实现
- `draft_model`：草稿模型名称；baseline 行通常为空
- `target_model`：目标模型名称
- `strategy`：解码策略；当前支持 `greedy`、`top_k`、`top_p`
- `candidate_length_k`：每轮 speculative 提出的候选长度 `k`；baseline 行通常为空
- `top_k`：仅在 `strategy=top_k` 时有效，表示 top-k 采样的截断大小
- `top_p`：仅在 `strategy=top_p` 时有效，表示 nucleus sampling 的累计概率阈值
- `temperature`：采样温度；`greedy` 路径下通常固定为 `1.0`
- `n`：仅出现在 `bench_summary.csv` 中，表示该配置汇总了多少次运行

### 2. 输入与输出规模字段

- `prompt_chars`：单次运行中 prompt 的字符数
- `prompt_chars_mean/std`：prompt 字符数的均值与标准差；通常标准差为 0，因为同一 `prompt_id` 的输入长度固定
- `generated_tokens`：单次运行实际生成的新 token 数
- `generated_tokens_mean/std`：生成 token 数的均值与标准差

### 3. 端到端性能字段

- `total_generation_time`：端到端总耗时，单位为秒
- `total_generation_time_mean/std`：端到端总耗时的均值与标准差
- `tokens_per_s`：吞吐率，即每秒生成 token 数；值越高越好
- `tokens_per_s_mean/std`：吞吐率的均值与标准差

### 4. speculative 专属统计字段

- `proposed_tokens`：草稿模型总共提出的候选 token 数
- `proposed_tokens_mean/std`：候选 token 数的均值与标准差
- `accepted_tokens`：被目标模型验证后接受的草稿 token 数
- `accepted_tokens_mean/std`：接受 token 数的均值与标准差
- `acceptance_rate`：接受率，定义为 *accepted_tokens / proposed_tokens*
- `acceptance_rate_mean/std`：接受率的均值与标准差
- `avg_accepted_prefix_length`：平均接受前缀长度，定义为每轮验证中连续被接受的前缀长度平均值
- `avg_accepted_prefix_length_mean/std`：平均接受前缀长度的均值与标准差
- `rejection_events`：实际发生拒绝的轮次数
- `rejection_events_mean/std`：拒绝事件数的均值与标准差
- `verify_rounds`：进行 speculative 验证的轮次数
- `verify_rounds_mean/std`：验证轮次数的均值与标准差

这几个字段的阅读方式通常是：

- `acceptance_rate` 越高，说明草稿模型与目标模型越一致
- `avg_accepted_prefix_length` 越高，说明每轮验证能连续吞下的候选前缀越长
- `rejection_events` 越高，说明生成过程中更频繁出现“候选被打断、需要纠正”的情况

### 5. 阶段耗时拆解字段

- `draft_time`：草稿模型生成候选所花的总时间
- `verify_time`：目标模型验证候选所花的总时间
- `rebuild_time`：发生拒绝后，裁剪 cache、接入纠正 token 并恢复状态推进所花的总时间
- `draft_time_mean/std`、`verify_time_mean/std`、`rebuild_time_mean/std`：分别是上述阶段耗时的均值与标准差
- `draft_time_ratio`：提议时间的占比，定义为 *draft_time / total_generation_time*
- `verify_time_ratio`：验证时间的占比，定义为 *verify_time / total_generation_time*
- `rebuild_time_ratio`：状态更新时间的占比，定义为 *rebuild_time / total_generation_time*
- `draft_time_ratio_mean/std`、`verify_time_ratio_mean/std`、`rebuild_time_ratio_mean/std`：分别是三类时间占比的均值与标准差
- `target_decode_step_time`：目标模型在已有 KV cache 上执行一次单 token decode step 的平均耗时
- `draft_decode_step_time`：草稿模型在已有 KV cache 上执行一次单 token decode step 的平均耗时
- `draft_target_latency_ratio`：原始论文里的延迟比 `c`，定义为 *draft_decode_step_time / target_decode_step_time*；当前版本按严格单步 decode latency 口径统计
- `target_decode_step_time_mean/std`、`draft_decode_step_time_mean/std`、`draft_target_latency_ratio_mean/std`：上述单步耗时与延迟比的均值与标准差

这些字段主要用于定位瓶颈：

- `draft_time_ratio` 高，说明瓶颈更偏向草稿模型提案阶段
- `verify_time_ratio` 高，说明瓶颈更偏向目标模型验证阶段
- `rebuild_time_ratio` 高，说明拒绝后的状态恢复成本仍然明显
- `draft_target_latency_ratio` 越小，越接近论文中“draft 明显快于 target”的理想条件；如果它偏大，即使参数量差很多，也未必能带来加速

### 6. 历史字段说明

- `output_exact_match`：是否与参考 baseline 输出完全一致；通常为 `0` 或 `1`
- `output_prefix_match_len`：与参考 baseline 从开头开始连续匹配的 token 长度
- `output_prefix_match_ratio`：前缀匹配长度占参考 baseline 输出长度的比例

说明：

- 新生成的 benchmark 结果已经移除 `output_exact_match`、`output_prefix_match_len`、`output_prefix_match_ratio`
- 旧实验目录中的历史 CSV 可能仍保留这些字段，它们仅代表旧口径结果，不建议继续作为当前版本的核心分析指标

当前 `run_bench.py` 默认会同时运行 `hf` 与 `manual` 两种 baseline，便于分别观察：

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

如果想复现实验，推荐显式使用当前 torch 环境。下面给出一个 GPT-2 示例和两个 Pythia 示例：

```powershell
D:\MiniConda\envs\torch\python.exe run_bench.py --model_pair distilgpt2_gpt2 --prompt_preset analysis --warmup 1 --repeats 3 --cache_dir D:\hf_cache --local_files_only --out_dir results/distilgpt2_gpt2
D:\MiniConda\envs\torch\python.exe run_bench.py --model_pair pythia_31m_1.4b --prompt_preset analysis --warmup 1 --repeats 3 --cache_dir D:\hf_cache --local_files_only --out_dir results/pythia_31m_1.4b
D:\MiniConda\envs\torch\python.exe run_bench.py --model_pair pythia_31m_1.4b --draft_quantization 8bit --prompt_preset analysis --warmup 1 --repeats 3 --cache_dir D:\hf_cache --local_files_only --out_dir results/pythia_31m_1.4b_8bit
D:\MiniConda\envs\torch\python.exe run_bench.py --model_pair pythia_14m_1.4b --prompt "Explain speculative decoding in simple terms." --strategies greedy --ks 4 --warmup 1 --repeats 1 --max_new_tokens 32 --cache_dir D:\hf_cache --local_files_only --out_dir results/smoke_strict_pythia_14m_1.4b
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

1. 概念解释类
`Explain speculative decoding in simple terms.`
2. 对比分析类
`Compare greedy decoding, top-k sampling, and top-p sampling in one paragraph.`
3. 机制说明类
`Explain how draft models and target models interact during speculative decoding.`
4. 原因分析类
`Why can a low acceptance rate reduce the speedup of speculative decoding?`
