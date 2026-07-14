# IndexKV

KV-cache **索引选择方法**的统一公平评测框架。

对每个方法只做一件事的对比:给定一层的 prompt K/V/Q(**只捕获一次**),每种方法构建一个与预算无关的每层索引,在**固定 token 预算**下决定解码时每一步能注意到哪些 prompt 位置。其余环节(捕获、mask、稀疏 SDPA 解码、打分)全部共享,所以分数衡量的是**索引质量**,而非工程实现。

> 本仓库是 KV-cache 索引选择方法的**公平评测框架**,内置有源码支撑的 baseline。

## 方法

| 名称 | 来源 | 选择方式 |
|---|---|---|
| `full` | 稠密参考(上界) | 全注意力 |
| `chunkkv` | kvpress ChunkKV | 静态块级(SnapKV 打分) |
| `hashattention` | HashAttention-1.0 | 逐头 · 学习哈希码 top-k · **需权重** |
| `hata` | HATA | 逐头 · 训练哈希码 · 组内 Hamming top-k · **需权重** |
| `range_search` | Louver | 逐头 · **采样估计阈值**半空间检索 |
| `selfindexing` | selfindexingkv | 逐头 · **符号正交 VQ/LUT** 近似 q·k top-k |
| `wave_index` | RetrievalAttention / RetroInfer | 分段 k-means · 组内 softmax 选簇 |

每个 adapter 都忠实复现对应上游方法**实际选中的位置集合**;纯系统实现(kernel、ball tree、bit-packing 等)被排除,但绝不改变"选哪些位置"。公平性(统一预算、`sink=4`/`recent=32`、`dense_prefix_layers=2`、共享捕获)由 `engine.py` 在运行时强制保证。

## 安装

```bash
pip install -e .          # torch, transformers>=5.0, datasets
pip install accelerate    # 用 device_map 加载模型时需要
```

## 快速开始

```bash
python scripts/run_ruler.py --list_methods    # 列出方法
```

**RULER**(需预先生成 `<data_root>/<task>/validation.jsonl`;prompt 按 kvpress 方式套 Llama-3.1 chat template):

```bash
python scripts/run_ruler.py \
  --model /path/to/Llama-3.1-8B-Instruct \
  --methods full chunkkv hashattention hata range_search selfindexing wave_index \
  --budgets 1024 --num_samples 50 \
  --data_root /path/to/ruler_8k \
  --out_dir results/ruler_demo
```

**LongBench v2**(自动拉取 `THUDM/LongBench-v2`;`--resume` 断点续跑):

```bash
python scripts/run_longbench2.py \
  --model /path/to/Llama-3.1-8B-Instruct \
  --methods full chunkkv range_search \
  --budgets 1024 --num_samples 50 --out_dir results/longbench2_demo
```

方法旋钮用 `--set 方法.参数=值`,例如 `--set hata.rbits=256 range_search.sample_size=256`。

## 评测忠实度

- **RULER**:打分(`qa_*` 用 part-match,其余 all-match、控制符清洗、逐任务 `max_new_tokens`)与 kvpress / NVIDIA RULER 逐字一致,prompt 与 kvpress `pipeline.preprocess` 同样套 chat template。
- **LongBench v2**:官方 THUDM 零样本/CoT 模板、中间截断、答案抽取与 `result.py` 打分。

## 权重

`hashattention` 与 `hata` 需要训练好的索引权重(因体积过大未随仓库分发)。放到 `weights/hashattention/`、`weights/hata/<model>-<rbits>/` 下,或用 `--set <方法>.weights_path=...` 指定;缺权重的方法会被自动跳过(`weights-missing`),其余方法照常运行。权重可从 HashAttention-1.0 / HATA 官方仓库获取或按其脚本训练。

## 测试

```bash
python -m pytest tests/ -q    # 纯 CPU,无需下载模型
```
