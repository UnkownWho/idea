# SURE Code Audit

审计目标：判断该仓库是否适合作为“不完整 + 不对齐多视图聚类”的 PyTorch 实验底座。

结论先行：该仓库适合半复用。建议复用数据读取思路、PVP/PSP 构造参考、基础 encoder-decoder、KMeans/ACC/NMI/ARI 指标代码；不建议直接在现有训练与评估流程上改新模型。主要原因是当前代码强绑定两视图、成对对比训练、batch 内补全/重配对推理，并且没有显式保留 global sample id。

## 1. 项目入口

关键文件：

- `run.py`
- `README.md`
- `data_loader.py`
- `models.py`
- `sure_inference.py`
- `Clustering.py`

主入口是 `run.py`，运行方式见 `README.md` Demo。典型命令：

```bash
python run.py --data 0 --gpu 0 --settings 2 --aligned-prop 0.5 --complete-prop 0.5
```

参数定义位置：`run.py:16-37`。

主要参数：

- `--data`: 数据集编号，`run.py:17-19` 中映射到 `data_name`，实际内置可用数据为 `Scene15` 和 `Reuters_dim10`。
- `--gpu`: GPU id，设置 `CUDA_VISIBLE_DEVICES`。
- `--settings`: `0-PVP`, `1-PSP`, `2-Both`。
- `--aligned-prop`: 已知对齐样本比例，越低表示 PVP 越强。
- `--complete-prop`: 完整样本比例，越低表示 PSP 越强。
- `--batch-size`: 默认 1024。
- `--epochs`: 默认 80。
- `--learn-rate`: 默认 1e-3。
- `--lam`: reconstruction loss 权重。
- `--neg-prop`: 每个正样本构造的负样本数量。
- `--margin`, `--robust`, `--switching-time`, `--start-fine`: noise-robust contrastive loss 相关。

随机种子：

- `run.py:176-181` 固定 `np.random.seed(0)`, `torch.manual_seed(0)`, `torch.cuda.manual_seed(0)`。
- 但 `random.seed(seed)` 被注释，`data_loader.load_data()` 里又用 `random.randint(1, 1000)` 生成 `divide_seed`，所以 PVP/PSP 构造默认不完全可复现。

启动风险：

- Windows 下 `run.py:211-212` 日志文件名使用 `time.strftime('%Y-%m-%d %H:%M:%S')`，包含冒号 `:`，可能导致 `logging.FileHandler(path + '.txt')` 创建文件失败。
- `argparse` 里 `type=bool` 用法有坑，例如命令行传 `--noisy-training False` 仍可能被解析为 `True`。

## 2. 数据加载与构造

当前 `run.py` 实际使用：

- `from data_loader import loader`
- 调用位置：`run.py:184-186`

其他 loader：

- `pvp_data_loader.py`: 只处理 PVP 的旧/分场景版本。
- `psp_data_loader.py`: 只处理 PSP 的旧/分场景版本。
- `data_loader.py`: 同时处理 PVP + PSP，是当前入口使用的版本。

### 2.1 `data_loader.load_data()`

函数位置：`data_loader.py:10`。

不同数据集加载结果：

- `Scene15`: `data = mat['X'][0][0:2]`，取前两个视图；`label = squeeze(mat['Y'])`。
- `Caltech101`: `data = mat['X'][0][3:5]`，取两个视图；当前仓库未内置该数据。
- `Reuters_dim10`: 拼接 `x_train[0/1]` 与 `x_test[0/1]` 作为两个视图，并 normalize；标签拼接 `y_train`, `y_test`。
- `NoisyMNIST-30000`: `X1`, `X2`, `Y`；当前仓库未内置。
- `2view-caltech101-8677sample`: `X[0][0].T`, `X[0][1].T`, `gt`；当前仓库未内置。
- `MNIST-USPS`: `X1`, normalized `X2`, `Y`；当前仓库未内置。
- `AWA-7view-10158sample`: `X[0][5].T`, `X[0][6].T`, `gt`；当前仓库未内置。

内置数据：

- `datasets/Scene15.mat`: `X` shape `(1, 3)`, `Y` shape `(4485, 1)`。
- `datasets/Reuters_dim10.mat`: `x_train` / `x_test` shape `(5, 9379, 10)`, `y_train` / `y_test` shape `(1, 9379)`。

返回变量：`data_loader.load_data()` 返回：

- `train_pairs`: `[view0.T, view1.T]`，成对训练样本。
- `train_pair_labels`: 训练用 pair label，默认 noisy labels。
- `train_pair_real_labels`: 用真实类别检查负样本是否 false negative。
- `all_data`: `[all_view0.T, all_view1.T]`，评估/推理用全量或构造后的数据。
- `all_label`: 评估标签，通常对应 view0 / anchor 顺序。
- `all_label_X`: view0 的类别标签。
- `all_label_Y`: view1 的类别标签；PVP 时会随 `test_Y` shuffle。
- `divide_seed`: 划分 seed。
- `mask`: shape `[N, 2]` 的可见矩阵。

Dataset 输出：

- `getDataset.__getitem__()` 返回 `(fea0, fea1, label, real_label)`。
- `getAllDataset.__getitem__()` 返回 `(fea0, fea1, label, class_labels0, class_labels1, mask)`。

### 2.2 是否存在关键变量

- 多视图数据列表：存在，变量名主要是 `data`, `all_data`, `train_pairs`，但只支持两视图。
- 标签：存在，变量名 `label`, `all_label`, `all_label_X`, `all_label_Y`。
- 缺失矩阵 / mask：存在，变量名 `mask`，由 `get_sn()` 构造。
- 不对齐矩阵 / permutation / shuffle index：临时存在 `shuffle_idx`，但没有返回或保存。
- 已配对样本索引：没有显式返回。训练正样本默认来自 `train_X[i]` 和 `train_Y[i]`。
- 每个视图有效样本索引：没有显式返回，只返回二值 `mask`。
- 全局样本 ID：没有保留。没有 `sample_id`, `global_id`, `orig_idx` 等字段。

判断：对后续严谨评估不够安全。当前评估时 `all_loader` 使用 `shuffle=True`，虽然表示与标签在同一迭代顺序中一起 append，KMeans 指标本身可以对上，但无法把预测结果稳定映射回原始样本 ID，也无法做跨 batch 的全局重配对/补全审计。

## 3. PVP / PSP 构造

### 3.1 PVP 构造

位置：

- `data_loader.py:58-73`
- `pvp_data_loader.py:54-71`

流程：

1. `TT_split(len(label), 1 - aligned_prop, divide_seed)` 划分 aligned 训练部分和 unaligned 测试部分。
2. 如果 `aligned_prop == 1.0`，两视图保持原始对应。
3. 如果 `aligned_prop < 1.0`：
   - `shuffle_idx = random.sample(range(len(test_Y)), len(test_Y))`
   - `test_Y = test_Y[shuffle_idx]`
   - `test_label_Y = test_label[shuffle_idx]`
   - `all_data = concat(train_X, test_X), concat(train_Y, shuffled_test_Y)`

实现方式：通过对 view1 的未对齐部分 `test_Y` 做 shuffle 来模拟 PVP。不保存 `shuffle_idx`，也没有 permutation matrix。

训练数据：

- PVP 情况下，训练 pair 只来自 aligned 的 `train_X`, `train_Y`。
- `valid_idx = np.ones_like(train_label)`，即不使用 PSP mask 过滤 aligned 训练部分。

### 3.2 PSP 构造

位置：

- `data_loader.py:77-82`
- `psp_data_loader.py:58-63`
- `get_sn()` 在 `data_loader.py:146` / `psp_data_loader.py:119`

流程：

1. 调用 `get_sn(2, len(test_label), 1 - complete_prop)` 生成二视图可见矩阵。
2. `get_sn()` 保证每个样本至少保留一个视图。
3. 在 `aligned_prop < 1.0` 时，aligned 训练部分拼接全 1 mask，只有 test/unaligned 部分使用缺失 mask。
4. 在 `aligned_prop == 1.0` 时，整个数据使用 PSP mask。

缺失样本处理方式：

- 原始 `all_data` 中没有删除缺失视图，也没有置零；仍保留完整特征。
- 缺失只通过 `mask` 在推理阶段的 latent imputation 中体现。
- 训练阶段如果 `aligned_prop == 1.0`，用 `valid_idx = logical_and(mask[:,0], mask[:,1])` 只选两视图都可见的样本构造 pair。
- 训练阶段如果 `aligned_prop < 1.0`，只用 aligned train 部分，且 aligned train 部分 mask 被强制全 1。

判断：PSP 构造可以作为参考，但它不是严格的数据层缺失模拟。缺失视图特征仍在内存中，只是在推理补全逻辑里通过 mask 判定。这对新模型底座不够干净，建议重写 Dataset，使缺失视图不可访问或显式置为占位，并始终返回 `global_id` 与 `valid_indices_per_view`。

### 3.3 训练 loss 是否只在可见样本上计算

训练 loss 在 `run.py:82-127`。

- 训练 batch 来自 `train_pair_loader`。
- pair 构造时已过滤到两视图可见样本或 aligned complete 部分。
- 所以当前训练 loss 不直接在缺失样本上计算。

但注意：这是通过 loader 预筛 pair 实现，不是 loss 内部用 `mask` 控制。若后续改成 mini-batch 全量样本训练，需要重新实现 mask-aware loss。

## 4. 模型结构

模型文件：`models.py`。

模型类：

- `SUREfcScene`
- `SUREfcReuters`
- `SUREfcCaltech`
- `SUREfcNoisyMNIST`
- `SUREfcMNISTUSPS`
- `SUREfcDeepCaltech`
- `SUREfcDeepAnimal`

主流程：

```text
X0 -> encoder0 -> h0
X1 -> encoder1 -> h1
[h0, h1] concat -> union
union -> decoder0 -> z0
union -> decoder1 -> z1

training:
h0, h1 -> pairwise_distance -> NoiseRobustLoss
x0, z0 and x1, z1 -> MSE reconstruction loss

inference:
h0, h1 -> latent imputation / nearest-neighbor alignment -> concat -> KMeans
```

没有发现显式 projection head、clustering head、posterior head 或 pseudo-label head。latent 维度大多是每视图 10，concat 后 decoder 输入为 20。

SURE/MvCLN 特定模块：

- 成对构造正负样本：`get_pairs()`。
- noise-robust contrastive loss：`run.py:49-66`。
- latent 层最近邻重配对：`sure_inference.both_infer()`。
- latent 层 kNN/平均补全：`sure_inference.both_infer()` / `pdp_infer()`。

可保留为 clean backbone 的部分：

- `encoder0`, `encoder1`, `decoder0`, `decoder1` 可以抽出来作为两视图 AE backbone。
- 但现有 decoder 输入依赖 `[h0,h1] concat`，不是单视图自编码器；若要支持任意缺失视图，建议拆成 view-specific 或 fusion-aware decoder。

代码风险：

- `SUREfcDeepCaltech.forward()` 和 `SUREfcDeepAnimal.forward()` 使用 `self.encoder0` 编码两个视图、`self.decoder0` 解码两个视图，疑似 bug；`encoder1` / `decoder1` 未使用。

## 5. 损失函数

### 5.1 Reconstruction loss

定义/调用：

- `criterion_mse = nn.MSELoss()`，`run.py:204`
- `ver_loss = MSE(x0, z0) + MSE(x1, z1)`，`run.py:109`
- 总 loss：`loss = ncl_loss + args.lam * ver_loss`

依赖变量：

- 输入 `x0`, `x1`
- decoder 输出 `z0`, `z1`

是否依赖标签：不依赖类别标签。

是否依赖已配对样本：依赖。decoder 输入是 `cat(h0,h1)`，如果 pair 不真实配对，重构会被污染。

可否关闭：可以把 `lam=0` 关闭 reconstruction loss，训练仍可运行，因为 `z0/z1` 只用于该 loss。但 decoder 将不被有效训练。

### 5.2 Alignment / Contrastive loss

定义：

- `NoiseRobustLoss`，`run.py:49-66`

调用：

- `pair_dist = F.pairwise_distance(h0, h1)`，`run.py:96`
- `ncl_loss = criterion[0](pair_dist, labels, args.margin, args.robust, args)`，`run.py:108`

依赖变量：

- `h0`, `h1`
- pair label `labels`
- `args.margin`, `args.robust`, `args.start_fine`

是否依赖标签：

- 训练用 `labels` 是 pair correspondence label，不是原始类别标签。
- `real_labels` 只用于统计 true/false negative 距离。
- 如果 `--noisy-training False`，`train_pair_labels = real_labels`，此时使用类别标签判断负样本真伪，存在训练期类别信息泄漏。

是否依赖已配对样本：强依赖。正样本 pair 默认必须是一一配对样本。

可否关闭：不建议直接关闭。若 `ncl_loss` 关闭，encoder 缺少主要跨视图对齐目标，且 `pos_dist/neg_dist` 日志逻辑仍依赖 pair labels。若新模型不用 NCL，建议重写 `train()`。

### 5.3 Imputation loss

没有显式训练期 imputation loss。补全发生在 inference：

- `sure_inference.both_infer()` 中 `setting != 0` 分支。
- 使用 latent 最近邻平均填补缺失视图表示。

可否关闭：

- 对 `settings=0` 自动关闭 PSP imputation。
- 对新模型建议不要复用该补全作为训练模块，因为它是 batch 内启发式，不是可学习损失。

### 5.4 Clustering / pseudo-label loss

没有训练期 clustering loss 或 pseudo-label update。

- KMeans 只在评估阶段调用：`run.py:239`, `Clustering.py:13-25`。
- 没有 clustering head。
- 没有 DEC/target distribution 一类 pseudo-label loss。

### 5.5 Regularization loss

没有显式 regularization loss。只有模型中的 `Dropout(0.2)` 和 optimizer 默认行为。

## 6. 评估流程

主调用：

- `run.py:237`: `v0, v1, gt_label = both_infer(model, device, all_loader, args.settings)`
- `run.py:239`: `y_pred, ret = Clustering(data, gt_label)`

推理文件：

- `sure_inference.py`
- `both_infer()` 是当前主入口实际使用的推理函数。
- `pvp_infer()` / `pdp_infer()` 是分场景版本。

KMeans：

- 使用 `sklearn.cluster.KMeans`，位置 `Clustering.py:7`, `Clustering.py:20`。
- 输入表示为 `np.concatenate([v0, v1], axis=1)`，即两个 latent 表示拼接。
- 聚类数 `n_clusters = len(unique(y))`，使用真实标签数量，仅用于评估设定聚类数。

ACC/NMI/ARI：

- `Clustering.py:102-119`。
- ACC 使用 Hungarian / Munkres 匹配。
- NMI/ARI 使用 sklearn metrics。

真实标签使用：

- 训练默认 `--noisy-training True` 时，类别标签不直接作为监督 loss。
- 评估中真实标签用于确定类别数、Hungarian matching、ACC/NMI/ARI。
- 但 `real_labels` 的构造使用了类别标签来统计 false negatives；如果关闭 noisy training，则类别标签会进入训练 pair label。

不对齐场景下 ID 对应问题：

- PVP 时 `all_label` 仍是 view0 / anchor 顺序，`all_label_Y` 是 shuffled view1 标签。
- `both_infer()` 在 batch 内对 `recover_out0[i]` 找最近 `recover_out1[idx[0]]`，输出拼接后再聚类。
- 没有 global sample id，`all_loader` 又 `shuffle=True`，因此无法稳定追踪每个预测对应的原始样本 ID。
- 重配对仅在当前 batch 内完成，不能跨 batch 搜索全局最近邻；batch size 会影响对齐结果。

判断：当前评估代码可作为快速论文复现实验，但不适合作为新模型严谨评估底座。若研究“不完整 + 不对齐”，必须显式保留并返回 `global_id`, `view_id/original_idx`, `permutation`, `mask`, `valid_indices_per_view`，并在评估阶段按 global id 对齐预测与 `labels`。

## 7. 最小可复现实验建议

内置小数据集推荐：

- 首选 `Scene15`，`--data 0`，样本 4485，两个输入视图维度 20 和 59。
- `Reuters_dim10` 样本更多，适合作第二步。

建议 sanity check 设置：

```bash
python run.py --data 0 --gpu 0 --settings 2 --aligned-prop 0.8 --complete-prop 0.8 --epochs 2 --batch-size 256 --neg-prop 5
```

如果机器没有 CUDA：

```bash
python run.py --data 0 --gpu 0 --settings 2 --aligned-prop 0.8 --complete-prop 0.8 --epochs 2 --batch-size 256 --neg-prop 5
```

代码会自动使用 `device = 'cuda' if torch.cuda.is_available() else 'cpu'`，所以同一命令也可在 CPU 上跑，只是 `--gpu` 无实际作用。

依赖：

- PyTorch
- NumPy
- SciPy
- scikit-learn
- munkres
- matplotlib

数据路径：

- `./datasets/Scene15.mat`
- `./datasets/Reuters_dim10.mat`

运行前注意：

- Windows 可能需要先处理日志文件名冒号问题，否则 `log/...time=YYYY-MM-DD HH:MM:SS.txt` 可能创建失败。
- 旧版代码使用 `np.int`，在较新 NumPy 版本可能报错；若 sanity check 报 `module 'numpy' has no attribute 'int'`，这是依赖兼容问题，不是模型逻辑问题。

## 8. 是否适合作为新模型底座

### 可以复用

- 数据集读取分支：`data_loader.load_data()` 中各 `.mat` 格式处理。
- PVP 构造参考：`shuffle_idx` 打乱 view1 的 unaligned subset。
- PSP mask 生成参考：`get_sn()`。
- 基础成对训练 Dataset 写法：`getDataset`。
- 基础全量评估 Dataset 写法：`getAllDataset`，但需要加 ID。
- encoder-decoder 结构：`models.py` 中各 `SUREfc*`。
- KMeans + Hungarian ACC/NMI/ARI：`Clustering.py`。
- seed 和日志框架思路：`run.py`。

### 不适合直接复用

- `train()`：强绑定 pairwise contrastive training，且日志统计依赖正负 pair。
- `NoiseRobustLoss`：SURE/MvCLN 特定，依赖 pair labels、margin、robust switching。
- inference imputation：batch 内最近邻平均，不能保证全局一致。
- inference alignment：batch 内 nearest-neighbor matching，不能跨 batch，且没有返回对齐索引。
- `data_loader.loader()`：只支持两视图，没有 global sample id，没有保存 permutation。
- 评估流程：不能把预测结果稳定对应回原始样本 ID。
- `all_loader shuffle=True`：对严谨 ID 追踪不友好。
- boolean CLI 参数：`type=bool` 不可靠。

### 总体建议

建议选择“半复用”：复用 `data/mask/eval/encoder-decoder` 的部分代码和思路，新写 model、Dataset 返回协议、train loop、loss 与评估对齐逻辑。

推荐的新底座接口：

- Dataset 返回：
  - `views`: list/tensor dict of view features
  - `mask`: `[N, V]`
  - `label`: `[N]`
  - `global_id`: `[N]`
  - `view_sample_ids` 或 `permutation`: 每个视图当前行对应的原始样本 ID
  - `paired_indices`: 已知配对样本索引
  - `valid_indices_per_view`: 每个视图有效样本索引
- 训练 loss 显式使用 `mask` 和 `paired_indices`。
- PVP/PSP 构造时保存 `shuffle_idx` / permutation matrix。
- 评估阶段按 `global_id` 还原预测与真实标签，再计算 ACC/NMI/ARI。

最终判断：不适合直接改成新模型；适合半复用，即复用数据读取、PVP/PSP 构造参考、metric 和部分 encoder-decoder，新写更干净的多视图不完整不对齐实验框架。
