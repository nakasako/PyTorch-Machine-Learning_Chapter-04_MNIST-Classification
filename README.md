# Chapter 04: MNIST Classification

この章では、MNIST の手書き数字分類を題材に、PyTorch の実践的な学習プロジェクト構成を扱います。

前章の「2次関数を近似する小さな回帰」から一歩進めて、今回は DataLoader、バッチ学習、検証、checkpoint、TensorBoard、途中再開、モデル切り替えを備えた構成にしています。

## セットアップ

```powershell
uv sync
```

## MNIST の画像を見る

```powershell
uv run python show_mnist.py
```

表示枚数や保存先を変えたい場合は、[show_mnist.py](show_mnist.py) の先頭にある `COUNT` と `SAVE_PATH` を変更します。

## 学習

```powershell
uv run python train.py
```

設定は [configs/train.yaml](configs/train.yaml) にまとめています。

学習の進捗は `tqdm` で表示され、ログは TensorBoard に保存されます。

```powershell
uv run tensorboard --logdir runs
```

TensorBoard では `loss` と `accuracy` のグラフに train/val が重ねて表示されます。

Ctrl+C で中断した場合も、最後の checkpoint が `checkpoints/mnist_classifier.pt` に保存されます。`training.resume: true` の場合、次回の学習は途中から再開されます。

## 推論

```powershell
uv run python infer.py
```

テストデータの index や外部画像を変えたい場合は、[infer.py](infer.py) の先頭にある `TEST_INDEX` または `IMAGE_PATH` を変更します。

## 学習設定

最適化関数は AdamW 固定です。学習率 scheduler は warmup 付き cosine decay 固定で、最後の学習率が 0 にならないよう `min_lr_ratio` を使います。

```yaml
training:
  epochs: 10
  learning_rate: 0.001
  weight_decay: 0.0001
  warmup_epochs: 1
  min_lr_ratio: 0.05
  max_grad_norm: 1.0
  mixed_precision: true
```

`mixed_precision: true` の場合、CUDA 環境では AMP を使います。CPU では自動的に通常精度で動きます。

## モデルの切り替え

[configs/train.yaml](configs/train.yaml) の `model.name` を変更します。

```yaml
model:
  name: mlp
```

指定できる値:

- `mlp`: 全結合ネットワーク
- `cnn`: 畳み込みニューラルネットワーク

MLP は `LayerNorm`、CNN は `GroupNorm` を使います。小さい batch size でも挙動が安定しやすく、教材としても扱いやすい構成です。

## 構成

```text
.
├── configs/
│   └── train.yaml
├── data/
│   └── mnist.py
├── engine/
│   └── trainer.py
├── models/
│   └── classifiers.py
├── utils/
│   ├── checkpoint.py
│   ├── config.py
│   └── seed.py
├── train.py
├── infer.py
└── show_mnist.py
```

主な役割:

- `data/mnist.py`: MNIST Dataset と DataLoader の作成
- `models/classifiers.py`: MLP、CNN の定義と切り替え
- `engine/trainer.py`: train/eval ループ、TensorBoard 記録、checkpoint 保存
- `utils/checkpoint.py`: checkpoint の保存と復元
- `train.py`: 学習入口
- `infer.py`: 推論入口
- `show_mnist.py`: 教材導入用の画像表示
