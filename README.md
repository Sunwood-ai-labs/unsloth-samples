# Unsloth Samples

Unslothを使ってLLMのファインチューニングをサクッと試すためのサンプル集です。

## これは何？

[Unsloth](https://unsloth.ai/)は、LLMのファインチューニングを高速化＆省メモリ化してくれる便利なライブラリです。このリポジトリでは、実際に動かせるサンプルコードを提供しています。

### Unslothの特徴

- **2倍高速** with **70%少ないVRAM**
- 精度を落とさずに学習可能
- LoRA、フルファインチューニング、事前学習に対応
- Llama 4, Phi-4, Qwen3, DeepSeek-R1など最新モデルに対応

### 🆕 対応環境

このサンプルは **Tesla T4 (15GB VRAM)** で動作確認済みです。
詳しいモデル情報は [GPU_MODELS_GUIDE.md](./GPU_MODELS_GUIDE.md) を参照してください。

## セットアップ

### 必要なもの

- Python 3.10以上
- CUDA対応のGPU（推奨）
- [uv](https://github.com/astral-sh/uv)（Pythonパッケージマネージャー）

### インストール

#### 方法1: 自動セットアップ（推奨）

```bash
chmod +x setup.sh
./setup.sh
```

#### 方法2: 手動セットアップ

```bash
# uvのインストール（まだの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストール
uv sync
```

## 使い方

### サンプルスクリプト

#### 1️⃣ 基本サンプル（軽量）
```bash
uv run python finetune_example.py
```
TinyLlamaを使った基本的なファインチューニング

#### 2️⃣ Llama 4サンプル（最新・推奨）
```bash
uv run python finetune_llama4.py
```
2026年最新のLlama 4を使った高性能ファインチューニング

### 各サンプルの内容

**finetune_example.py** - 基本編
1. 軽量モデル（TinyLlama）のロード
2. LoRAを使った効率的なファインチューニング
3. Alpacaデータセット（最初の1000件）で学習
4. モデルの保存
5. 簡単な推論テスト

**finetune_llama4.py** - 最新モデル編
1. Llama 4 (8B) のロード
2. Tesla T4用に最適化された設定
3. 日本語/英語データセットで学習
4. 詳細なメモリ使用状況表示
5. 複数の推論テスト

## カスタマイズのポイント

### モデルの変更

```python
model_name = "unsloth/tinyllama"  # ← ここを変更
```

使えるモデルの例（2026年最新）：
- `unsloth/Llama-4-8B` - Meta最新（推奨）
- `unsloth/Phi-4` - Microsoft軽量モデル
- `unsloth/Qwen3-7B` - 多言語対応
- `unsloth/DeepSeek-R1-7B` - 推論特化
- `unsloth/Gemma-3n-7B` - Google最新

📖 詳しくは [GPU_MODELS_GUIDE.md](./GPU_MODELS_GUIDE.md) を参照

### LoRAのパラメータ調整

```python
r = 16  # LoRAのランク（大きいほど表現力が高いが重くなる）
lora_alpha = 16  # スケーリングファクター
```

### トレーニングステップ数

```python
max_steps = 100  # お試しなら少なめ、本格的には1000以上
```

## ファイル構成

```
.
├── README.md              # このファイル
├── GPU_MODELS_GUIDE.md    # Tesla T4で動くモデル一覧
├── pyproject.toml         # プロジェクト設定（uv用）
├── requirements.txt       # 依存関係リスト
├── setup.sh              # 自動セットアップスクリプト
├── finetune_example.py   # 基本サンプル（TinyLlama）
├── finetune_llama4.py    # 最新モデルサンプル（Llama 4）
└── .gitignore            # Git除外設定
```

## トラブルシューティング

### CUDAが見つからない

CPUでも動きますが遅いです。Google ColabやKaggleなどのGPU環境での実行を推奨します。

### メモリ不足エラー

```python
load_in_4bit = True  # 4bit量子化を有効に
per_device_train_batch_size = 1  # バッチサイズを小さく
```

## 参考リンク

- [Unsloth公式ドキュメント](https://unsloth.ai/docs)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [uv公式サイト](https://github.com/astral-sh/uv)

## ライセンス

このサンプルコードはMITライセンスです。ご自由にお使いください。
