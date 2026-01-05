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
├── .claude/agents/
│   └── workflow-automation-agent.md  # 🤖 環境構築エージェント
│
├── README.md                         # このファイル
├── DEVELOPMENT_WORKFLOW.md           # 📖 完全開発ワークフロー
├── GPU_MODELS_GUIDE.md               # 🎯 Tesla T4で動くモデル一覧
│
├── pyproject.toml                    # プロジェクト設定（uv用）
├── requirements.txt                  # 依存関係リスト
├── setup.sh                          # 自動セットアップスクリプト
├── .gitignore                        # Git除外設定
│
├── test_setup.py                     # 環境セットアップテスト
├── finetune_quick_test.py            # 動作確認用（TinyLlama、10ステップ）
├── finetune_7b_test.py               # 中規模モデルテスト（Qwen 2.5 7B、20ステップ）✨
├── finetune_example.py               # 基本サンプル（TinyLlama、100ステップ）
└── finetune_llama4.py                # 最新モデルサンプル（Llama 4、200ステップ）
```

## トラブルシューティング

### CUDAが見つからない

CPUでも動きますが遅いです。Google ColabやKaggleなどのGPU環境での実行を推奨します。

### メモリ不足エラー

```python
load_in_4bit = True  # 4bit量子化を有効に
per_device_train_batch_size = 1  # バッチサイズを小さく
```

## 動作確認結果

### テスト環境
- **GPU**: Tesla T4 (14.74 GB VRAM)
- **CUDA**: 12.6
- **Unsloth**: 2026.1.1
- **PyTorch**: 2.9.0+cu126
- **Transformers**: 4.57.3

### ファインチューニング動作確認

#### 軽量モデル: TinyLlama (10ステップ)

```bash
python finetune_quick_test.py
```

**結果:**
- モデル: TinyLlama (4bit量子化)
- データセット: Alpaca (100件)
- トレーニングパラメータ: 12.6M / 1.1B (1.13%)
- トレーニング時間: 約58秒
- 損失: 1.859 (初期: 1.910)
- GPUメモリ使用: 最大 1.00 GB (6.8%)

#### 中規模モデル: Qwen 2.5 7B (20ステップ) 🔥

```bash
python finetune_7b_test.py
```

**結果:**
- モデル: Qwen 2.5 7B (4bit量子化)
- データセット: Alpaca (500件)
- トレーニングパラメータ: 40.4M / 7.66B (0.53%)
- トレーニング時間: 約138秒
- 損失: 1.014 (初期: 1.218)
- GPUメモリ使用: 最大 6.40 GB (43.4%)

**推論品質の比較:**

| 質問 | TinyLlama | Qwen 2.5 7B |
|------|-----------|-------------|
| 日本の首都は？ | 繰り返しが発生 | ✅ **東京です** |
| Hello World | 不完全な回答 | ✅ **print("Hello World")** |
| 機械学習とは？ | - | ✅ **詳細な説明** |

### セットアップテスト

```bash
python test_setup.py
```

**結果:**
- ✅ 全パッケージのインポート成功
- ✅ CUDA環境認識
- ✅ モデルロード成功
- ✅ 推論テスト成功

## 🤖 自動化エージェント

このリポジトリには、同じ環境を自動構築できる **Claude Code エージェント** が含まれています！

### `llm-finetuning-env-builder`

**機能:**
- 🔍 GPU環境の自動確認とモデル選定
- 📦 GitHubリポジトリの作成と初期化
- ⚙️ 依存関係のセットアップ（トラブルシューティング対応済み）
- 📝 テストコードとサンプルスクリプトの生成
- 📊 動作確認とベンチマーク
- 📚 ドキュメント自動生成（実測値付き）
- 🔄 Git管理の自動化

**使い方:**
```bash
# Claude Codeで実行
/agent llm-finetuning-env-builder
```

**詳細:**
- エージェント定義: `.claude/agents/workflow-automation-agent.md`
- ワークフロー詳細: [DEVELOPMENT_WORKFLOW.md](./DEVELOPMENT_WORKFLOW.md)

このエージェントは、このリポジトリ構築で得た知見を完全に再現できます：
- ✅ 8フェーズの体系的な構築プロセス
- ✅ GPU最適化されたモデル選定
- ✅ 実績ベースのトラブルシューティング
- ✅ 品質チェックリスト付き

**実績:**
- TinyLlama環境: 約10分
- Qwen 2.5 7B環境: 約15分
- 完全な環境構築: 約90分

## 📖 開発者向けドキュメント

### DEVELOPMENT_WORKFLOW.md

このリポジトリを構築した完全なワークフローを記録しています：

- **Phase 1-8**: 要件定義から完成まで
- **トラブルシューティング**: 実際に遭遇した問題と解決策
- **ベストプラクティス**: 学んだ知見の集約
- **再現手順**: コマンドレベルの詳細手順

**主なトピック:**
- wandbエラー対応（`report_to="none"`）
- UV build エラー対応（requirements.txt推奨）
- GPU別モデル選定マトリクス
- 段階的なモデルサイズアップ戦略
- 実測値ベースのドキュメント作成

👉 詳細は [DEVELOPMENT_WORKFLOW.md](./DEVELOPMENT_WORKFLOW.md) を参照

## 参考リンク

- [Unsloth公式ドキュメント](https://unsloth.ai/docs)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [uv公式サイト](https://github.com/astral-sh/uv)
- [このリポジトリの開発ワークフロー](./DEVELOPMENT_WORKFLOW.md)

## ライセンス

このサンプルコードはMITライセンスです。ご自由にお使いください。
