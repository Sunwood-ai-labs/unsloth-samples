# Unsloth LLM ファインチューニング環境構築ワークフロー

このドキュメントは、Claude Codeエージェントを使ってUnslothベースのLLMファインチューニング環境を構築した際の完全なワークフローを記録したものです。

## 🎯 目的

- Tesla T4 (15GB VRAM) でLLMファインチューニング環境を構築
- 複数サイズのモデル（軽量〜中規模）で動作確認
- 実用的なサンプルコードとドキュメントを整備
- GitHubリポジトリとして公開

## 📋 ワークフロー概要

### Phase 1: 要件定義と調査 (10分)

#### 1.1 技術スタック調査
```bash
# Unsloth公式ドキュメントを確認
- URL: https://unsloth.ai/docs
- 特徴: 2x faster, 70% less VRAM
- 対応モデル: Llama, Qwen, Phi, Gemma, DeepSeek等
```

**出力:**
- Unslothの特徴理解
- インストール方法
- サポートモデル一覧

#### 1.2 GPU環境確認
```bash
nvidia-smi
```

**確認事項:**
- GPU名: Tesla T4
- VRAM: 14.74 GB
- CUDA: 12.6
- 使用可能なモデルサイズを判断

### Phase 2: プロジェクト初期化 (5分)

#### 2.1 GitHubリポジトリ作成
```bash
gh repo create unsloth-samples --public \
  --description "Unsloth LLM Fine-tuning Samples" --clone
```

**ポイント:**
- 親しみやすいリポジトリ名（-samplesサフィックス）
- mainブランチに設定
- 公開リポジトリ

#### 2.2 基本ファイル構成
```
unsloth-samples/
├── README.md              # プロジェクト概要
├── requirements.txt       # 依存関係
├── pyproject.toml        # プロジェクト設定
├── .gitignore            # Git除外設定
└── setup.sh              # セットアップスクリプト
```

### Phase 3: 環境構築とトラブルシューティング (15分)

#### 3.1 依存関係管理

**初期実装（失敗）:**
```toml
# pyproject.toml - build-system使用
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**問題:**
- `uv sync` でビルドエラー
- 直接参照（git+）に対応していない
- サンプルプロジェクトには過剰

**解決策:**
```toml
# pyproject.toml - シンプル化
[project]
name = "unsloth-samples"
requires-python = ">=3.10"

# サンプルプロジェクトなので、requirements.txtを使用してください
```

```txt
# requirements.txt
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
torch>=2.1.0
transformers>=4.36.0
...
```

```bash
# setup.sh
uv pip install -r requirements.txt --system
```

#### 3.2 wandbエラー対応

**問題:**
```
wandb.errors.UsageError: No API key configured
```

**解決策:**
```python
# TrainingArgumentsに追加
args=TrainingArguments(
    ...
    report_to="none",  # wandbを無効化
)
```

**適用箇所:**
- すべてのサンプルスクリプト
- テストスクリプト

### Phase 4: GPU別モデル選定 (10分)

#### 4.1 モデルサイズとVRAM使用量の調査

**調査結果（4bit量子化 + LoRA）:**

| モデル | パラメータ数 | VRAM使用量 | Tesla T4適合 |
|--------|--------------|------------|--------------|
| TinyLlama | 1.1B | ~1GB | ✅ 余裕 |
| Phi-3 | 3.8B | ~2.5GB | ✅ 余裕 |
| Qwen 2.5 7B | 7.66B | ~6.4GB | ✅ 適合 |
| Llama 3.2 11B | 11B | ~12GB | ⚠️ ギリギリ |
| Llama 4 8B | 8B | ~9GB | ✅ 適合 |

#### 4.2 推奨モデルリスト作成

`GPU_MODELS_GUIDE.md` に記載:
- 用途別おすすめモデル
- VRAM使用量の目安
- トラブルシューティング

### Phase 5: サンプルコード実装 (30分)

#### 5.1 環境テストスクリプト

```python
# test_setup.py
# 目的: 環境が正しくセットアップされているか確認

1. パッケージインポートテスト
2. CUDA環境確認
3. モデルロードテスト（TinyLlama）
4. 推論テスト
```

**実行時間:** 約30秒

#### 5.2 軽量モデルテスト（TinyLlama）

```python
# finetune_quick_test.py
# 目的: 最小構成でファインチューニングが動作するか確認

- モデル: TinyLlama (4bit)
- データセット: Alpaca 100件
- ステップ数: 10
- バッチサイズ: 2
```

**実行時間:** 約58秒
**VRAM使用:** 最大 1.00 GB

**結果:**
- ✅ トレーニング成功
- 損失: 1.910 → 1.859
- ⚠️ 推論品質は限定的

#### 5.3 中規模モデルテスト（Qwen 2.5 7B）

```python
# finetune_7b_test.py
# 目的: 実用的なモデルサイズで性能確認

- モデル: Qwen 2.5 7B (4bit)
- データセット: Alpaca 500件
- ステップ数: 20
- バッチサイズ: 2
```

**実行時間:** 約138秒
**VRAM使用:** 最大 6.40 GB (43.4%)

**結果:**
- ✅ トレーニング成功
- 損失: 1.218 → 1.014
- ✅ 推論品質が大幅向上
  - 日本の首都 → 「東京です」（正確）
  - Hello World → `print("Hello World")`（正確）
  - 機械学習の説明 → 詳細な回答

#### 5.4 本番サンプル

```python
# finetune_example.py (100ステップ)
# finetune_llama4.py (200ステップ)
# 目的: 実用的なファインチューニング例
```

### Phase 6: .gitignore整備 (5分)

#### 6.1 ML/ファインチューニング特有の除外パターン

```gitignore
# Model files
*.bin
*.safetensors
models/
checkpoints/
outputs*/
finetuned_*/

# Unsloth & HuggingFace cache
huggingface_tokenizers_cache/
unsloth_compiled_cache/
.cache/
wandb/

# Dataset cache
*.arrow
*.cache

# UV
uv.lock
```

### Phase 7: ドキュメント整備 (20分)

#### 7.1 README.md構成

```markdown
# 構成
1. プロジェクト概要
2. 特徴（Unslothの利点）
3. 対応環境（Tesla T4動作確認済み）
4. セットアップ手順
5. 使い方（サンプル別）
6. カスタマイズポイント
7. ファイル構成
8. トラブルシューティング
9. 動作確認結果（重要！）
   - テスト環境
   - 軽量モデル結果
   - 中規模モデル結果
   - 推論品質比較表
10. 参考リンク
```

**ポイント:**
- 実測値を記載（VRAM、時間、損失）
- 比較表で違いを明確化
- 動作確認済みバッジ

#### 7.2 GPU_MODELS_GUIDE.md

```markdown
# 内容
- Tesla T4で動くモデル一覧
- 最新モデル（2026年）の情報
- VRAM別推奨設定
- メモリ使用量の目安表
- 用途別おすすめモデル
- トラブルシューティング
```

### Phase 8: Git管理とコミット戦略 (継続)

#### 8.1 コミットメッセージ規則

```
<Type>: <Short description>

<Detailed description>

<Results/Metrics if applicable>

🦥 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

**Type:**
- `Initial commit` - 初期コミット
- `Fix` - バグ修正
- `Add` - 機能追加
- `Improve` - 改善

#### 8.2 コミット履歴

```
ae30784 - Add 7B model finetuning test with excellent results
b18d629 - Add working finetuning tests and fix wandb issue
6aad87f - Improve .gitignore for ML/finetuning workflows
353de58 - Fix: Update build configuration for samples project
c499ae6 - Initial commit: Unsloth LLM finetuning samples
```

**ポイント:**
- 論理的な単位でコミット
- 動作確認後にコミット
- 結果（メトリクス）を含める

## 🎓 学んだベストプラクティス

### 1. 環境構築

✅ **DO:**
- シンプルな依存関係管理（requirements.txt）
- 実際にテストしてから公開
- GPU情報を先に確認

❌ **DON'T:**
- 過剰なビルドシステム
- 未検証のサンプルコード
- wandbなど外部サービスへの依存

### 2. モデル選定

✅ **DO:**
- 軽量モデルで動作確認
- 段階的にサイズアップ
- VRAM使用率を記録

❌ **DON'T:**
- いきなり大きなモデル
- GPU限界までVRAM使用
- 実測なしの推奨

### 3. ドキュメント

✅ **DO:**
- 実測値を記載
- 比較表を作成
- トラブルシューティング記載

❌ **DON'T:**
- 理論値のみ記載
- エラー対処法なし
- 環境情報なし

### 4. コード品質

✅ **DO:**
- エラーハンドリング
- プログレス表示
- メモリ使用量表示

❌ **DON'T:**
- サイレントエラー
- ユーザーフィードバックなし
- リソース管理なし

## 📊 最終成果物

### リポジトリ統計

- **ファイル数:** 11
- **総コード行数:** ~1,200行
- **ドキュメント:** 3ファイル
- **サンプルスクリプト:** 5種類
- **動作確認済みモデル:** 2種類

### パフォーマンス実績

| 項目 | TinyLlama | Qwen 2.5 7B |
|------|-----------|-------------|
| パラメータ数 | 1.1B | 7.66B |
| VRAM使用 | 1.00 GB | 6.40 GB |
| トレーニング時間（10-20ステップ） | 58秒 | 138秒 |
| 推論品質 | ⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🔄 再現手順（Claude Codeエージェント向け）

### 1. 初期セットアップ
```bash
# リポジトリ作成
gh repo create <project-name> --public --clone

# ブランチをmainに設定
git branch -m master main
gh auth setup-git
gh repo edit --default-branch main
```

### 2. GPU環境確認
```bash
nvidia-smi
# VRAM容量を確認し、適切なモデルサイズを決定
```

### 3. 基本ファイル作成
```bash
# requirements.txt, pyproject.toml, .gitignore, setup.sh
# シンプルな構成を維持
```

### 4. テストスクリプト作成
```python
# 1. test_setup.py（環境確認）
# 2. finetune_quick_test.py（軽量モデル）
# 3. finetune_7b_test.py（中規模モデル）
```

### 5. 実行と結果記録
```bash
python test_setup.py
python finetune_quick_test.py
python finetune_7b_test.py

# 実測値をREADMEに記載
```

### 6. ドキュメント整備
```markdown
# README.md - 動作確認結果を含む
# GPU_MODELS_GUIDE.md - モデル選定ガイド
# DEVELOPMENT_WORKFLOW.md - このファイル
```

### 7. コミット＆プッシュ
```bash
git add -A
git commit -m "適切なコミットメッセージ"
git push
```

## 🎯 成功の鍵

1. **段階的アプローチ**
   - 軽量モデルから開始
   - 動作確認後にスケールアップ
   - 各段階で実測値を記録

2. **実用性重視**
   - 理論だけでなく実際に動くコード
   - トラブルシューティング情報
   - 比較データの提供

3. **ドキュメントファースト**
   - コードと同時にドキュメント更新
   - 実測値を必ず記載
   - ユーザー視点で記述

4. **継続的な改善**
   - エラー発生時は即座に修正
   - ベストプラクティスを反映
   - Git履歴で変更を追跡

## 📚 参考リンク

- [Unsloth公式ドキュメント](https://unsloth.ai/docs)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [このプロジェクト](https://github.com/Sunwood-ai-labs/unsloth-samples)

---

**作成日**: 2026-01-05
**環境**: Tesla T4 (15GB VRAM), CUDA 12.6
**Unsloth**: 2026.1.1
**Claude Code**: Sonnet 4.5
