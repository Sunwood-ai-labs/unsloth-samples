# Tesla T4 (15GB) で動かせる最新モデルガイド

このガイドでは、Tesla T4 (15GB VRAM) で実行可能な最新のLLMモデルを紹介します。

## GPU情報

- **モデル**: NVIDIA Tesla T4
- **VRAM**: 15GB (15360MiB)
- **CUDA**: 12.4

## 推奨モデル（2026年最新）

### 🔥 最新＆推奨モデル

#### 1. **Llama 4** by Meta (NEW!)
```python
model_name = "unsloth/Llama-4-8B"
```
- Metaの最新モデル（Scout & Maverick対応）
- 8Bパラメータなら15GBで余裕
- 最新のアーキテクチャと性能

#### 2. **Phi-4** by Microsoft (NEW!)
```python
model_name = "unsloth/Phi-4"
```
- Microsoftの最新軽量モデル
- 効率的な推論性能
- 4bit版とGGUF版が利用可能

#### 3. **Qwen3** by Alibaba (NEW!)
```python
model_name = "unsloth/Qwen3-7B"
```
- Alibabaの最新モデル
- 多言語対応が強力（日本語も◎）
- Qwen3-30B-A3Bは17.5GBなので、小さいバージョン推奨

#### 4. **DeepSeek-R1** (NEW!)
```python
model_name = "unsloth/DeepSeek-R1-7B"
```
- 推論タスクに特化した最新モデル
- 小さいバージョン（7B）なら15GBで動作可能

#### 5. **Gemma 3n** by Google (NEW!)
```python
model_name = "unsloth/Gemma-3n-7B"
```
- Googleの最新Gemmaシリーズ
- GGUF版と4bit版が利用可能

### 🎨 Vision・マルチモーダルモデル

#### Llama 3.2 Vision (11B)
```python
model_name = "unsloth/Llama-3.2-11B-Vision"
```
- 画像とテキストの両方を扱えるマルチモーダルモデル
- 11Bなので4bit量子化で動作可能

#### Qwen 2.5 VL (7B)
```python
model_name = "unsloth/Qwen-2.5-VL-7B"
```
- Qwenのビジョンモデル
- 日本語＋画像認識の組み合わせに強い

### 🎙️ 音声モデル

#### Text-to-Speech (TTS)
```python
model_name = "unsloth/csm-1b"
```
- テキスト音声合成モデル
- 1Bと軽量なので余裕で動作

## VRAM別推奨設定

### 15GB Tesla T4での推奨設定

```python
from unsloth import FastLanguageModel

# 7B-8Bモデルの場合（余裕あり）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-4-8B",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,  # 4bit量子化で約4GB
)

# トレーニング設定
per_device_train_batch_size = 4  # バッチサイズ余裕あり
gradient_accumulation_steps = 4
```

### 11B-14Bモデルの場合（ギリギリ）
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-11B-Vision",
    max_seq_length=1024,  # シーケンス長を短めに
    dtype=None,
    load_in_4bit=True,
)

# トレーニング設定
per_device_train_batch_size = 1  # バッチサイズ小さめ
gradient_accumulation_steps = 8  # 代わりに勾配累積を増やす
```

## メモリ使用量の目安

| モデルサイズ | フルFP16 | 4bit量子化 | 4bit+LoRA学習 |
|------------|---------|-----------|--------------|
| 7B         | ~14GB   | ~4GB      | ~8GB         |
| 8B         | ~16GB   | ~4.5GB    | ~9GB         |
| 11B        | ~22GB   | ~6GB      | ~12GB        |
| 13B        | ~26GB   | ~7GB      | ~14GB        |
| 30B        | ~60GB   | ~15GB     | ~30GB        |

※LoRA学習時は追加のVRAMが必要

## 🎯 用途別おすすめモデル

### チャットボット・対話
- **Llama 4 8B**: 最新の対話性能
- **Qwen3 7B**: 多言語対応が優秀

### 日本語特化
- **Qwen3 7B**: 日本語の理解・生成が強い
- **Llama 4 8B**: ファインチューニングで日本語対応可能

### 推論・数学
- **DeepSeek-R1 7B**: 推論タスクに特化

### 画像認識＋対話
- **Llama 3.2 Vision 11B**: マルチモーダルの最新
- **Qwen 2.5 VL 7B**: 日本語＋画像

### 軽量・高速
- **Phi-4**: 効率重視
- **Gemma 3n 7B**: バランス型

## トラブルシューティング

### OOM (Out of Memory) エラーが出る場合

1. **4bit量子化を有効化**
```python
load_in_4bit = True
```

2. **シーケンス長を短く**
```python
max_seq_length = 1024  # 2048 → 1024
```

3. **バッチサイズを小さく**
```python
per_device_train_batch_size = 1
gradient_accumulation_steps = 16  # 増やして補う
```

4. **勾配チェックポイントを有効化**
```python
use_gradient_checkpointing = "unsloth"
```

## 参考情報

すべてのUnsloth対応モデルは以下から確認できます：
- [Unsloth Model Catalog](https://docs.unsloth.ai/get-started/unsloth-model-catalog)
- [All Models Guide](https://docs.unsloth.ai/get-started/all-our-models)
- [Model Selection Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/what-model-should-i-use)

## まとめ

Tesla T4 (15GB) では：
- ✅ **7B-8Bモデル**: 余裕で動く（推奨）
- ⚠️ **11B-14Bモデル**: 設定次第で動く
- ❌ **30B以上**: 厳しい（マルチGPU必要）

**2026年最新のおすすめ**: Llama 4 8B または Qwen3 7B
