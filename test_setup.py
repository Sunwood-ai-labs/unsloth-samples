"""
Unslothのインストール確認用テストスクリプト
"""

print("=" * 50)
print("Unsloth環境テスト")
print("=" * 50)

# 基本的なインポートテスト
print("\n1. 基本パッケージのインポートテスト...")
try:
    import torch
    print(f"   ✅ torch {torch.__version__}")
except Exception as e:
    print(f"   ❌ torch インポートエラー: {e}")

try:
    import transformers
    print(f"   ✅ transformers {transformers.__version__}")
except Exception as e:
    print(f"   ❌ transformers インポートエラー: {e}")

try:
    import datasets
    print(f"   ✅ datasets {datasets.__version__}")
except Exception as e:
    print(f"   ❌ datasets インポートエラー: {e}")

try:
    from unsloth import FastLanguageModel
    print(f"   ✅ unsloth インポート成功")
except Exception as e:
    print(f"   ❌ unsloth インポートエラー: {e}")

# CUDA確認
print("\n2. CUDA環境の確認...")
try:
    print(f"   CUDA利用可能: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA バージョン: {torch.version.cuda}")
        print(f"   GPU数: {torch.cuda.device_count()}")
        print(f"   GPU名: {torch.cuda.get_device_name(0)}")
        print(f"   GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
except Exception as e:
    print(f"   エラー: {e}")

# 軽量モデルのロードテスト
print("\n3. モデルロードテスト（軽量版）...")
try:
    print("   TinyLlamaをロード中...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/tinyllama-bnb-4bit",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    print("   ✅ モデルロード成功！")

    # 簡単な推論テスト
    print("\n4. 推論テスト...")
    FastLanguageModel.for_inference(model)
    inputs = tokenizer(["Hello, how are you?"], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=20, use_cache=True)
    result = tokenizer.batch_decode(outputs)[0]
    print(f"   入力: Hello, how are you?")
    print(f"   出力: {result[:100]}...")
    print("   ✅ 推論テスト成功！")

except Exception as e:
    print(f"   ❌ エラー: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("テスト完了！")
print("=" * 50)
