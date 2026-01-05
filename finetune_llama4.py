"""
Llama 4を使用したファインチューニング例（Tesla T4最適化版）
最新モデルで高性能なチャットボットを作成
"""

from unsloth import FastLanguageModel
import torch

print("=" * 50)
print("Llama 4 Fine-tuning Example")
print("GPU: Tesla T4 (15GB)")
print("=" * 50)

# Tesla T4用の最適化設定
max_seq_length = 2048
dtype = None  # 自動検出
load_in_4bit = True  # 4bit量子化でVRAM節約

# Llama 4モデルのロード
print("\n📦 Llama 4モデルをロード中...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-4-8B-bnb-4bit",  # 4bit版
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

print("✅ モデルロード完了")

# LoRA設定（効率的なファインチューニング）
print("\n🔧 LoRA設定を適用中...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRAランク
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # メモリ節約
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

print("✅ LoRA設定完了")

# データセットの準備
print("\n📚 データセットを準備中...")

# Alpacaフォーマット
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples.get("instruction", [""] * len(examples.get("output", [])))
    inputs = examples.get("input", [""] * len(examples.get("output", [])))
    outputs = examples.get("output", [])
    texts = []

    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_text, output)
        texts.append(text)
    return {"text": texts}

# データセットのロード
from datasets import load_dataset

# 日本語データセット例（または英語のalpaca-cleaned）
try:
    # 日本語データセットを試す
    dataset = load_dataset("kunishou/databricks-dolly-15k-ja", split="train[:1000]")
except:
    # フォールバック: 英語データセット
    print("⚠️ 日本語データセットが見つからないため、英語データセットを使用します")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")

dataset = dataset.map(formatting_prompts_func, batched=True)

print(f"✅ データセット準備完了（{len(dataset)}件）")

# トレーニング設定（Tesla T4最適化）
print("\n⚙️ トレーニング設定...")
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=4,  # Tesla T4で余裕のあるバッチサイズ
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=200,  # デモ用（本番は1000以上推奨）
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_llama4",
        save_strategy="steps",
        save_steps=100,
        report_to="none",  # wandbを無効化
    ),
)

# メモリ使用状況を表示
print("\n💾 現在のGPUメモリ使用状況:")
print(f"割り当て済み: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"予約済み: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# トレーニング実行
print("\n🚀 トレーニングを開始します...")
print("=" * 50)

trainer_stats = trainer.train()

print("\n" + "=" * 50)
print("✅ トレーニング完了！")
print("=" * 50)

# モデルの保存
print("\n💾 モデルを保存中...")
output_dir = "finetuned_llama4"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ モデルを '{output_dir}' に保存しました")

# 推論テスト
print("\n🧪 推論テストを実行中...")
FastLanguageModel.for_inference(model)

test_prompts = [
    ("日本の首都はどこですか？", ""),
    ("Pythonでリストをソートする方法を教えてください", ""),
    ("機械学習とは何ですか？", "簡単に説明してください"),
]

print("\n" + "=" * 50)
print("推論結果")
print("=" * 50)

for instruction, input_text in test_prompts:
    prompt = alpaca_prompt.format(instruction, input_text, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True,
        temperature=0.7,
        top_p=0.9,
    )

    result = tokenizer.batch_decode(outputs)[0]

    print(f"\n📝 質問: {instruction}")
    if input_text:
        print(f"💡 補足: {input_text}")
    print(f"🤖 回答:")
    # レスポンス部分のみ抽出
    response_start = result.find("### Response:") + len("### Response:")
    response = result[response_start:].strip()
    print(response[:500])  # 最初の500文字
    print("-" * 50)

print("\n✨ すべての処理が完了しました！")

# 最終メモリ使用状況
print("\n💾 最終GPUメモリ使用状況:")
print(f"割り当て済み: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"予約済み: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
print(f"最大使用量: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
