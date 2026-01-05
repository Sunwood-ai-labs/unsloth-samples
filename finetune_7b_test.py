"""
7Bモデルを使用したファインチューニング動作確認
中規模モデルでGPU性能を活用
"""

from unsloth import FastLanguageModel
import torch

print("=" * 50)
print("7B Model Fine-tuning Test")
print("GPU: Tesla T4 (15GB)")
print("=" * 50)

# Tesla T4用の設定
max_seq_length = 2048
dtype = None
load_in_4bit = True

# 7Bモデルの選択（日本語に強いQwen2.5-7B）
print("\n📦 Qwen 2.5 7B モデルをロード中...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
print("✅ モデルロード完了")

print("\n🔧 LoRA設定を適用中...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
print("✅ LoRA設定完了")

# データセット準備
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_prompt.format(instruction, output)
        texts.append(text)
    return {"text": texts}

print("\n📚 データセットをロード中...")
from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:500]")
dataset = dataset.map(formatting_prompts_func, batched=True)
print(f"✅ データセット準備完了（{len(dataset)}件）")

# トレーニング設定
from trl import SFTTrainer
from transformers import TrainingArguments

print("\n⚙️  トレーナーを設定中...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,  # 7Bモデル用
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=20,  # 動作確認用
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_7b_test",
        report_to="none",
    ),
)
print("✅ トレーナー設定完了")

# GPU メモリ使用状況
print("\n💾 トレーニング前のGPUメモリ:")
print(f"   割り当て済み: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"   予約済み: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# トレーニング実行
print("\n🚀 トレーニングを開始します（20ステップ）...")
print("=" * 50)
trainer_stats = trainer.train()
print("=" * 50)

# GPU メモリ使用状況
print("\n💾 トレーニング後のGPUメモリ:")
print(f"   割り当て済み: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"   予約済み: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
print(f"   最大使用量: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

# モデルの保存
print("\n💾 モデルを保存中...")
model.save_pretrained("finetuned_qwen25_7b_test")
tokenizer.save_pretrained("finetuned_qwen25_7b_test")
print("✅ モデル保存完了")

print(f"\n📊 トレーニング統計:")
print(f"   損失: {trainer_stats.training_loss:.4f}")
print(f"   実行時間: {trainer_stats.metrics['train_runtime']:.2f}秒")
print(f"   ステップ/秒: {trainer_stats.metrics['train_samples_per_second']:.2f}")

# 推論テスト
print("\n🧪 推論テストを実行中...")
FastLanguageModel.for_inference(model)

test_prompts = [
    "日本の首都はどこですか？",
    "Pythonで Hello World を出力する方法を教えてください。",
    "機械学習とディープラーニングの違いは何ですか？",
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n[テスト {i}]")
    print(f"質問: {prompt}")

    inputs = tokenizer(
        [alpaca_prompt.format(prompt, "")],
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True,
        temperature=0.7,
        top_p=0.9,
    )

    result = tokenizer.batch_decode(outputs)[0]

    # Response部分のみ抽出
    if "### Response:" in result:
        response = result.split("### Response:")[1].strip()
        # EOSトークンなどを除去
        if "<|endoftext|>" in response:
            response = response.split("<|endoftext|>")[0].strip()
        print(f"回答: {response[:300]}")
    else:
        print(f"回答: {result[:300]}")
    print("-" * 50)

print("\n" + "=" * 50)
print("✅ 7Bモデル ファインチューニング動作確認完了！")
print("=" * 50)

# モデル情報サマリー
print("\n📝 モデル情報:")
print(f"   モデル名: Qwen 2.5 7B (4bit)")
print(f"   パラメータ数: 約70億")
print(f"   トレーニング可能パラメータ: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"   最大GPUメモリ使用量: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB / 14.74 GB")
print(f"   GPU使用率: {(torch.cuda.max_memory_allocated() / 1024**3) / 14.74 * 100:.1f}%")
