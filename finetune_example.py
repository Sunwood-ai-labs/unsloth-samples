"""
Unslothを使用したLLMファインチューニングの基本例
"""

from unsloth import FastLanguageModel
import torch

# ハイパーパラメータの設定
max_seq_length = 2048
dtype = None  # Noneにすると自動検出
load_in_4bit = True  # 4bit量子化を使用してVRAMを削減

# モデルとトークナイザーのロード
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/tinyllama",  # 軽量なモデルでテスト
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# LoRAの設定 (Parameter Efficient Fine-Tuning)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRAのランク
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

# データセットの準備（サンプル）
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

# トレーニングデータの例
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_prompt.format(instruction, output)
        texts.append(text)
    return {"text": texts}

# データセットのロード（例：Hugging Face datasetsから）
from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")  # 最初の1000件のみ
dataset = dataset.map(formatting_prompts_func, batched=True)

# トレーニング設定
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
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,  # 軽いトレーニングのため少なめに設定
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # wandbを無効化
    ),
)

# トレーニング実行
print("トレーニングを開始します...")
trainer_stats = trainer.train()

# モデルの保存
print("モデルを保存しています...")
model.save_pretrained("finetuned_model")
tokenizer.save_pretrained("finetuned_model")

print("ファインチューニングが完了しました！")
print(f"トレーニング統計: {trainer_stats}")

# 推論テスト
FastLanguageModel.for_inference(model)
inputs = tokenizer(
    [alpaca_prompt.format(
        "日本の首都はどこですか？",
        "",
    )],
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
print("\n推論結果:")
print(tokenizer.batch_decode(outputs))
