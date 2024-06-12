import torch
from subprocess import call

try:
    major_version, minor_version = torch.cuda.get_device_capability()
except RuntimeError:
    major_version = 0

if major_version >= 8:
    call(["pip", "install", "--no-deps", "packaging", "ninja", "einops", "flash-attn", "xformers", "trl", "peft", "accelerate", "bitsandbytes"])
else:
    call(["pip", "install", "--no-deps", "xformers", "trl", "peft", "accelerate", "bitsandbytes"])

from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
import json


# data = load_dataset("qiaojin/PubMedQA", "pqa_labelled")

# Check CUDA availability
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Model parameters
model_name = "unsloth/llama-3-8b-bnb-4bit"
max_seq_length = 2048
dtype = torch.float16
load_in_4bit = True

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name, 
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit
)

EOS_TOKEN = tokenizer.eos_token 
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
     
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# Training the model
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=data['train'],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()

FastLanguageModel.for_inference(model) 
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", 
        "1, 1, 2, 3, 5, 8", # input
        "", 
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

model.save_pretrained("lora_model") 
tokenizer.save_pretrained("lora_model")

if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", 
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) 



inputs = tokenizer(
[
    alpaca_prompt.format(
        "What is a famous tall tower in Paris?",
        "",
        "", 
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)
