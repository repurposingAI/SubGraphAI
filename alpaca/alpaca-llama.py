import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


model_name = "meta-llama/Llama-2-7b-chat-hf"

# The instruction dataset to use
dataset_name = "starmpcc/Asclepius-Synthetic-Clinical-Notes"

# Fine-tuned model name
new_model = "Llama-2-7b-chat-hf"


lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1



# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False


# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25



# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False




alpaca_prompt = """You are an intelligent clinical language model.
Below is a snippet of patient's electronic health record note and a following instruction with question from healthcare professional.
Write a response that appropriately completes the instruction.
The response should provide the accurate answer to the instruction, while being concise.

### Instruction:
{}

### Patient's Electronic Health Record Note:
{}

### Question:
{}

### Response:
{}
"""



def formatting_prompts_func(examples):
    instructions = examples["task"]
    inputs       = examples["question"]
    outputs      = examples["answer"]
    context      = examples["note"]
    return {"text": [alpaca_prompt.format(instruction, note, input, output) for instruction, note, input, output in zip(instructions, context, inputs, outputs)]}


dataset = load_dataset(dataset_name, split='train[:2000]')
dataset = dataset.map(formatting_prompts_func, batched = True)




compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)


# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training



# Load LoRA configuration
peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)


trainer.train()
trainer.model.save_pretrained(new_model)



# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our next model
prompt = """
You are an intelligent clinical languge model.
Below is a snippet of patient's electronic health record note and a following instruction with question from healthcare professional.
Write a response that appropriately completes the instruction.
The response should provide the accurate answer to the instruction, while being concise.

### Instruction:
Paraphrasing

### Patient's Electronic Health Record Note:
Discharge Summary:

Patient: 60-year-old male with moderate ARDS from COVID-19

Hospital Course:

The patient was admitted to the hospital with symptoms of fever, dry cough, and dyspnea. During physical therapy on the acute ward, the patient experienced coughing attacks that induced oxygen desaturation and dyspnea with any change of position or deep breathing. To avoid rapid deterioration and respiratory failure, a step-by-step approach was used for position changes. The breathing exercises were adapted to avoid prolonged coughing and oxygen desaturation, and with close monitoring, the patient managed to perform strength and walking exercises at a low level. Exercise progression was low initially but increased daily until hospital discharge to a rehabilitation clinic on day 10.

Clinical Outcome:

The patient was discharged on day 10 to a rehabilitation clinic making satisfactory progress with all symptoms resolved.

Follow-up:

The patient will receive follow-up care at the rehabilitation clinic, with regular monitoring of progress and further rehabilitation exercises until full recovery. Any new symptoms or concerns should be reported to the clinic immediately.

Overall Impression:

The patient responded well to treatment, and with appropriate medical intervention, was able to overcome the difficulties faced during hospitalization for ARDS from COVID-19. The patient's level of care was of a high standard, with all necessary therapy provided and monitoring of progress before discharge.

### Question:
Can you provide a simplified paraphrase of the sentence, 'To avoid rapid deterioration and respiratory failure, a step-by-step approach was used for position changes' in the patient's discharge summary?
"""
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
result = pipe(f"[INST] {prompt} [/INST]")
print(result[0]['generated_text'])



import gc
del model
del pipe
del trainer
gc.collect()




# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"



prompt = """
You are an intelligent clinical languge model.
Below is a snippet of patient's electronic health record note and a following instruction with question from healthcare professional.
Write a response that appropriately completes the instruction.
The response should provide the accurate answer to the instruction, while being concise.

### Instruction:
Abbreviation Expansion

### Patient's Electronic Health Record Note:
Hospital Course: 

This 66-year-old male patient was admitted due to an ischemic left-hemispheric stroke in addition to a dry cough and fever. The patient tested positive for SARS-CoV-2 and experienced severe ARDS, resulting in intubation and ICU admission. The patient underwent veno-venous extracorporeal membrane oxygenation and physical therapy was initiated to focus on perception training, movement exercises, airway-clearing techniques, dysphagia therapy, and mobilization. Despite a trial of sedation cessation, the patient remained somnolent and unable to communicate or follow commands. A side-edge positioning was initiated in combination with intensive exercise training including trunk and head control. Muscle tone and strength remained severely reduced, particularly on his hemiplegic side, and a second SOEB trial failed. Occupational therapy was involved to support functional initiation of upper limb movements and to integrate perception-training into activities of daily living. Currently, the patient remains functionally dependent, tolerates spontaneous breathing trials, and is alert during therapy, although he cannot communicate. He is considered stable and functionally dependent (CPAx 6/50).

### Question:
What are the abbreviated terms in the given discharge summary that require expansion?
"""
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
result = pipe(f"[INST] {prompt} [/INST]",max_length=584)[0]['generated_text']
start_index = result.find("[/INST]") + len("[/INST]")
end_index = result.find("'", start_index)
response = result[start_index:end_index]
print(response)


from huggingface_hub import notebook_login
notebook_login()


model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)