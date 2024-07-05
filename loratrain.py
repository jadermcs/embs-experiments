from peft import LoraConfig, TaskType
from peft import get_peft_model
from transformers import AutoModelForCausalLM

peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())
