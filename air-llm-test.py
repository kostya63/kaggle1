import os
from airllm import AutoModel

#os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/sdb1/projects"
print(os.environ["HUGGINGFACE_HUB_CACHE"])

MAX_LENGTH = 128
# could use hugging face model repo id:
#model = AutoModel.from_pretrained("garage-bAInd/Platypus2-70B-instruct")

# or use model's local path...
model = AutoModel.from_pretrained("/mnt/sdb1/projects/models--garage-bAInd--Platypus2-70B-instruct/snapshots/31389b50953688e4e542be53e6d2ab04d5c34e87", compression='4bit')

input_text = [
        'What is the capital of United States?',
    ]

model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=True)
           
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)