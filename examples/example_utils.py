import torch
import os
os.environ['TRANSFORMERS_CACHE'] = '/workspace/cache/' # must be set before import
from transformers import AutoModelForCausalLM, AutoTokenizer

def vanilla_generate_wrapper(model, tokenizer, user_tag, assistant_tag, device):
    
    def vanilla_generate(prompt, max_new_tokens=12, **kwargs):
        template_str = '{user_tag}{prompt}{assistant_tag}'
        prompt = template_str.format(user_tag=user_tag, prompt=prompt, assistant_tag=assistant_tag)
        model_inputs = tokenizer(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(**model_inputs, max_new_tokens=max_new_tokens, **kwargs)
            text = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(text)
            
    return vanilla_generate


def load_model(model_name_or_path, revision, device):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device, revision=revision, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, padding_side="left")
    tokenizer.pad_token_id = 0
    return model, tokenizer