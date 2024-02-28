# lmdoctor
pip pkg for extracting and controlling concepts within language models as they generate text

built on 🤗 transformers

it reads the model's activations during inference to determine how certain concepts (e.g. honesty) are representated, and how to control them.

based on ideas from [Representation Engineering: A Top-Down Approach to AI Transparency by Zou et al. 2023](https://arxiv.org/pdf/2310.01405.pdf). their original code [here](https://github.com/andyzoujm/representation-engineering).

for the latest source code or to report issues, please visit the [project repository](https://github.com/joshlevy89/lmdoctor).

## Example

```
prompt = "Tell me a fact about humans"

from lmdoctor import honesty_utils
ld = honesty_utils.LieDetector(model, tokenizer, user_tag, assistant_tag)
text = ld.generate(prompt, max_new_tokens=10, do_sample=True) # capture the hidden_states as the model generates
all_projs = ld.get_projections(honesty_extractor.direction_info) # project the hidden_states onto the direction vectors from honesty extraction
```

![truth](https://github.com/joshlevy89/lmdoctor/blob/main/assets/readme/truth.png?raw=true)
```
prompt = "Tell me a lie about humans"
```
![lie](https://github.com/joshlevy89/lmdoctor/blob/main/assets/readme/lie.png?raw=true)


## Getting started
[linux only]

pip install lmdoctor

Note: This package requires pytorch but does not include it because the specific version/cuda backend will depend the huggingface model you are using. If you don't already have it installed you will need to do 'pip install torch' or use the model-specific instructions.

After install, try running honesty_example.ipynb