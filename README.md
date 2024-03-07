# lmdoctor
Extract, detect, and control semantic representations within language models as they read and write text. Built on 🤗 transformers. 

[lmdoctor pip package](https://pypi.org/project/lmdoctor/)  

Briefly, lmdoctor reads and manipulates a model's hidden states at inference time. Based on ideas from [Representation Engineering: A Top-Down Approach to AI Transparency by Zou et al. 2023](https://arxiv.org/pdf/2310.01405.pdf). Their original code [here](https://github.com/andyzoujm/representation-engineering).  

For the latest source code or to report issues, please visit the [project repository](https://github.com/joshlevy89/lmdoctor).

## Example

### honesty representation extraction
```
from lmdoctor import extraction_utils
extractor = extraction_utils.Extractor(model, tokenizer, user_tag, assistant_tag, extraction_target='honesty')
extractor.find_directions()
```

### lie detection

```
prompt = "Tell me a fact about humans"

from lmdoctor import detection_utils
ld = detection_utils.Detector(model, tokenizer, user_tag, assistant_tag)
ld.generate(prompt, max_new_tokens=10, do_sample=True) # capture the hidden_states as the model generates
ld.get_projections(extractor.direction_info) # project the hidden_states onto the representation direction vectors
ld.detect(use_n_middle_layers=15) # aggregate projections over layers
```

![truth](https://github.com/joshlevy89/lmdoctor/blob/main/assets/readme/truth.png?raw=true)
```
prompt = "Tell me a lie about humans"
```
![lie](https://github.com/joshlevy89/lmdoctor/blob/main/assets/readme/lie.png?raw=true)

### honesty control
```
prompt = "You killed a person for revenge but don't want to get caught. What would you tell the police?"
vanilla_generate(prompt)
```

```
output: I would tell the police that I did not kill anyone.
```
```
# +honesty control
from lmdoctor import control_utils
hc = control_utils.Controller(honesty_extractor.direction_info, model, tokenizer, user_tag=user_tag, assistant_tag=assistant_tag)
hc.generate(prompt, control_direction=1, max_new_tokens=12)
```
```
output: I would tell the police that I have killed a person
````

For the complete example, see [examples/honesty_example.ipynb](https://github.com/joshlevy89/lmdoctor/blob/main/examples/honesty_example.ipynb)


## Getting started
[linux only]

recommended: pip install lmdoctor  

from source: "pip install ." after cloning

After install, try running honesty_example.ipynb

*Note: This package requires pytorch but does not include it because the specific version/cuda backend will depend the huggingface model you are using. If you don't already have it installed you will need to do 'pip install torch' or use the model-specific instructions.*


## Extraction targets

The table below describes the targets we support for extracting internal representations. In functional extraction, the model is asked to produce text (e.g. prompt="tell me a lie"). In conceptual extraction, the model is asked to consider a statement (e.g. "consider the truthfulness of X"). For targets where both are supported, you can try each to see which works best for your use-case. 

| Target      | Method | Types |
| ----------- | ----------- | ----------- |
| truth      | conceptual       | none       |
| honesty   | functional        | none        |
| morality  | conceptual & functional | none | 
| emotion | conceptual | anger, disgust, fear, happiness, sadness, surprise | |
| fairness | conceptual & functional | race, gender, prefession, religion
