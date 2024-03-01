from .utils import format_prompt

class Controller:
    """
    Wrapper around model that enables it to control generation by manipulating representation of
    concepts at inference time. 
    """
    def __init__(self, direction_info, model, tokenizer, user_tag, assistant_tag, device='cuda:0'):
        self.model = model
        self.tokenizer = tokenizer
        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.directions = direction_info['directions']

    def _clear_hooks(self, model):
        for module in model.model.layers:
            module._forward_hooks.clear()
    
    def generate(self, prompt, control_direction=None, n_trim_layers=10, alpha=1, **kwargs):
        """
        Adds/subtracts representation of a concept at inference time. 
        control_direction: 1 adds the vector, -1 subtracts it
        alpha: multiplicative factor applied to vector
        n_trim_layers: number of layers to NOT manipulate on either side of model. 0 would manipulate all layers.
        """
        if control_direction is None:
            raise ValueException('Must set control_direction to either +1 (adds vector) or -1 (subtracts vector)')

        # add hooks 
        self._clear_hooks(self.model) # good practice to clear hooks first
        start_layer, end_layer = n_trim_layers, self.model.config.num_hidden_layers-n_trim_layers
        layers = range(start_layer, end_layer)
        for layer in layers:
            def hook(m, inp, op):
                if op[0].shape[1] > 1:
                    # Doesn't effect the text produced, but as a good practice, this will skip over the input prompt (which is passed as a group of tokens)
                    return op
                op[0][0, 0, :] += alpha * self.directions[layer] / self.directions[layer].norm()  * control_direction
                return op
            # per https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L710, 
            # the first value in module output (used in hook) is the input to the layer
            h = self.model.model.layers[layer].register_forward_hook(hook)
            
        # generate after hooks have been
        prompt = format_prompt(prompt, self.user_tag, self.assistant_tag)
        model_inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        output = self.model.generate(**model_inputs, **kwargs)
        text = self.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        self._clear_hooks(self.model)
        return text

    def __getattr__(self, name):
        return getattr(self.model, name)