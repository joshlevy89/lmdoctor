from .utils import format_prompt

class Controller:
    """
    Wrapper around model that enables it to control generation by manipulating representation of
    concepts at inference time. 
    """
    def __init__(self, extractor, device='cuda:0', transformer_layers=None):
        self.model = extractor.model
        self.tokenizer = extractor.tokenizer
        self.user_tag = extractor.user_tag
        self.assistant_tag = extractor.assistant_tag
        self.directions = extractor.direction_info['directions']
        if transformer_layers is None:
            self.transformer_layers = self.get_transformer_layers()

    def _clear_hooks(self, model):
        for module in self.transformer_layers:
            module._forward_hooks.clear()
    
    def generate(self, prompt, control_direction=None, n_trim_layers=10, alpha=1, 
                 control_gen=True, control_prompt=False, should_format_prompt=True, **kwargs):
        """
        Adds/subtracts representation of a concept at inference time. 
        control_direction: 1 adds the vector, -1 subtracts it
        alpha: multiplicative factor applied to vector. alpha of 0 would apply no control
        n_trim_layers: number of layers to NOT manipulate on either side of model. 0 would manipulate all layers.
        control_gen/control_prompt: applies control element to generated text and input prompt, respectively. if both
        set to False, no control is applied.
        """
        if control_direction is None:
            raise ValueException('Must set control_direction to either +1 (adds vector) or -1 (subtracts vector)')

        # add hooks 
        self._clear_hooks(self.model) # good practice to clear hooks first
        start_layer, end_layer = n_trim_layers, self.model.config.num_hidden_layers-n_trim_layers
        layers = range(start_layer, end_layer)
        for layer in layers:
            def hook(m, inp, op):
                if op[0].shape[1] > 1: # corresponds to the prompt, which is passed as a chunk
                    if control_prompt:
                        op[0][0, :, :] += alpha * self.directions[layer] / self.directions[layer].norm()  * control_direction
                    else:
                        return op
                else: # corresponds to the text generation, which is passed one at a time
                    if control_gen:
                        op[0][0, 0, :] += alpha * self.directions[layer] / self.directions[layer].norm()  * control_direction
                    else:
                        return op
                return op
            # per https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L710, 
            # the first value in module output (used in hook) is the input to the layer
            h = self.transformer_layers[layer].register_forward_hook(hook)
            
        # generate after hooks have been
        if should_format_prompt:
            prompt = format_prompt(prompt, self.user_tag, self.assistant_tag)
        model_inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        output = self.model.generate(**model_inputs, **kwargs)
        text = self.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        self._clear_hooks(self.model)
        return text

    
    def get_transformer_layers(self):
        try:
            return self.model.model.layers # mistral and llama
        except:
            try:
                return self.model.transformer.h # gpt-2
            except:
                raise ValueError('Could not locate the transformer layers module list within model. Can pass the module' \
                                 'directly with transformer_layers when instantiating controller')

    def __getattr__(self, name):
        return getattr(self.model, name)
        
    