import torch
from collections import defaultdict
import numpy as np

class Detector:
    """
    Wraps model in order to capture hidden_states during generation and perform computations with those hidden_states.
    Specific detectors (e.g. LieDetector) inherit from it. 
    """
    def __init__(self, model, tokenizer, user_tag, assistant_tag, device='cuda:0'):
        self.model = model
        self.tokenizer = tokenizer
        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.device = device
        self.hiddens = None
        self.all_projs = None
    
    def generate(self, prompt, **kwargs):        
        """
        Ensures hidden_states are saved during generation.
        """
        kwargs['return_dict_in_generate'] = True
        kwargs['output_hidden_states'] = True

        prompt = _format_prompt(prompt, self.user_tag, self.assistant_tag)
        model_inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(**model_inputs, **kwargs)
        self.hiddens = output.hidden_states
        output_text = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return output_text

    def get_projections(self, direction_info, input_text=None):
        """
        Computes the projections of hidden_states onto concept directions.
        """
        directions = direction_info['directions']
        mean_diffs = direction_info['mean_diffs']
        
        if input_text:
            layer_to_acts = _get_layeracts_from_text(input_text, self.model, self.tokenizer)
        else:
            if self.hiddens is None:
                raise RuntimeError('Missing hidden states. Must either run generate method before calling get_projections OR provide an input_text')
            layer_to_acts = _get_layeracts_from_hiddens(self.hiddens)

        all_projs = []
        for layer in layer_to_acts:
            projs = do_projections(layer_to_acts[layer], directions[layer], mean_diffs[layer], layer=layer)
            all_projs.append(projs)
        self.all_projs = torch.stack(all_projs)
        return self.all_projs

    def detect(self, layers_to_use=None, use_n_middle_layers=None):
        """
        Logic for aggregating over layers to score tokens for degree of lying. 
        set one of:
            layers_to_use: -1 (all layers) or [start_layer, end_layer]
            use_n_middle_layers: number of middle layers to use
        """        
        
        if layers_to_use:
            if use_n_middle_layers:
                raise ValueError('Either specify layers_to_use or use_n_middle_layers, not both')
            if layers_to_use == -1:
                layers_to_use = range(0, self.model.config.num_hidden_layers)
            else:
                layers_to_use = range(layers_to_use[0], layers_to_use[1])
        elif use_n_middle_layers:
            n_layers = self.model.config.num_hidden_layers
            mid = n_layers // 2
            diff = use_n_middle_layers // 2
            layers_to_use = range(max(0, mid - diff), min(mid + diff, n_layers))
        else:
            raise ValueError('Either specify layers_to_use or use_n_middle_layers')
            
        layer_avg = self.all_projs[layers_to_use, :].mean(axis=0)
        layer_avg = layer_avg.detach().cpu().numpy()
        layer_avg = layer_avg.reshape(1, -1)
        return layer_avg
    
    def __getattr__(self, name):
        return getattr(self.model, name)


class ConceptController:
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
        prompt = _format_prompt(prompt, self.user_tag, self.assistant_tag)
        model_inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        output = self.model.generate(**model_inputs, **kwargs)
        text = self.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        self._clear_hooks(self.model)
        return text

    def __getattr__(self, name):
        return getattr(self.model, name)
        

def _format_prompt(prompt, user_tag, assistant_tag):
    template_str = '{user_tag} {prompt} {assistant_tag}'
    prompt = template_str.format(user_tag=user_tag, prompt=prompt, assistant_tag=assistant_tag)
    return prompt


def do_projections(acts, direction, mean_diff, center=True, layer=None):
    if center:
        acts = (acts - mean_diff).clone()
    projections =  acts @ direction / direction.norm() # taking norm
    return projections


def _get_layeracts_from_hiddens(hiddens):
    """
    Get activations per layer directly from the hiddens produced during generation.
    Will be roughly equivalent to get_layeracts_from_text but may differ slightly due to floating point precision.
    Useful if want to avoid re-running the model (e.g. if batching samples).
    # Caveat: won't have hiddens for final token (bc not input to model).
    """
    layer_to_acts = defaultdict(list)
    for output_token in range(len(hiddens)): 
        token_hiddens = hiddens[output_token]
        for layer in range(0, len(token_hiddens)-1):
            layer_to_acts[layer].append(token_hiddens[layer+1][0, :, :])
    for layer in layer_to_acts:
        layer_to_acts[layer] = torch.cat(layer_to_acts[layer])
    return layer_to_acts


def get_layeracts_from_text(input_text, model, tokenizer):
    """
    Get activations per layer from the generated text (which includes prompt). 
    This requires re-running the bc the model was already used to generate the text.
    Useful if just have the text (and not the hiddens produced during generation).
    """
    model_inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)
    
    layer_to_acts = {}
    with torch.no_grad():
        hiddens = model(**model_inputs, output_hidden_states=True)
    for layer in range(model.config.num_hidden_layers):
        layer_to_acts[layer] = hiddens['hidden_states'][layer+1].squeeze()
    return layer_to_acts