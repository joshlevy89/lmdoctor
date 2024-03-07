import torch
from collections import defaultdict
from .utils import format_prompt

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
    
    def generate(self, prompt, gen_only=False, **kwargs):
        """
        If gen only, get hiddens/text for newly generated text only (i.e. exclude prompt)
        """
        kwargs['return_dict_in_generate'] = True
        kwargs['output_hidden_states'] = True

        prompt = format_prompt(prompt, self.user_tag, self.assistant_tag)
        model_inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(**model_inputs, **kwargs)
        
        if gen_only:
            sequences = output.sequences[:, model_inputs.input_ids.shape[1]:]
            self.hiddens = output.hidden_states[1:]
        else:
            sequences = output.sequences
            self.hiddens = output.hidden_states

        output_text = self.tokenizer.batch_decode(sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

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