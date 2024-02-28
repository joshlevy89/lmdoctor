import torch
from collections import defaultdict
from sklearn.decomposition import PCA
import numpy as np

class Detector:
    def __init__(self, model, tokenizer, user_tag, assistant_tag, device='cuda:0'):
        self.model = model
        self.tokenizer = tokenizer
        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.device = device
        self.hiddens = None
        self.all_projs = None
    
    def generate(self, prompt, **kwargs):

        template_str = '{user_tag} {prompt} {assistant_tag}'
        prompt = template_str.format(user_tag=self.user_tag, prompt=prompt, assistant_tag=self.assistant_tag)
        model_inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        kwargs['return_dict_in_generate'] = True
        kwargs['output_hidden_states'] = True
        with torch.no_grad():
            output = self.model.generate(**model_inputs, **kwargs)
        self.hiddens = output.hidden_states
        output_text = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return output_text

    def get_projections(self, direction_info, input_text=None):
        directions = direction_info['directions']
        signs = direction_info['signs']
        mean_diffs = direction_info['mean_diffs']
        
        if input_text:
            layer_to_acts = _get_layeracts_from_text(input_text, self.model, self.tokenizer)
        else:
            if self.hiddens is None:
                raise RuntimeError('Missing hidden states. Must either run generate method before calling get_projections OR provide an input_text')
            layer_to_acts = _get_layeracts_from_hiddens(self.hiddens)

        all_projs = []
        for layer in layer_to_acts:
            projs = do_projections(layer_to_acts[layer], directions[layer], signs[layer], mean_diffs[layer], layer=layer)
            all_projs.append(projs)
        self.all_projs = torch.stack(all_projs)
        return self.all_projs
    
    def __getattr__(self, name):
        return getattr(self.model, name)
        

def get_activations_for_paired_statements(statement_pairs, model, tokenizer, sample_range, read_token=-1, batch_size=16, device='cuda:0'):
    layer_to_act_pairs = defaultdict(list)
    for i in range(sample_range[0], sample_range[1], batch_size):
        pairs = statement_pairs[i:i+batch_size]
        statements = pairs.reshape(-1)
        model_inputs = tokenizer(list(statements), padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            hiddens = model(**model_inputs, output_hidden_states=True)
        for layer in range(model.config.num_hidden_layers):
            act_pairs = hiddens['hidden_states'][layer+1][:, read_token, :].view(batch_size, 2, -1)
            layer_to_act_pairs[layer].extend(act_pairs)
    
    for key in layer_to_act_pairs:
        layer_to_act_pairs[key] = torch.stack(layer_to_act_pairs[key])

    return layer_to_act_pairs


def get_directions(train_acts, device='cuda:0'):
    directions = {}
    signs = {}
    mean_diffs = {}
    direction_info = defaultdict(dict)
    for layer in train_acts:
        act_pairs = train_acts[layer]
        shuffled_pairs = [] # shuffling train labels before pca useful for some reason 
        for pair in act_pairs:
            pair = pair[torch.randperm(2)]
            shuffled_pairs.append(pair)
        shuffled_pairs = torch.stack(shuffled_pairs)
        diffs = shuffled_pairs[:, 0, :] - shuffled_pairs[:, 1, :] 
        mean_diffs[layer] = torch.mean(diffs, axis=0)
        centered_diffs = diffs - mean_diffs[layer] # is centering necessary?
        pca = PCA(n_components=1)
        pca.fit(centered_diffs.detach().cpu())
        directions[layer] = torch.tensor(pca.components_[0], dtype=torch.float16).to(device)
        
        # get signs
        projections = do_projections(train_acts[layer], directions[layer], 1, mean_diffs[layer])
        acc = np.mean([(pi > pj).item() for (pi, pj) in projections])
        sign = -1 if acc < .5 else 1
        signs[layer] = sign
    direction_info['directions'] = directions
    direction_info['signs'] = signs
    direction_info['mean_diffs'] = mean_diffs
    return direction_info


def do_projections(acts, direction, sign, mean_diff, center=True, layer=None):
    if center:
        acts = (acts - mean_diff).clone()
    projections = sign * acts @ direction / direction.norm() # i don't think this projection is exactly right
    return projections


def get_accs_for_pairs(test_acts, direction_info):
    directions = direction_info['directions']
    signs = direction_info['signs']
    mean_diffs = direction_info['mean_diffs']
    accs = []
    for layer in test_acts:
        projections = do_projections(test_acts[layer], directions[layer], signs[layer], mean_diffs[layer])
        acc = np.mean([(pi > pj).item() for (pi, pj) in projections])
        accs.append(acc)
    return accs


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