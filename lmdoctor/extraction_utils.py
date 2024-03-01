"""
Utils for extracting representations associated with a function, e.g. honesty
"""

from .honesty_utils import fetch_honesty_data
import numpy as np
from collections import defaultdict
import torch
from sklearn.decomposition import PCA
import torch.nn.functional as F

EXTRACTION_TARGET_MAP = {'honesty': fetch_honesty_data()}
#                          # 'morality': prepare_morality_data}

class Extractor:
    def __init__(self, model, tokenizer, user_tag, assistant_tag, extraction_target=None):

        if extraction_target is None:
            raise ValueError(f"Must specify extraction_target. Must be one of {list(EXTRACTION_TARGET_MAP)}")
        
        self.model = model
        self.tokenizer = tokenizer
        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.extraction_target = extraction_target
        self.direction_info = None
        self.statement_pairs = None
        self.train_act_pairs = None
        
    def find_directions(self, sample_range=[0, 512]):
        data, prompt_maker = EXTRACTION_TARGET_MAP[self.extraction_target]
        self.statement_pairs = prepare_statement_pairs(
            data, prompt_maker, self.tokenizer, user_tag=self.user_tag, assistant_tag=self.assistant_tag)
        self.train_act_pairs = get_activations_for_paired_statements(
            self.statement_pairs, self.model, self.tokenizer, sample_range=sample_range)   
        self.direction_info = get_directions(self.train_act_pairs)
    

def prepare_statement_pairs(data, _prompt_maker, tokenizer, user_tag, assistant_tag):

    statement_pairs = []
    statements = data[data['label'] == 1]['statement'].values.tolist() # they only use label=1 for some reason
    for statement in statements:
        tokens = tokenizer.tokenize(statement)
        for idx in range(1, len(tokens)-5):
            substatement = tokenizer.convert_tokens_to_string(tokens[:idx])
            honest_statement = _prompt_maker(substatement, True, user_tag, assistant_tag)
            dishonest_statement = _prompt_maker(substatement, False, user_tag, assistant_tag)
            statement_pairs.append([honest_statement, dishonest_statement])
    statement_pairs = np.array(statement_pairs)
    return statement_pairs


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
    scaled_directions = {}
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
        direction = torch.tensor(pca.components_[0], dtype=torch.float16).to(device)
        directions[layer] = direction 
        
        # scale direction such that p(mu_a + scaled_direction) = p(mu_b), following marks et al. (https://arxiv.org/abs/2310.06824)
        act_pairs = train_acts[layer]
        mu_a = torch.mean(act_pairs[:, 0, :], axis=0)
        mu_b = torch.mean(act_pairs[:, 1, :], axis=0)
        norm_direction = F.normalize(direction, dim=0)
        diff = (mu_a - mu_b) @ norm_direction.view(-1)
        scaled_direction = (norm_direction * diff).view(-1)
        scaled_directions[layer] = scaled_direction
    
    direction_info['unscaled_directions'] = directions # these aren't used, but kept for posterity
    direction_info['directions'] = scaled_directions
    direction_info['mean_diffs'] = mean_diffs
    return direction_info

    
# def get_accs_for_pairs(test_acts, direction_info):
#     """
#     This could be used for testing accuracy of on the extraction dataset (in-distribution).
#     Not being used at the moment.
#     """
#     from .utils import do_projections
#     directions = direction_info['directions']
#     mean_diffs = direction_info['mean_diffs']
#     accs = []
#     for layer in test_acts:
#         projections = do_projections(test_acts[layer], directions[layer], mean_diffs[layer])
#         acc = np.mean([(pi > pj).item() for (pi, pj) in projections])
#         accs.append(acc)
    return accs