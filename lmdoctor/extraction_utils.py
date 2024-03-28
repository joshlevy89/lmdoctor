"""
Utils for extracting representations associated with a function, e.g. honesty
"""
from .target_specific_utils.honesty_utils import fetch_factual_data_conceptual, fetch_factual_data_functional 
from .target_specific_utils.morality_utils import fetch_morality_data_conceptual, fetch_morality_data_functional
from .target_specific_utils.emotion_utils import fetch_emotion_data_wrapper
from .target_specific_utils.fairness_utils import fetch_fairness_data_conceptual_wrapper, fetch_fairness_data_functional_wrapper
from .target_specific_utils.harmlessness_utils import fetch_harmlessness_data_conceptual
from .probe_utils import probe_pca, probe_logreg, probe_massmean 


import numpy as np
from collections import defaultdict
import torch
from sklearn.decomposition import PCA
import torch.nn.functional as F
import random
from .utils import setup_logger
logger = setup_logger()


class Extractor:
    def __init__(self, model, tokenizer, user_tag, assistant_tag, device='cuda:0', 
                 extraction_target=None, extraction_method=None, probe_type='pca',
                 **kwargs):
        """
        kwargs: Additional keyword arguments, such as:
        - probe_type (str): The computation used to find directions. Supports pca, massmean, logreg
        - bias_type (str): The type of bias to use for fairness extraction.
        - emotion_type (str): The type of emotion to use for emotion extraction.
        - n_trim_tokens (int): The number of tokens to trim from the end of statements for functional extraction. Defaults to 5.
        - shuffle (bool): Whether to shuffle statements for conceptual extraction. Defaults to True.
        """

        if extraction_target is None:
            raise ValueError(f"Must specify extraction_target. Must be one of {list(get_extraction_target_map())}")
        
        self.model = model
        self.tokenizer = tokenizer
        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.extraction_target = extraction_target
        self.extraction_method = extraction_method
        self.probe_type = probe_type
        self.device = device
        self.kwargs = kwargs
        self.direction_info = None
        self.statement_pairs = None
        self.train_acts = None
        
    def extract(self, batch_size=8, n_train_pairs=128, n_dev_pairs=64, n_test_pairs=32):
        """
        n_train_pairs: how many statement pairs to use to calculate directions. setting to None will use all pairs. 
        """        
        self.statement_pairs = prepare_statement_pairs(
            self.extraction_target, self.extraction_method, self.tokenizer, 
            self.user_tag, self.assistant_tag, n_train_pairs, n_dev_pairs, n_test_pairs, **self.kwargs)
        self.train_acts = get_activations_for_paired_statements(
            self.statement_pairs['train'], self.model, self.tokenizer, batch_size, device=self.device)   
        self.direction_info = get_directions(self.train_acts, self.device, self.probe_type)
    

def get_extraction_function(target, extraction_method=None, **kwargs):
    """
    Get the data extraction function for the given target and extraction. 
    If no extraction_method supplied, tries to infer one. 
    """
    target_map = get_extraction_target_map(target, **kwargs)
    target_matches = [k for k in list(target_map) if k[0]==target] # all the tuple-keys that match the target
    if not target_matches:
        valid_targets = np.unique([k[0] for k in list(target_map)])
        raise ValueError(f"Extraction target must be one of {valid_targets}")
    if extraction_method:
        return target_map[(target, extraction_method)], extraction_method
    else:
        if len(target_matches) == 1:
            # Infer extraction_method, if only one target match
            inferred_extraction_method = target_matches[0][1]
            logger.info(f'Inferring {inferred_extraction_method} extraction_method because none was passed') 
            return target_map[(target, inferred_extraction_method)], inferred_extraction_method
        else:
            raise RuntimeError(f"Cannot infer extraction_method for {target} extraction_target because it has {len(target_matches)} entries in target_map: {target_matches}")
    

def get_extraction_target_map(target=None, emotion_type=None, bias_type=None):

    if target:
        # Check to make sure the target comes with its required arguments
        if target == 'fairness':
            if bias_type is None:
                raise ValueError('Must specify a bias_type when using fairness target')
            bias_types = ['race', 'gender', 'religion', 'profession']
            if bias_type not in bias_types:
                raise ValueError(f'bias_type must be one of {bias_types}')
        if target == 'emotion':
            if emotion_type is None:
                raise ValueError('Must specify an emotion_type when using emotion target')
            emotion_types = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
            if emotion_type not in emotion_types:
                raise ValueError(f'emotion_type must be one of {emotion_types}')
    
    target_map = {
        ('truth', 'conceptual'): fetch_factual_data_conceptual,
        ('honesty', 'functional'): fetch_factual_data_functional,
        ('morality', 'functional'): fetch_morality_data_functional,
        ('morality', 'conceptual'): fetch_morality_data_conceptual,
        ('emotion', 'conceptual'): fetch_emotion_data_wrapper(emotion_type), 
        ('fairness', 'conceptual'): fetch_fairness_data_conceptual_wrapper(bias_type),
        ('fairness', 'functional'): fetch_fairness_data_functional_wrapper(bias_type),
        ('harmlessness', 'conceptual'): fetch_harmlessness_data_conceptual
    }
    return target_map


def prepare_statement_pairs(
    extraction_target, extraction_method, tokenizer, user_tag, assistant_tag, 
    n_train_pairs=None, n_dev_pairs=None, n_test_pairs=None, **kwargs):
            
    extraction_fn, extraction_method = get_extraction_function(extraction_target, extraction_method, **kwargs)
    result = extraction_fn()
    data = result.get('data')
    prompt_maker = result.get('prompt_maker')
    kwargs = result.get('kwargs', {})    
    
    if extraction_method == 'functional':
        statement_pairs = prepare_functional_pairs(
            data, prompt_maker, tokenizer, user_tag, assistant_tag, **kwargs)
    elif extraction_method == 'conceptual':
        statement_pairs = prepare_conceptual_pairs(
            data, prompt_maker, tokenizer, user_tag, assistant_tag, **kwargs)

    d = {}
    if n_train_pairs:
        d['train'] = statement_pairs[:n_train_pairs]
        if n_dev_pairs:
            d['dev'] = statement_pairs[n_train_pairs:n_train_pairs+n_dev_pairs]
        if n_test_pairs:
            d['test'] = statement_pairs[n_train_pairs+n_dev_pairs:n_train_pairs+n_dev_pairs+n_test_pairs]
    else:
        d['all'] = statement_pairs
    return d
    

def prepare_functional_pairs(data, _prompt_maker, tokenizer, user_tag, assistant_tag, n_trim_tokens=5, stop_token=None):
    """
    Pair statements by expanding positive and negative version of a prompt (e.g. tell me a fact about..., 
    tell me a lie about...), one token a token at a time.
    """
    statement_pairs = []
    statements = data['statement'].values.tolist()
    for statement in statements:
        if stop_token:
            statement = statement.split(stop_token)[0]    
        tokens = tokenizer.tokenize(statement)
        for idx in range(1, len(tokens)-n_trim_tokens):
            substatement = tokenizer.convert_tokens_to_string(tokens[:idx])
            positive_statement = _prompt_maker(substatement, True, user_tag, assistant_tag)
            negative_statement = _prompt_maker(substatement, False, user_tag, assistant_tag)
            statement_pairs.append([positive_statement, negative_statement])
    statement_pairs = np.array(statement_pairs)
    return statement_pairs


def prepare_conceptual_pairs(data, _prompt_maker, tokenizer, user_tag, assistant_tag, shuffle=True):
    """
    Pair statements that contain concept with statements that are missing the concept.
    """
    statement_pairs = []
    contain_statements = data.loc[data['label'] == 1]['statement'].values.tolist()
    missing_statements = data.loc[data['label'] == 0]['statement'].values.tolist()
    if shuffle:
        random.shuffle(missing_statements)

    n_shared = min(len(contain_statements), len(missing_statements))
    for i in range(n_shared):
        contain_statement = _prompt_maker(contain_statements[i], user_tag, assistant_tag)
        missing_statement = _prompt_maker(missing_statements[i], user_tag, assistant_tag)
        statement_pairs.append([contain_statement, missing_statement])
    statement_pairs = np.array(statement_pairs)
    return statement_pairs


def get_activations_for_paired_statements(statement_pairs, model, tokenizer, batch_size, device, read_token=-1):
    layer_to_act_pairs = defaultdict(list)
    for i in range(0, len(statement_pairs), batch_size):
        pairs = statement_pairs[i:i+batch_size]
        statements = pairs.reshape(-1)
        model_inputs = tokenizer(list(statements), padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            hiddens = model(**model_inputs, output_hidden_states=True)
        for layer in range(model.config.num_hidden_layers):
            act_pairs = hiddens['hidden_states'][layer+1][:, read_token, :].view(len(pairs), 2, -1)
            layer_to_act_pairs[layer].extend(act_pairs)
    
    for key in layer_to_act_pairs:
        layer_to_act_pairs[key] = torch.stack(layer_to_act_pairs[key])

    return layer_to_act_pairs  


def get_directions(train_acts, device, probe_type):
    if probe_type == 'pca':
        return probe_pca(train_acts, device)
    elif probe_type == 'logreg':
        return probe_logreg(train_acts, device)
    elif probe_type == 'massmean':
        return probe_massmean(train_acts, device)
    else:
        probe_types = ['pca', 'logreg', 'massmean']
        raise ValueError(f'probe_type must be one of {probe_types} but is {probe_type}.')

        
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
    # return accs