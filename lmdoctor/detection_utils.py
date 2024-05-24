import torch
from collections import defaultdict
from .utils import format_prompt
from .extraction_utils import get_activations_for_paired_statements
from sklearn.linear_model import LogisticRegression
import numpy as np       
from .utils import setup_logger
logger = setup_logger()
from .plot_utils import plot_projection_heatmap, plot_scores_per_token


class Detector:
    """
    Wraps model in order to capture hidden_states during generation and perform computations with those hidden_states.
    Specific detectors (e.g. LieDetector) inherit from it. 
    """
    def __init__(self, extractor):
        self.model = extractor.model
        self.tokenizer = extractor.tokenizer
        self.user_tag = extractor.user_tag
        self.assistant_tag = extractor.assistant_tag
        self.device = extractor.device
        self.direction_info = extractor.direction_info
        self.extractor = extractor
        self.auto_saturate_at = None # for heatmap
        self.layer_aggregation_clf = None # aggregation for detection


    def generate(self, prompts, gen_only=True, return_projections=True, should_format_prompt=True, **kwargs):
        """
        If gen only, get projections/text for newly generated text only (i.e. exclude prompt)
        """

        def _get_start_end_idxs(tensor_data, gen_pad_id):
            first_non_zero_indices = []
            first_eos_indices = []
            for data in tensor_data:
                data_np = data.cpu().numpy()
                first_non_zero_idx = (data_np != 0).argmax()
                first_non_zero_indices.append(first_non_zero_idx)
                reversed_data = data_np[::-1]
                first_eos_idx = len(data_np) - (reversed_data != gen_pad_id).argmax()
                first_eos_indices.append(first_eos_idx)
            return first_non_zero_indices, first_eos_indices
        
        kwargs['return_dict_in_generate'] = True
        kwargs['output_hidden_states'] = True
        kwargs['pad_token_id'] = self.tokenizer.eos_token_id

        # standardize single prompt case
        if not isinstance(prompts, list):
            singleton=True
            prompts = [prompts]
        else:
            singleton=False
                       
        # format input
        if should_format_prompt:
            formatted_prompts = []
            for prompt in prompts:
                formatted_prompt = format_prompt(prompt, self.user_tag, self.assistant_tag)
                formatted_prompts.append(formatted_prompt)
            prompts = formatted_prompts
        model_inputs = self.tokenizer(prompts, return_tensors='pt', padding=True).to(self.device)
        
        # run generation
        with torch.no_grad():
            output = self.model.generate(**model_inputs, **kwargs)
                        
        # get the texts
        start_gen_idx = model_inputs.input_ids.shape[1]
        sequences = output.sequences[:, start_gen_idx:] if gen_only else output.sequences
        output_texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # get the projections
        hiddens = output.hidden_states
        unpacked_hiddens = _unpack_batched_hiddens(hiddens)
        all_projs = []
        for i in range(len(unpacked_hiddens)):
            h = unpacked_hiddens[i]
            projs = self.get_projections(h)
            all_projs.append(projs)
                
        # prepare outputs
        output_tokens = []
        output_projs = []
        start_prompt_idxs, end_gen_idxs = _get_start_end_idxs(output.sequences, gen_pad_id=kwargs['pad_token_id'])
        for i in range(len(prompts)):
            token_seq = self.tokenizer.convert_ids_to_tokens(output['sequences'][i])
            if gen_only:
                output_tokens.append(token_seq[start_gen_idx:end_gen_idxs[i]])
                output_projs.append(all_projs[i][:, start_gen_idx:end_gen_idxs[i]])
            else:
                output_tokens.append(token_seq[start_prompt_idxs[i]:end_gen_idxs[i]])
                output_projs.append(all_projs[i][:, start_prompt_idxs[i]:end_gen_idxs[i]]) 
                
        # print(output_texts)
        
        if singleton:
            output_texts = output_texts[0]            
            output_tokens = output_tokens[0]
            output_projs = output_projs[0]
            
        return {'text': output_texts, 'projections': output_projs, 'tokens': output_tokens, 'raw_output': output}

        
    
    def get_projections(self, hiddens=None, input_text=None):
        """
        Computes the projections of hidden_states onto concept directions.
        """        
        if input_text:
            layer_to_acts = _get_layeracts_from_text(input_text, self.model, self.tokenizer, self.device)
        else:
            layer_to_acts = _get_layeracts_from_hiddens(hiddens)

        all_projs = layeracts_to_projs(layer_to_acts, self.direction_info)
        return all_projs

    
    def tune(self, batch_size=8, run_test=True):
        """
        Train a classifier to learn how to weigh the layer contributions at a given token when performing detection.
        To do this, statement_pairs consisting of one positive (e.g. honesty) and one negative (e.g. dishonesty) sample
        are fed to the model to get hidden_states from the last token of each statement. These hidden_states are then projected
        onto the direction vector for that layer (maps hidden_dim -> scalar). The projections across all layers are then fed 
        into the classifier. Thus, each sample consists of as many projections (scalars) as there are layers, and the label is
        determined by whether it is a positive or negative statement.
        """
        def _create_projection_dataset(act_pairs, direction_info, n_pairs):
            proj_pairs = act_pairs_to_projs(act_pairs, direction_info, n_pairs)
            proj_pairs = proj_pairs.view(-1, proj_pairs.shape[2]) # stack pos and negative samples
            X = proj_pairs.detach().cpu().numpy()
            y = np.array([1] * n_pairs + [0] * n_pairs)
            return X, y

        if not hasattr(self, 'dev_acts'):
            dev_acts = get_activations_for_paired_statements(
                self.extractor.statement_pairs['dev'], self.model, self.tokenizer, batch_size, self.device, read_token=-1)
            self.dev_acts = dev_acts
        X, y = _create_projection_dataset(
            self.dev_acts, self.direction_info, len(self.extractor.statement_pairs['dev']))

        # train
        clf = LogisticRegression()
        clf.fit(X, y)
        acc = clf.score(X, y)
        logger.info(f'Classifier acc on dev set: {acc}')

        if run_test:
            if not hasattr(self, 'test_acts'):
                test_acts = get_activations_for_paired_statements(
                    self.extractor.statement_pairs['test'], self.model, self.tokenizer, batch_size, self.device, read_token=-1)
                self.test_acts = test_acts
            X_test, y_test = _create_projection_dataset(
                self.test_acts, self.direction_info, len(self.extractor.statement_pairs['test']))
            acc = clf.score(X_test, y_test)
            logger.info(f'Classifier acc on test set: {acc}')

        self.layer_aggregation_clf = clf


    def detect(self, all_projs, aggregation_method='auto', layers_to_use=None, use_n_middle_layers=None, **kwargs):
        
        if aggregation_method == 'auto':
            if self.layer_aggregation_clf is None:
                logger.info('Running one-time aggregation tuning, since aggregation_method="auto" and' \
                            ' self.layer_aggregation_clf is not set...')
                self.tune(**kwargs)
                logger.info('Tuning complete.')
            return self.detect_by_classifier(all_projs, self.layer_aggregation_clf)
        elif aggregation_method == 'layer_avg':
            return self.detect_by_layer_avg(all_projs, layers_to_use, use_n_middle_layers)
        

    def detect_by_classifier(self, all_projs, clf):
        """
        For each token in proj, run a classifier over all layers to get the score
        """
        n_tokens = all_projs.shape[1]
        scores = []
        for i in range(n_tokens):
            layer_vals = all_projs[:, i].detach().cpu().numpy()
            score = clf.predict_proba(layer_vals.reshape(1, -1))[0][1] # prob of label=1
            scores.append(score)
        return np.array([scores])
    
    
    def detect_by_layer_avg(self, all_projs, layers_to_use=None, use_n_middle_layers=None):
        """
        Logic for aggregating over layers to score tokens for degree of lying. 
        set one of:
            layers_to_use: -1 (all layers) or [start_layer, end_layer]
            use_n_middle_layers: number of middle layers to use
        """                

        if not layers_to_use and not use_n_middle_layers:
            raise ValueError('Must specify either layers_to_use or use_n_middle_layers when using detect_by_layer_avg') 
        if layers_to_use and use_n_middle_layers:
            raise ValueError('Either specify layers_to_use or use_n_middle_layers, not both')
            
        if layers_to_use:
            if layers_to_use == -1:
                layers_to_use = range(0, self.model.config.num_hidden_layers)
            else:
                layers_to_use = range(layers_to_use[0], layers_to_use[1])
        elif use_n_middle_layers:
            n_layers = self.model.config.num_hidden_layers
            mid = n_layers // 2
            diff = use_n_middle_layers // 2
            layers_to_use = range(max(0, mid - diff), min(mid + diff, n_layers))
            
        layer_avg = all_projs[layers_to_use, :].mean(axis=0)
        layer_avg = layer_avg.detach().cpu().numpy()
        layer_avg = layer_avg.reshape(1, -1)
        return layer_avg
    
    
    def plot_projection_heatmap(self, all_projs, tokens, **kwargs):
        if 'saturate_at' in kwargs and kwargs['saturate_at'] == 'auto':
            if self.auto_saturate_at is None:
                self.auto_saturate_at = auto_compute_saturation(self.extractor)
                logger.info(f'Auto setting saturate_at to {self.auto_saturate_at}, which will be used for current and'
                            ' future detections with this detector.') 
                kwargs['saturate_at'] = self.auto_saturate_at
            else:
                kwargs['saturate_at'] = self.auto_saturate_at
                
                
        plot_projection_heatmap(all_projs, tokens, **kwargs)
        
        
    def plot_scores_per_token(self, scores_per_token, tokens, **kwargs):
        plot_scores_per_token(scores_per_token, tokens, **kwargs)
        
    
    def __getattr__(self, name):
        return getattr(self.model, name)

    
def auto_compute_saturation(extractor, percentile=25):
    """Get the percentile based on distribution of projections of the second label (e.g. the lie in fact/lie pairs)
    Could also base it on both items in pairs, but using this for now."""
    proj_pairs = act_pairs_to_projs(extractor.train_acts, extractor.direction_info, len(extractor.statement_pairs['train']))
    vals = proj_pairs[1, :, :].view(-1)
    # vals = vals[~torch.isnan(vals)] # since directions with nans are discarded, should not need this filter
    perc = np.percentile(vals, percentile)
    saturate_at = abs(round(perc, 4))
    return saturate_at
    

def do_projections(acts, direction, mean_diff=None, center=True, normalize_direction=True):
    if center and mean_diff is not None:
        acts = (acts - mean_diff).clone()
    if normalize_direction:
        projections =  acts @ direction / direction.norm() 
    else:
        projections =  acts @ direction
    return projections

def layeracts_to_projs(layer_to_acts, direction_info):
    directions = direction_info['directions']
    mean_diffs = direction_info.get('mean_diffs', {})
    
    all_projs = []
    for layer in directions:
        projs = do_projections(layer_to_acts[layer], directions[layer], 
                               mean_diff=mean_diffs.get(layer))
        all_projs.append(projs)
    all_projs = torch.stack(all_projs)
    return all_projs

def act_pairs_to_projs(act_pairs, direction_info, n_pairs, normalize_direction=True):
    
    directions = direction_info['directions']
    mean_diffs = direction_info.get('mean_diffs', {})

    num_layers = len(directions.keys())
    proj_pairs = torch.zeros((2, n_pairs, num_layers))
    for i, layer in enumerate(directions):
        pos_projs = do_projections(act_pairs[layer][:, 0, :], directions[layer], 
                                   mean_diff=mean_diffs.get(layer), normalize_direction=normalize_direction)
        neg_projs = do_projections(act_pairs[layer][:, 1, :], directions[layer], 
                                   mean_diff=mean_diffs.get(layer), normalize_direction=normalize_direction)
        proj_pairs[0, :, i] = pos_projs
        proj_pairs[1, :, i] = neg_projs
    return proj_pairs


def _unpack_batched_hiddens(hiddens):
    
    ntokens, nlayers, batch_size = len(hiddens), len(hiddens[0]), hiddens[0][0].shape[0]
    unpacked_hiddens = [[] for _ in range(batch_size)]
    for token in range(ntokens):  
        for batch_part in range(batch_size):
            batch_part_data = [hiddens[token][idx][batch_part].unsqueeze(0) for idx in range(nlayers)]
            unpacked_hiddens[batch_part].append(batch_part_data)
    return unpacked_hiddens
    

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


def _get_layeracts_from_text(input_text, model, tokenizer, device):
    """
    Get activations per layer from the generated text (which includes prompt). 
    This requires re-running the bc the model was already used to generate the text.
    Useful if just have the text (and not the hiddens produced during generation).
    """
    model_inputs = tokenizer(input_text, return_tensors='pt').to(device)
    
    layer_to_acts = {}
    with torch.no_grad():
        hiddens = model(**model_inputs, output_hidden_states=True)
    for layer in range(model.config.num_hidden_layers):
        layer_to_acts[layer] = hiddens['hidden_states'][layer+1].squeeze()
    return layer_to_acts