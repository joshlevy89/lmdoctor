import pandas as pd
import importlib
import json
import numpy as np

def fetch_fairness_data_conceptual_wrapper(bias_type):
    
    def fetch_fairness_data_conceptual():
        def _prompt_maker(stimulus, user_tag, assistant_tag):
            prompt = f"{user_tag}Consider the bias in the following scenario:\nScenario: {stimulus}. The probability of the scenario being biased is{assistant_tag}"
            return prompt
    
        pairs, _ = fetch_fairness_data(bias_type)
        data = pd.DataFrame({'statement': np.array(pairs).reshape(-1)}) # rearrange to be single list so that it works downstream 
        data['label'] = [0, 1]*(len(pairs))
        
        return {'data': data, 'prompt_maker': _prompt_maker, 'kwargs': {'shuffle': False}}
        
    return fetch_fairness_data_conceptual

def fetch_fairness_data_functional_wrapper(bias_type):

    def fetch_fairness_data_functional():
        def _prompt_maker(stimulus, fairness, user_tag, assistant_tag):
            prompt = f"{user_tag}Pretend you're a highly {'unbiased' if fairness else 'biased'}"\
            f" person making statements.{assistant_tag} {stimulus}"
            return prompt
            
        _, contexts = fetch_fairness_data(bias_type)
        data = pd.DataFrame({'statement': contexts})
        return {'data': data, 'prompt_maker': _prompt_maker, 'kwargs': {'stop_token': 'BLANK', 'n_trim_tokens': 0}}

        
    return fetch_fairness_data_functional

def fetch_fairness_data(bias_type):

    with importlib.resources.open_text('lmdoctor.data.fairness', "dev.json") as data_path:
        data = json.load(data_path)

    intrasentence_data = data['data']['intrasentence']
    pairs = []
    contexts = []
    for datum in intrasentence_data:
        if datum['bias_type'] != bias_type:
            continue
            
        context = datum['context']
        contexts.append(context)
        
        sentences = datum['sentences']
        for i in range(3):
            if sentences[i]['gold_label'] == 'stereotype':
                stereo_sentence = sentences[i]['sentence']
            elif sentences[i]['gold_label'] == 'anti-stereotype':
                anti_stereo_sentence = sentences[i]['sentence']
            elif sentences[i]['gold_label'] == 'unrelated':
                unrelated_sentence = sentences[i]['sentence']
        pair = [stereo_sentence, anti_stereo_sentence]
        pairs.append(pair)

    return pairs, contexts



