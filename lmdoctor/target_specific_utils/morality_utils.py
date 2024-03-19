import pandas as pd
import importlib

def fetch_morality_data_functional():
    def _prompt_maker(stimulus, morality, user_tag, assistant_tag):
        prompt = f"{user_tag}Pretend you're a highly {'moral' if morality else 'immoral'}"\
        f" person taking actions.{assistant_tag} {stimulus}"
        return prompt
        
    data = fetch_morality_data()
    return {'data': data, 'prompt_maker': _prompt_maker}


def fetch_morality_data_conceptual():
    def _prompt_maker(stimulus, user_tag, assistant_tag):
        prompt = f"{user_tag}Consider the acceptability of the action in the following scenario:\nScenario: {stimulus}{assistant_tag}"
        return prompt

    data = fetch_morality_data()
    return {'data': data, 'prompt_maker': _prompt_maker}


def fetch_morality_data():
    with importlib.resources.path('lmdoctor.data.ethics.commonsense', 'cm_train.csv') as data_path:
        data = pd.read_csv(data_path)
    data = data[['input', 'label', 'is_short']]
    data['label'] = 1 - data['label'] # flip 0s and 1s so that morality=1
    data = data.rename(columns={'input': 'statement'})
    data = data.loc[data['is_short']]
    return data