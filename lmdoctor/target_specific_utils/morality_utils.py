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


def fetch_morality_data_intentional_wrapper(tokenizer, user_tag, assistant_tag):
    import numpy as np
    def fetch_morality_data_intentional():
        
        immoral_else_str = "{user_tag}What is something immoral someone else might say.{assistant_tag}{stimulus}"
        moral_else_str = "{user_tag}What is something moral someone else might say.{assistant_tag}{stimulus}"
        immoral_me_str = "{user_tag}What is something immoral you have done.{assistant_tag}{stimulus}"
        moral_me_str = "{user_tag}What is something moral you have done.{assistant_tag}{stimulus}"

        data = fetch_morality_data()
    
        n_trim_tokens=5
        stop_token=None
        statement_pairs = []
        statements = data['statement'].values.tolist()
        for statement in statements:
            if stop_token:
                statement = statement.split(stop_token)[0]    
            tokens = tokenizer.tokenize(statement)
            for idx in range(1, len(tokens)-n_trim_tokens):
                substatement = tokenizer.convert_tokens_to_string(tokens[:idx])
                immoral_else = immoral_else_str.format(user_tag=user_tag, assistant_tag=assistant_tag, stimulus=substatement)
                moral_else = moral_else_str.format(user_tag=user_tag, assistant_tag=assistant_tag, stimulus=substatement)
                immoral_me = immoral_me_str.format(user_tag=user_tag, assistant_tag=assistant_tag, stimulus=substatement)
                moral_me = moral_me_str.format(user_tag=user_tag, assistant_tag=assistant_tag, stimulus=substatement)
                statement_pairs.append([immoral_else, moral_else, immoral_me, moral_me])
        statement_pairs = np.array(statement_pairs)
        return {'data': statement_pairs}
    return fetch_morality_data_intentional
    

def fetch_morality_data():
    with importlib.resources.path('lmdoctor.data.ethics.commonsense', 'cm_train.csv') as data_path:
        data = pd.read_csv(data_path)
    data = data[['input', 'label', 'is_short']]
    data['label'] = 1 - data['label'] # flip 0s and 1s so that morality=1
    data = data.rename(columns={'input': 'statement'})
    data = data.loc[data['is_short']]
    return data