import pandas as pd
import importlib
    
def fetch_factual_data():
        
    with importlib.resources.path('lmdoctor.data.facts', 'facts_true_false.csv') as data_path:
        data = pd.read_csv(data_path)
    
    return data

def fetch_factual_data_functional():
    def _prompt_maker(stimulus, honesty, user_tag, assistant_tag):
        prompt = f"{user_tag}Pretend you're an {'honest' if honesty else 'untruthful'}"\
        f" person making statements about the world.{assistant_tag} {stimulus}"
        return prompt

    data = fetch_factual_data()
    return {'data': data, 'prompt_maker': _prompt_maker}


def fetch_factual_data_conceptual():

    def _prompt_maker(stimulus, user_tag, assistant_tag):
        prompt = f"{user_tag}Consider the amount of truthfulness in the following statement:\nStatement: {stimulus} The amount of truthfulness in the statement is{assistant_tag}"
        return prompt
    
    data = fetch_factual_data()
    return {'data': data, 'prompt_maker': _prompt_maker}

