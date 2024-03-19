import pandas as pd
import importlib

def fetch_harmlessness_data_conceptual():
    def _prompt_maker(stimulus, user_tag, assistant_tag):
        prompt = f"{user_tag}{stimulus}{assistant_tag}"
        return prompt

    data = fetch_harmlessness_data()
    return {'data': data, 'prompt_maker': _prompt_maker}


def fetch_harmlessness_data():
    with importlib.resources.path('lmdoctor.data.harmlessness', 'instructions.csv') as data_path:
        data = pd.read_csv(data_path)
    data = data.rename(columns={'instruction': 'statement'})
    return data