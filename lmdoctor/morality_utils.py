import importlib
import pandas as pd

def fetch_morality_data():
    
    def _prompt_maker(stimulus, morality, user_tag, assistant_tag):
        prompt = f"{user_tag} Pretend you're a highly {'moral' if morality else 'immoral'}"\
        f" person taking actions. {assistant_tag} {stimulus}"
        return prompt
        
    with importlib.resources.path('lmdoctor.data.ethics.commonsense', 'cm_train.csv') as data_path:
        data = pd.read_csv(data_path)

    data = data[['input', 'label']]
    data = data.rename(columns={'input': 'statement'})
    
    return data, _prompt_maker