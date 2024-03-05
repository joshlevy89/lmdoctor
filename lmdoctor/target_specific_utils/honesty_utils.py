import pandas as pd
import importlib
    
def fetch_honesty_data():
    
    def _prompt_maker(stimulus, honesty, user_tag, assistant_tag):
        prompt = f"{user_tag} Pretend you're an {'honest' if honesty else 'untruthful'}"\
        f" person making statements about the world. {assistant_tag} {stimulus}"
        return prompt
        
    with importlib.resources.path('lmdoctor.data.facts', 'facts_true_false.csv') as data_path:
        data = pd.read_csv(data_path)
    
    return data, _prompt_maker

