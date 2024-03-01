import json
import pandas as pd
import importlib
    
def fetch_anger_data():
    
    def _prompt_maker(stimulus, angry, user_tag, assistant_tag):
        prompt = f"{user_tag} As a person {'filled with anger' if angry else 'without any anger'}"\
        f" how would you react in this scenario: {stimulus} {assistant_tag}"
        return prompt
        
    with importlib.resources.open_text('lmdoctor.data.emotions', 'anger.json') as data_path:
        scenarios = json.load(data_path)

    data = pd.DataFrame({'statement': scenarios})
    
    return data, _prompt_maker


def fetch_happiness_data():
    
    def _prompt_maker(stimulus, happy, user_tag, assistant_tag):
        prompt = f"{user_tag} As a person {'filled with happiness' if happy else 'without any happiness'}"\
        f" how would you react in this scenario: {stimulus} {assistant_tag}"
        return prompt
        
    with importlib.resources.open_text('lmdoctor.data.emotions', 'anger.json') as data_path:
        scenarios = json.load(data_path)

    data = pd.DataFrame({'statement': scenarios})
    
    return data, _prompt_maker

