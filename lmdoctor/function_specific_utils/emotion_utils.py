import json
import pandas as pd
import importlib

def fetch_emotion_data(emotion_name):
    
    def emotion_wrapper():
        def _prompt_maker(stimulus, emotion, user_tag, assistant_tag):
            prompt = f"{user_tag} As a person {f'filled with {emotion_name}' if emotion else f'without any {emotion_name}'}"\
            f" how would you react in this scenario: {stimulus} {assistant_tag}"
            return prompt
            
        with importlib.resources.open_text('lmdoctor.data.emotions', 'anger.json') as data_path:
            scenarios = json.load(data_path)
    
        data = pd.DataFrame({'statement': scenarios})
        
        return data, _prompt_maker

    return emotion_wrapper

