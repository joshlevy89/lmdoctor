import json
import pandas as pd
import importlib
import numpy as np

def fetch_all_emotion_data():
    emotion_datasets = [] 
    for emotion in ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']:
        with importlib.resources.open_text('lmdoctor.data.emotions', f'{emotion}.json') as data_path:
            scenarios = json.load(data_path)
            data = pd.DataFrame({'statement': scenarios})
            data['emotion'] = emotion
            emotion_datasets.append(data)

    data = pd.concat(emotion_datasets, ignore_index=True)
    return data
    

def fetch_emotion_data(emotion_name):
    
    def emotion_wrapper():
        
        def _prompt_maker(stimulus, user_tag, assistant_tag):
            prompt = f"{user_tag} Consider the emotion in the following scenario:\nScenario: {stimulus} {assistant_tag}"
            return prompt

        data = fetch_all_emotion_data()
        data['has_concept'] = np.where(data['emotion'] == emotion_name, 1, 0)
        return data, _prompt_maker

    return emotion_wrapper

