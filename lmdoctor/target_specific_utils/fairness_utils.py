import pandas as pd
import importlib
import json
import numpy as np

def fetch_fairness_data_conceptual():
    def _prompt_maker(stimulus, user_tag, assistant_tag):
        prompt = f"{user_tag} Consider the bias in the following scenario:\nScenario: {stimulus}. The probability of the scenario being biased is {assistant_tag}"
        return prompt

    data = fetch_fairness_data()
    return data, _prompt_maker, False

def fetch_fairness_data():
    with importlib.resources.open_text('lmdoctor.data.fairness', "dev.json") as data_path:
        data = json.load(data_path)

    intrasentence_data = data['data']['intrasentence']
    pairs = []
    for datum in intrasentence_data:
        if datum['bias_type'] != 'race':
            continue
        sentences = datum['sentences']
        for i in range(3):
            if sentences[i]['gold_label'] == 'stereotype':
                stereo_sentence = sentences[i]['sentence']
            elif sentences[i]['gold_label'] == 'anti-stereotype':
                anti_stereo_sentence = sentences[i]['sentence']
        pair = [stereo_sentence, anti_stereo_sentence]
        pairs.append(pair)

    data = pd.DataFrame({'statement': np.array(pairs).reshape(-1)}) # rearrange to be single list so that it works downstream 
    data['label'] = [0, 1]*(len(pairs))
    return data
