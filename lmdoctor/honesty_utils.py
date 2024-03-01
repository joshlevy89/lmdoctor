import pandas as pd
import numpy as np
from .utils import Detector
import importlib


class LieDetector(Detector):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def detect_lie(self, layers_to_use=range(32-25, 32-9)):
        """
        Logic for aggregating over layers to score tokens for degree of lying. 
        """
        layer_avg = self.all_projs[layers_to_use, :].mean(axis=0)
        layer_avg = layer_avg.detach().cpu().numpy()
        layer_avg = layer_avg.reshape(1, -1)
        return layer_avg
    
def fetch_honesty_data():
    
    def _prompt_maker(stimulus, honesty, user_tag, assistant_tag):
        prompt = f"{user_tag} Pretend you're an {'honest' if honesty else 'untruthful'}"\
        f" person making statements about the world. {assistant_tag} {stimulus}"
        return prompt
        
    with importlib.resources.path('lmdoctor.data.facts', 'facts_true_false.csv') as data_path:
        data = pd.read_csv(data_path)
    
    return data, _prompt_maker

