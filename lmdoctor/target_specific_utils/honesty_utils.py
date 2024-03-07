import pandas as pd
import importlib
    
def fetch_factual_data():
        
    with importlib.resources.path('lmdoctor.data.facts', 'facts_true_false.csv') as data_path:
        data = pd.read_csv(data_path)
    
    return data

def fetch_factual_data_functional():
    def _prompt_maker(stimulus, honesty, user_tag, assistant_tag):
        prompt = f"{user_tag} Pretend you're an {'honest' if honesty else 'untruthful'}"\
        f" person making statements about the world. {assistant_tag} {stimulus}"
        return prompt

    data = fetch_factual_data()
    return {'data': data, 'prompt_maker': _prompt_maker}


def fetch_factual_data_conceptual():

    def _prompt_maker(stimulus, user_tag, assistant_tag):
        prompt = f"{user_tag} Consider the amount of truthfulness in the following statement:\nStatement: {stimulus} The amount of truthfulness in the statement is {assistant_tag}"
        return prompt
    
    data = fetch_factual_data()
    return {'data': data, 'prompt_maker': _prompt_maker}

def fetch_factual_data_custom_intentional(tokenizer, user_tag, assistant_tag):
    import numpy as np
    import random
    def _prompt_maker(stimulus, honesty, user_tag, assistant_tag):
        # if honesty is None:
        #     prompt = f"{user_tag} Directly copy the following statement.\nStatement: {stimulus} {assistant_tag}"
        # else:
        #     prompt = f"{user_tag} Pretend you're an {'honest' if honesty else 'untruthful'}"\
        #     f" person making statements about the world. {assistant_tag} {stimulus}"

        if honesty == 'repeat':
            prompt = f"{user_tag} Repeat this lie about the world. {assistant_tag} {stimulus}"
        elif honesty == 'lie':
            prompt = f"{user_tag} Tell a lie about the world. {assistant_tag} {stimulus}"
        elif honesty == 'example':
            prompt = f"{user_tag} Give an example of a lie someone might tell about the world. {assistant_tag} {stimulus}"
        elif honesty == 'example2':
            prompt = f'{user_tag} Give an example of a lie someone might tell about the world. {assistant_tag} For example, someone might say "{stimulus}'
        elif honesty == 'truth':
            prompt = f"{user_tag} Tell a fact about the world. {assistant_tag} {stimulus}"
            
        return prompt
    
    from lmdoctor.target_specific_utils.honesty_utils import fetch_factual_data
    data = fetch_factual_data()
    # data
    n_trim_tokens = 5
    stop_token = None
    
    statement_pairs = []
    statements = data['statement'].values.tolist()
    for statement in statements:
        if stop_token:
            statement = statement.split(stop_token)[0]    
        tokens = tokenizer.tokenize(statement)
        for idx in range(10, len(tokens)-n_trim_tokens):
            substatement = tokenizer.convert_tokens_to_string(tokens[:idx])
            positive_statement = _prompt_maker(substatement, 'truth', user_tag, assistant_tag)
            negative_statement = _prompt_maker(substatement, 'lie', user_tag, assistant_tag)
            repeat_statement = _prompt_maker(substatement, 'repeat', user_tag, assistant_tag)
            example_statement =  _prompt_maker(substatement, 'example', user_tag, assistant_tag)
            example2_statement =  _prompt_maker(substatement, 'example2', user_tag, assistant_tag)


            # positive_statement = _prompt_maker(substatement, True, user_tag, assistant_tag)
            # negative_statement = _prompt_maker(substatement, False, user_tag, assistant_tag)
            # repeat_statement = _prompt_maker(substatement, None, user_tag, assistant_tag)
            # statement_pairs.append([positive_statement, negative_statement])
            # statement_pairs.append([positive_statement, repeat_statement])
            # if random.randint(0, 1):
            #     statement_pairs.append([positive_statement, negative_statement]) 
            # else:
            #     statement_pairs.append([positive_statement, repeat_statement])
            statement_pairs.append([repeat_statement, negative_statement])
            statement_pairs.append([example_statement, negative_statement])
            statement_pairs.append([example2_statement, negative_statement])
            # statement_pairs.append([positive_statement, negative_statement])
            # statement_pairs.append([positive_statement, negative_statement])
    statement_pairs = np.array(statement_pairs)
    return statement_pairs
    

