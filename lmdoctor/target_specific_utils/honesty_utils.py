import pandas as pd
import importlib
import numpy as np
    
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


def fetch_hallucination_data():
        
    with importlib.resources.path('lmdoctor.data.hallucinations', 'hallucination_prompts.csv') as data_path:
        data = pd.read_csv(data_path)
    
    return data


def fetch_hallucination_data_functional_wrapper(tokenizer, user_tag, assistant_tag):
    def fetch_hallucination_data_functional():
        
        def _prompt_maker(question, answer, user_tag, assistant_tag):
            prompt = f"{user_tag}{question}{assistant_tag} {answer}"
            return prompt
    
        def _make_substatements(tokenizer, question, answer, user_tag, assistant_tag):
            statements = []
            tokens = tokenizer.tokenize(answer)
            for idx in range(1, len(tokens)):
                subanswer = tokenizer.convert_tokens_to_string(tokens[:idx])
                statement = _prompt_maker(question, subanswer, user_tag, assistant_tag)
                statements.append(statement)
            return statements
            
        
        data = fetch_hallucination_data()
    
        statement_pairs = []
        for _, row in data.iterrows():
            factual_statements = _make_substatements(
                tokenizer, row['Factual Question'], row['Factual Answer'], user_tag, assistant_tag)
            hallucination_statements = _make_substatements(
                tokenizer, row['Hallucination Question'], row['Hallucination Answer'], user_tag, assistant_tag)
            these_pairs = list(zip(factual_statements, hallucination_statements))
            statement_pairs.extend(these_pairs)
        statement_pairs = np.array(statement_pairs)
        return {'statement_pairs': statement_pairs}
        
    return fetch_hallucination_data_functional
