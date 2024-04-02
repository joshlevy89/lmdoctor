import pandas as pd
import importlib
import numpy as np
import json
    
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


# def fetch_hallucination_data_functional_wrapper(tokenizer, user_tag, assistant_tag):
#     def fetch_hallucination_data_functional():
        
#         def _prompt_maker(question, answer, user_tag, assistant_tag):
#             prompt = f"{user_tag}{question}{assistant_tag} {answer}"
#             return prompt
    
#         def _make_substatements(tokenizer, question, answer, user_tag, assistant_tag):
#             statements = []
#             tokens = tokenizer.tokenize(answer)
#             for idx in range(1, len(tokens)):
#                 subanswer = tokenizer.convert_tokens_to_string(tokens[:idx])
#                 statement = _prompt_maker(question, subanswer, user_tag, assistant_tag)
#                 statements.append(statement)
#             return statements
            
        
#         data = fetch_hallucination_data()
    
#         statement_pairs = []
#         for _, row in data.iterrows():
#             factual_statements = _make_substatements(
#                 tokenizer, row['Factual Question'], row['Factual Answer'], user_tag, assistant_tag)
#             hallucination_statements = _make_substatements(
#                 tokenizer, row['Hallucination Question'], row['Hallucination Answer'], user_tag, assistant_tag)
#             these_pairs = list(zip(factual_statements, hallucination_statements))
#             statement_pairs.extend(these_pairs)
#         statement_pairs = np.array(statement_pairs)
#         return {'statement_pairs': statement_pairs}
        
#     return fetch_hallucination_data_functional


# def fetch_hallucination_data_functional_wrapper(tokenizer, user_tag, assistant_tag):
#     def fetch_hallucination_data_functional():
        
#         def _prompt_maker(question, answer, user_tag, assistant_tag):
#             prompt = f"{user_tag}{question}{assistant_tag} {answer}"
#             return prompt
    
#         def _make_substatements(tokenizer, question, answer, user_tag, assistant_tag):
#             statements = []
#             tokens = tokenizer.tokenize(answer)
#             for idx in range(1, len(tokens)):
#                 subanswer = tokenizer.convert_tokens_to_string(tokens[:idx])
#                 statement = _prompt_maker(question, subanswer, user_tag, assistant_tag)
#                 statements.append(statement)
#             return statements
            
        
#         # data = fetch_hallucination_data()
#         data = fetch_hallucination_answers()
#         hallucination_list, real_list, hallucination_question_list, real_question_list = parse_hallucination_answers(
#             data, answer_as_fake_to_real_ratio=1)
    
#         statement_pairs = []
#         for i in range(len(hallucination_list)):
#             factual_statements = _make_substatements(
#                 tokenizer, real_question_list[i], real_list[i], user_tag, assistant_tag)
#             hallucination_statements = _make_substatements(
#                 tokenizer, hallucination_question_list[i], hallucination_list[i], user_tag, assistant_tag)
#             these_pairs = list(zip(factual_statements, hallucination_statements))
#             statement_pairs.extend(these_pairs)
#         statement_pairs = np.array(statement_pairs)
#         return {'statement_pairs': statement_pairs}
        
#     return fetch_hallucination_data_functional



# def fetch_hallucination_data_functional_wrapper(tokenizer, user_tag, assistant_tag):
#     def fetch_hallucination_data_functional():
        
#         def _prompt_maker(question, answer, user_tag, assistant_tag):
#             prompt = f"{user_tag}{question}{assistant_tag} {answer}"
#             return prompt
    
#         def _make_substatements(tokenizer, question, answer, user_tag, assistant_tag):
#             statements = []
#             tokens = tokenizer.tokenize(answer)
#             for idx in range(1, len(tokens)):
#                 subanswer = tokenizer.convert_tokens_to_string(tokens[:idx])
#                 statement = _prompt_maker(question, subanswer, user_tag, assistant_tag)
#                 statements.append(statement)
#             return statements
            
        
#         data = fetch_hallucination_data()
    
#         statement_pairs = []
#         for _, row in data.iterrows():
#             factual_statements = _make_substatements(
#                 tokenizer, row['Factual Question'], row['Factual Answer'], user_tag, assistant_tag)
#             hallucination_statements = _make_substatements(
#                 tokenizer, row['Hallucination Question'], row['Hallucination Answer'], user_tag, assistant_tag)
#             these_pairs = list(zip(factual_statements, hallucination_statements))
#             statement_pairs.extend(these_pairs)
#         statement_pairs = np.array(statement_pairs)
#         return {'statement_pairs': statement_pairs}
        
#     return fetch_hallucination_data_functional


def fetch_hallucination_data_functional_wrapper(tokenizer, user_tag, assistant_tag):
    def fetch_hallucination_data_functional():
        
        def _prompt_maker(question, cautious, user_tag, assistant_tag):
            prompt = f"{user_tag}Pretend you're an intellectually {'cautious' if cautious else 'brash'} bot answering questions that you may not know the answer to. Question: {question}{assistant_tag} The"
            return prompt
        
        data = fetch_hallucination_data()
    
        statement_pairs = []
        for _, row in data.iterrows():
            factual_statement = _prompt_maker(row['Hallucination Question'], True, user_tag, assistant_tag)
            hallucination_statement = _prompt_maker(row['Hallucination Question'], False, user_tag, assistant_tag)
            statement_pairs.append([factual_statement, hallucination_statement])
        statement_pairs = np.array(statement_pairs)
        return {'statement_pairs': statement_pairs}
        
    return fetch_hallucination_data_functional


def fetch_hallucination_answers():
    with importlib.resources.path('lmdoctor.data.hallucinations', 'hallucination_answers.json') as data_path:
        with open(data_path, 'r') as file:
            data = json.load(file)

    return data

def parse_hallucination_answers(data, answer_as_fake_to_real_ratio=.5):
    import random 
    seed = 42
    random.seed(seed)

    prompt_pairs = data['respones']
    real_list = []
    hallucination_list = []
    hallucination_question_list = []
    real_question_list = []
    for pair in prompt_pairs:
        if 'made_up_prompt' in pair[0]:
            hallucination_set, real_set = pair[0], pair[1]
        elif 'made_up_prompt' in pair[1]:
            hallucination_set, real_set = pair[1], pair[0]
        else:
            raise Exception('Did not find made_up_prompt')


        hallucination_list.append(hallucination_set['answer_as_real'])
        hallucination_question_list.append(hallucination_set['made_up_prompt'])
        # for the "good", non hallucination, sometimes use the answer to the real question and sometimes 
        # use the answer to the hallucination question that treats it as fake (i.e. denies hallucinating)
        if random.random() < answer_as_fake_to_real_ratio:
            real_list.append(hallucination_set['answer_as_fake'])
            real_question_list.append(hallucination_set['made_up_prompt'])
        else:
            real_list.append(real_set['answer'])
            real_question_list.append(real_set['real_prompt'])
    
    return hallucination_list, real_list, hallucination_question_list, real_question_list

def fetch_hallucination_data_conceptual():

    def _prompt_maker(stimulus, user_tag, assistant_tag):
        # prompt = f"{user_tag}Consider the amount of truthfulness in the following statement:\nStatement: {stimulus} The amount of truthfulness in the statement is{assistant_tag}"
        prompt = f"{user_tag}Consider the amount of truthfulness in the following statement.\nStatement: {stimulus}{assistant_tag} The amount of truthfulness in this statement is"
        return prompt

        
    data = fetch_hallucination_answers()
    hallucination_list, real_list, _, _ = parse_hallucination_answers(data, answer_as_fake_to_real_ratio=.5)

    answers = []    
    for item1, item2 in zip(real_list, hallucination_list):
        answers.append(item1)
        answers.append(item2)
    labels = [1, 0] * len(prompt_pairs)

    data = pd.DataFrame({'statement': answers, 'label': labels})
            
    return {'data': data, 'prompt_maker': _prompt_maker, 'kwargs': {'shuffle': False}}
