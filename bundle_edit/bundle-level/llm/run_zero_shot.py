'''
Description: 
Author: anyiran
Date: 2024-10-27 16:42:03
LastEditors: anyiran
LastEditTime: 2026-01-14 15:45:47
'''
#暂时按照session进行设计 bundle rec
import numpy as np
import yaml
from utils.ChatAPI import OpenAI, Claude
from utils.logger import Logger
from utils.functions import output_parser, process_results
from utils.metrics import findErrors, compute
from prompt.prompts import PromptGenerator
from tqdm import tqdm
import argparse
import re 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='electronic')
opt = parser.parse_args()

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
if __name__ == '__main__':
     
    logger = Logger(config['log_path'])
    data_path = config['data_path']+opt.dataset+'/'
    temp_path = config['temp_path']+opt.dataset+'/'
    train_set = np.load(f'{data_path}training_set.npy', allow_pickle=True).item()
    test_set = np.load(f'{data_path}test_set.npy', allow_pickle=True).item()
    k_neareast_sessions = np.load(f'{data_path}TopK_related_sessions.npy', allow_pickle=True).item()
    session_items = np.load(f'{data_path}session_items.npy', allow_pickle=True).item()
    session_bundles = np.load(f'{data_path}session_bundles_deduplication.npy', allow_pickle=True).item()
    all_item_titles = np.load(f'{data_path}item_titles.npy', allow_pickle=True).item()
    
    # Create a new OpenAI instance
    chat = OpenAI(config['model'], config['api_key'], config['temperature'])
    # Create a new prompt generator
    prompt_generator = PromptGenerator(session_items, session_bundles)

    # Construct meta info for training sessions
    prompt_generated_bundles = {} 
    bundle_res = {}
    for test_id in test_set.keys():
        # topk_session_idx = k_neareast_sessions[test_id][0]  # consider top-1 related session
        # item_titles = train_set[topk_session_idx]
        item_titles = test_set[test_id]
        idx_item_titles = {}
        for idx, item_title in enumerate(item_titles.split('|')):
            idx_item = "product" + str(idx+1)
            idx_item_titles[idx_item] = item_title
        prompt = prompt_generator.get_zero_shot_prompts(str(idx_item_titles))
        print(prompt)
        prompt_generated_bundles[test_id] = prompt
    # for test_id, prompt in tqdm(prompt_generated_bundles.items()):
        message = [{"role": "user", "content": prompt}]
        # print(message)
        zero_shot_res = chat.create_chat_completion(message)
        # print(test_id)
        # print(zero_shot_res)
    
    # logger.info('Evaluating the generated bundles...')
    # bundle_res = {}

    # for test_id, (topk_session_idx, context) in tqdm(All_context.items()):
        parsered_res = output_parser(zero_shot_res.replace('\n', ''))
        print(parsered_res)
        if parsered_res['state_code'] == 404:
            logger.warning(f'Error when evaluating test_id: {test_id}')
            continue 
        bundle_res[test_id] = parsered_res['output']

    # np.save(f'{temp_path}bundle_res.npy', bundle_res, allow_pickle=True)

    # # remove the bundles containing only 1 product
    format_res = process_results(bundle_res)
    
    session_precision, session_recall, coverage = compute(session_items, session_bundles, format_res)
    print(f'Precision: {session_precision}, Recall: {session_recall}, Coverage: {coverage}')