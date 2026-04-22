'''
Description: 
Author: anyiran
Date: 2024-10-27 16:42:03
LastEditors: anyiran
LastEditTime: 2024-11-03 19:47:56
'''
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
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='electronic')
parser.add_argument('--method', type=str, default='random')
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

        if opt.method=="random":
            #train      
            train_id=random.choice(list(train_set.keys()))   
            item_titles_train = train_set[train_id]
            idx_item_titles_train = {}
            for idx, item_title_train in enumerate(item_titles_train.split('|')):
                idx_item = "product" + str(idx+1)
                idx_item_titles_train[idx_item] = item_title_train
            example_bundle=session_bundles[train_id]
            example_items=session_items[train_id].split(',')
            # print(example_bundle)
            # print(type(example_items))
            example_bundle_info={}
            for idx,(bundle_name,bundle_items) in enumerate(example_bundle):
                # for j,ei in enumerate(example_items):
                # print(type(bundle_items))
                #session_items和set中的产品顺序是一样的，所以可以直接使用example_items.index(bi)去找bi是item(/desccrition)的索引,
                # item的titles其实就是desccription
                items=[]
                for bi in bundle_items.split(','):
                    idx_item = "product" + str(example_items.index(bi)+1)
                    items.append(idx_item)
                # 对 items 列表进行排序
                items.sort(key=lambda x: int(x.replace('product', '')))
                # 打印结果
                # print("排序后的 items:", items)
                idx_bundle='Bundle' + str(idx+1) + ': '
                example_bundle_info[idx_bundle]=items

            prompt = prompt_generator.get_few_shot_random_prompts(str(idx_item_titles),str(idx_item_titles_train),str(example_bundle_info))
        if opt.method=="fix":
            random.seed(42)
            train_id=random.choice(list(train_set.keys()))   
            item_titles_train = train_set[train_id]
            idx_item_titles_train = {}
            for idx, item_title_train in enumerate(item_titles_train.split('|')):
                idx_item = "product" + str(idx+1)
                idx_item_titles_train[idx_item] = item_title_train
            example_bundle=session_bundles[train_id]
            example_items=session_items[train_id].split(',')
            # print(example_bundle)
            # print(type(example_items))
            example_bundle_info={}
            for idx,(bundle_name,bundle_items) in enumerate(example_bundle):
                # for j,ei in enumerate(example_items):
                # print(type(bundle_items))
                items=[]
                for bi in bundle_items.split(','):
                    idx_item = "product" + str(example_items.index(bi)+1)
                    items.append(idx_item)
                # 对 items 列表进行排序
                items.sort(key=lambda x: int(x.replace('product', '')))
                # 打印结果
                # print("排序后的 items:", items)
                idx_bundle='Bundle' + str(idx+1) + ': '
                example_bundle_info[idx_bundle]=items

            prompt = prompt_generator.get_few_shot_fix_prompts(str(idx_item_titles),str(idx_item_titles_train),str(example_bundle_info))
        if opt.method=="top":
            topk_session_idx = k_neareast_sessions[test_id][0]  # consider top-1 related session
            # item_titles = train_set[topk_session_idx] 
            train_id=topk_session_idx
            item_titles_train = train_set[train_id]
            idx_item_titles_train = {}
            for idx, item_title_train in enumerate(item_titles_train.split('|')):
                idx_item = "product" + str(idx+1)
                idx_item_titles_train[idx_item] = item_title_train
            example_bundle=session_bundles[train_id]
            example_items=session_items[train_id].split(',')
            # print(example_bundle)
            # print(type(example_items))
            example_bundle_info={}
            for idx,(bundle_name,bundle_items) in enumerate(example_bundle):
                # for j,ei in enumerate(example_items):
                # print(type(bundle_items))
                items=[]
                for bi in bundle_items.split(','):
                    idx_item = "product" + str(example_items.index(bi)+1)
                    items.append(idx_item)
                # 对 items 列表进行排序
                items.sort(key=lambda x: int(x.replace('product', '')))
                # 打印结果
                # print("排序后的 items:", items)
                idx_bundle='Bundle' + str(idx+1) + ': '
                example_bundle_info[idx_bundle]=items

            prompt = prompt_generator.get_few_shot_fix_prompts(str(idx_item_titles),str(idx_item_titles_train),str(example_bundle_info))

        
        
        print(prompt)
        prompt_generated_bundles[test_id] = prompt
    # for test_id, prompt in tqdm(prompt_generated_bundles.items()):
        message = [{"role": "user", "content": prompt}]
        zero_shot_res = chat.create_chat_completion(message)
        print(test_id)
        print(zero_shot_res)
    
    # logger.info('Evaluating the generated bundles...')
    # bundle_res = {}

    # for test_id, (topk_session_idx, context) in tqdm(All_context.items()):
        parsered_res = output_parser(zero_shot_res)

        if parsered_res['state_code'] == 404:
            logger.warning(f'Error when evaluating test_id: {test_id}')
            continue 
        bundle_res[test_id] = parsered_res['output']

    # np.save(f'{temp_path}bundle_res.npy', bundle_res, allow_pickle=True)

    # # remove the bundles containing only 1 product
    format_res = process_results(bundle_res)
    
    session_precision, session_recall, coverage = compute(session_items, session_bundles, format_res)
    print(f'Precision: {session_precision}, Recall: {session_recall}, Coverage: {coverage}')