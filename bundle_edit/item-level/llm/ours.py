'''
Description: Process bundle operation test data using LLM with YAML configuration
Author: anyiran
Date: 2024-03-21
'''
import pandas as pd
import openai
import time
import math
import re
import os
import argparse
import yaml
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer, util

# GPT-3.5 API Configuration
openai.api_key = ""
openai.api_base = ""

# Initialize SentenceTransformer for NBR
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# NBR Module: Neighbor Bundle Retrieval
def build_session_repr(bundle_id, bundle_item_df, item_titles_df):
    """Build session representation for a bundle"""
    item_ids = bundle_item_df[bundle_item_df['bundle ID'] == bundle_id]['item ID'].tolist()
    items_df = pd.DataFrame({'item ID': item_ids})
    items_df = items_df.merge(item_titles_df[['item ID', 'titles']], on='item ID', how='left').dropna()
    session_text = " ; ".join(items_df['titles'].tolist())
    return session_text

def get_neighbor_bundles(target_id, bundle_embeddings, bundle_sessions, top_k=3):
    """Get neighbor bundles using semantic similarity"""
    target_emb = bundle_embeddings[target_id]
    all_ids = list(bundle_embeddings.keys())
    all_embs = torch.stack([bundle_embeddings[bid] for bid in all_ids])
    
    cos_scores = util.cos_sim(target_emb, all_embs)[0]
    top_results = cos_scores.topk(k=top_k+1)  # +1 because includes self
    
    neighbors = []
    for score, idx in zip(top_results[0], top_results[1]):
        nid = all_ids[idx]
        if nid == target_id:
            continue
        neighbors.append((nid, float(score)))
        if len(neighbors) >= top_k:
            break
    return neighbors

def get_bundle_topic(bundle_lines,test_type, max_retries=3):
    """Get bundle topic using GPT"""
    if test_type == 'delete':
        prompt = f"""
Given the following bundle of products:
{bundle_lines}
Please identify what kind of product seems out of place or unrelated to the main theme.
Just describe briefly in one sentence.
"""
    else:
        prompt = f"""
Here is a bundle of products:
{bundle_lines}
Please briefly summarize the overall theme or topic of this bundle in one short sentence.
"""
#     topic_prompt = f"""
# Here is a bundle of products:
# {bundle_lines}

# Please briefly summarize the overall theme or topic of this product bundle in one short sentence.
# Do not recommend products. Just describe the topic.
# """
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.2,
                top_p=0.9,
            )
            topic_text = response["choices"][0]["message"]["content"].strip()
            return topic_text
        except Exception as e:
            print(f"Error getting topic: {e}")
            time.sleep(1.5)
    return "Unknown topic"

def extract_pred_id(response_text):
    """Extract prediction ID from LLM response using multiple patterns"""
    # 1. 匹配 Final Selection: 16. "xxx"
    match = re.search(r"Final Selection:\s*(\d+)\.", response_text)
    if match:
        return match.group(1)
    # 2. 匹配 Step 3: ... is: 16. "xxx"
    match = re.search(r"is:\s*(\d+)\.", response_text)
    if match:
        return match.group(1)
    # 3. 匹配 Item(5, "Dominique Seamless Strapless Bra") 格式
    match = re.search(r"Item\(\s*(\d+)\s*,\s*\"[^\"]+\"\)", response_text)
    if match:
        return match.group(1)
    # 4. 匹配 Item(5, 'Dominique Seamless Strapless Bra') 格式（单引号）
    match = re.search(r"Item\(\s*(\d+)\s*,\s*'[^']+'\)", response_text)
    if match:
        return match.group(1)
    # 5. 兼容原有格式
    match = re.search(r"\[(\d+)\]\s*[-–]", response_text)
    if match: return match.group(1)
    match = re.search(r"\b(\d+)\s*[-–]", response_text)
    if match: return match.group(1)
    match = re.search(r"Item\((\d+),", response_text)
    if match: return match.group(1)
    match = re.search(r"\[Item\((\d+),", response_text)
    if match: return match.group(1)
    match = re.search(r"\[\s*(\d+)\s*,", response_text)
    if match: return match.group(1)
    # 6. 匹配纯数字（作为最后的备选方案）
    match = re.search(r"\b(\d+)\b", response_text)
    if match: return match.group(1)
    return None

def load_ours(config_path='ours.yaml'):
    """
    Load operations configuration from YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary containing operations and prompt template
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file {config_path} not found. Please create the YAML configuration file.")
        return None

def extract_operation_results(response_text, operation_type):
    """
    Extracts operation-specific results from the LLM response.
    
    Args:
        response_text (str): The raw text response from the LLM.
        operation_type (str): The type of operation (add, delete, update)
    
    Returns:
        dict: A dictionary containing operation-specific results:
            - For ADD: {'selected_item': (id, title), 'final_bundle': [(id, title), ...]}
            - For DELETE: {'removed_item': (id, title), 'final_bundle': [(id, title), ...]}
            - For UPDATE: {'removed_item': (id, title), 'selected_item': (id, title), 'final_bundle': [(id, title), ...]}
    """
    # 提取所有Item对象
    items = re.findall(r'Item\((\d+),\s*"([^"]+)"\)', response_text)
    items = [(int(item_id), title) for item_id, title in items]
    
    # 根据操作类型解析结果
    results = {}
    
    # 提取Final bundle
    final_bundle_match = re.search(r'Final bundle:\s*\[(.*?)\]', response_text, re.DOTALL)
    if final_bundle_match:
        final_bundle_items = re.findall(r'Item\((\d+),\s*"([^"]+)"\)', final_bundle_match.group(1))
        results['final_bundle'] = [(int(item_id), title) for item_id, title in final_bundle_items]
    else:
        results['final_bundle'] = []

    if operation_type == 'add':
        # For ADD operations, use enhanced prediction extraction
        pred_id = extract_pred_id(response_text)
        if pred_id:
            # Find the corresponding item from the response text
            selected_match = re.search(r'Selected item:\s*Item\((\d+),\s*"([^"]+)"\)', response_text)
            if selected_match:
                results['selected_item'] = (int(selected_match.group(1)), selected_match.group(2))
            else:
                # Try to find the item by ID in the available items
                pred_id_int = int(pred_id)
                for item_id, title in items:
                    if item_id == pred_id_int:
                        results['selected_item'] = (item_id, title)
                        break
    
    elif operation_type == 'delete':
        # 提取Removed item
        removed_match = re.search(r'Removed item:\s*Item\((\d+),\s*"([^"]+)"\)', response_text)
        if removed_match:
            results['removed_item'] = (int(removed_match.group(1)), removed_match.group(2))
    
    elif operation_type == 'update':
        # 提取Removed item和Selected item
        removed_match = re.search(r'Removed item:\s*Item\((\d+),\s*"([^"]+)"\)', response_text)
        selected_match = re.search(r'Selected item:\s*Item\((\d+),\s*"([^"]+)"\)', response_text)
        if removed_match:
            results['removed_item'] = (int(removed_match.group(1)), removed_match.group(2))
        if selected_match:
            results['selected_item'] = (int(selected_match.group(1)), selected_match.group(2))
    
    return results

def extract_operation(response_text):
    """
    Extracts the operation type (ADD, DELETE, UPDATE) from the LLM response.
    
    Args:
        response_text (str): The raw text response from the LLM.
    
    Returns:
        str: The extracted operation type, or None if not found.
    """
    match = re.search(r'Operation Type: (ADD|DELETE|UPDATE)', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback to check for keywords in the prompt response
    if "UPDATE" in response_text.upper():
        return "UPDATE"
    if "ADD" in response_text.upper():
        return "ADD"
    if "DELETE" in response_text.upper():
        return "DELETE"
        
    return None

def get_bundle_items(bundle_ids, bundle_item_df, item_titles_df):
    """
    Retrieves the item IDs and titles for a given list of bundle IDs.
    
    Args:
        bundle_ids (list): The list of bundle IDs to retrieve items for.
        bundle_item_df (pd.DataFrame): DataFrame loaded from bundle_item.csv.
        item_titles_df (pd.DataFrame): DataFrame loaded from item_titles.csv.
        
    Returns:
        list: A list of tuples (item_id, title) for all items in the bundles.
    """
    if not bundle_ids:
        return []
    bundle_items_ids_df = bundle_item_df[bundle_item_df['bundle ID'].isin(bundle_ids)]
    
    bundle_items_df = bundle_items_ids_df.merge(item_titles_df, on='item ID', how='left').dropna()
    
    return list(zip(bundle_items_df['item ID'], bundle_items_df['titles']))

def build_prompt(config, test_type, bundle_lines, cand_lines, **kwargs):
    """
    Build prompt using configuration from YAML file.
    
    Args:
        config (dict): Configuration dictionary from YAML
        test_type (str): Operation type (add, delete, update)
        bundle_lines (str): Formatted bundle items
        cand_lines (str): Formatted candidate items
        **kwargs: Additional arguments for enhanced prompts (nbr_context, topic, etc.)
        
    Returns:
        str: Complete prompt
    """
    # Check if enhanced version should be used for ADD, DELETE, or UPDATE operations
    if test_type in ['add', 'delete', 'update'] and ('nbr_context' in kwargs or 'topic' in kwargs):
        # Use enhanced prompt template
        prompt_key = f'{test_type}_prompt_enhanced'
        prompt_template = config.get(prompt_key, '')
        
        if not prompt_template:
            raise ValueError(f"Enhanced prompt template '{prompt_key}' not found in configuration.")
        
        # Format the enhanced prompt with additional context
        nbr_context = kwargs.get('nbr_context', '')
        topic = kwargs.get('topic', 'Unknown topic')
        
        prompt = prompt_template.format(
            bundle_lines=bundle_lines,
            cand_lines=cand_lines,
            nbr_context=nbr_context,
            topic=topic,
            semantic_info=topic  # For delete operation, semantic_info is same as topic
        )
        return prompt
    else:
        # Use standard template from YAML config
        prompt_key = f"{test_type}_prompt"
        prompt_template = config.get(prompt_key, '')
        
        if not prompt_template:
            raise ValueError(f"Prompt template not found for operation: {test_type}")
        
        prompt = prompt_template.format(
            bundle_lines=bundle_lines,
            cand_lines=cand_lines
        )
        return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_name', type=str, required=True, help='Dataset name: clothing, electronic, food')
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['add', 'delete', 'update'], help='Operation mode: add, delete, or update')
    parser.add_argument('-c', '--config', type=str, default='../props/ours.yaml', help='Path to operations configuration YAML file')
    parser.add_argument('-k', '--top_k', type=int, default=3, help='Number of similar neighbor bundles to retrieve (default: 3)')  # 新增参数
    args = parser.parse_args()

    # Load operations configuration
    config = load_ours(args.config)
    if config is None:
        return
    print(f"Loaded configuration from: {args.config}")
    print(f"Running in mode: {args.mode}")

    # Construct paths
    dataset_path = f'../../dataset/{args.data_name}/'
    test_data_path = f'../../testdata1/{args.data_name}/'
    bundle_item_path = os.path.join(dataset_path, 'bundle_item.csv').replace('\\', '/')
    item_titles_path = os.path.join(dataset_path, 'item_titles.csv').replace('\\', '/')
    
    # Check if files exist
    if not os.path.exists(bundle_item_path) or not os.path.exists(item_titles_path):
        print("Error: Missing bundle_item.csv or item_titles.csv in the dataset directory.")
        return

    # Read dataframes, specifying correct column names if necessary
    bundle_item_df = pd.read_csv(bundle_item_path)
    item_titles_df = pd.read_csv(item_titles_path)

    # Read test files - only process the specified mode
    test_type = args.mode  # Use the specified mode
    prompts = []
    test_cases = []
    
    # Check if operation prompt is defined in config
    prompt_key = f"{test_type}_prompt"
    if prompt_key not in config:
        print(f"Error: Prompt template '{prompt_key}' not found in configuration.")
        return
        
    test_file = os.path.join(test_data_path, f'{test_type}_test.txt')
    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found.")
        return
        
    print(f"Processing test file: {test_file}")
    test_data = pd.read_csv(test_file, sep='\t', header=None, on_bad_lines='skip')
    
    # For ADD, DELETE, and UPDATE operations, precompute bundle embeddings and sessions for NBR
    bundle_embeddings = {}
    bundle_sessions = {}
    bundle_topics = {}
    
    if test_type in ['add', 'delete', 'update']:
        print(f"Initializing NBR module for {test_type.upper()} operation...")
        # Precompute bundle embeddings for all bundles
        all_bundle_ids = bundle_item_df['bundle ID'].unique()
        for bid in all_bundle_ids:
            session_text = build_session_repr(bid, bundle_item_df, item_titles_df)
            bundle_sessions[bid] = session_text
            bundle_embeddings[bid] = st_model.encode(session_text, convert_to_tensor=True)
        print(f"Precomputed embeddings for {len(bundle_embeddings)} bundles")
    
    for _, row in test_data.iterrows():
        bundle_id = int(row[0])
        all_items_ids = [int(x) for x in str(row[1]).split()]
        
        # 处理候选集，对于delete类型可能为空
        candidates_str = str(row[2]).strip()
        candidates_ids = [int(x) for x in candidates_str.split()] if candidates_str and candidates_str != 'nan' else []
        
        # Get item titles for bundle
        bundle_items_df_temp = pd.DataFrame({'item ID': all_items_ids})
        bundle_items_df_temp = bundle_items_df_temp.merge(item_titles_df[['item ID', 'titles']], on='item ID', how='left').dropna()
        bundle_items = bundle_items_df_temp[['item ID', 'titles']].values.tolist()
        
        # Get item titles for candidates (只在有候选集的情况下处理)
        candidate_items = []
        if candidates_ids:
            candidate_items_df_temp = pd.DataFrame({'item ID': candidates_ids})
            candidate_items_df_temp = candidate_items_df_temp.merge(item_titles_df[['item ID', 'titles']], on='item ID', how='left').dropna()
            candidate_items = candidate_items_df_temp[['item ID', 'titles']].values.tolist()
        
        # 创建ID映射：bundle从0开始，candidate也从0开始
        original_to_new_id = {}
        new_to_original_id = {}
        
        # 为bundle items分配新的连续ID（从0开始）
        for new_id, (original_id, title) in enumerate(bundle_items):
            original_to_new_id[original_id] = new_id
            new_to_original_id[new_id] = original_id
        
        # 为candidate items分配新的连续ID（从0开始）
        for idx, (original_id, title) in enumerate(candidate_items):
            new_id = idx
            original_to_new_id[original_id] = new_id
            new_to_original_id[new_id] = original_id
        
        # Format items for prompt using new IDs
        bundle_lines = ',\n    '.join([f'Item({original_to_new_id[item[0]]}, "{item[1]}")' for item in bundle_items])
        cand_lines = ',\n    '.join([f'Item({original_to_new_id[item[0]]}, "{item[1]}")' for item in candidate_items])
        
        # For ADD, DELETE, and UPDATE operations, enhance with NBR and Topic analysis
        if test_type in ['add', 'delete', 'update']:
            # Get bundle topic
            if bundle_id not in bundle_topics:
                topic_bundle_lines = ',\n    '.join([f'{idx}. "{item[1]}"' for idx, item in enumerate(bundle_items)])
                topic = get_bundle_topic(topic_bundle_lines,test_type)
                bundle_topics[bundle_id] = topic
                print(f"[Bundle {bundle_id}] Topic: {topic}")
            else:
                topic = bundle_topics[bundle_id]
            
            # Get NBR neighbors
            neighbors = get_neighbor_bundles(bundle_id, bundle_embeddings, bundle_sessions, top_k=args.top_k)
            neighbor_texts = []
            for nid, score in neighbors:
                neighbor_texts.append(f"- Neighbor bundle {nid}: {bundle_sessions[nid]}")
            nbr_context = "\n".join(neighbor_texts)
            
            # Build enhanced prompt with NBR and Topic
            prompt = build_prompt(config, test_type, bundle_lines, cand_lines, 
                                nbr_context=nbr_context, topic=topic)
        else:
            # This should not occur with current operation types, but kept for completeness
            prompt = build_prompt(config, test_type, bundle_lines, cand_lines)
        
        prompts.append(prompt)
        test_cases.append({
            'test_type': test_type,
            'bundle_id': bundle_id,
            'all_items_ids': all_items_ids,
            'candidates_ids': candidates_ids,
            'bundle_items': bundle_items,
            'candidate_items': candidate_items,
            'original_to_new_id': original_to_new_id,
            'new_to_original_id': new_to_original_id
        })

    # Evaluation loop
    max_retries = 5
    results = []
    start_time = time.time()
    
    # 指标统计
    total_samples = 0
    total_hits = 0  # 记录hit的数量
    failed_extractions = 0  # 记录提取失败的数量

    for idx, (prompt, test_case) in enumerate(zip(prompts, test_cases)):
        success = False
        for attempt in range(max_retries):
            try:
                print(f"\nEvaluating prompt {idx + 1}/{len(prompts)} - Attempt {attempt + 1}")
                print(f"Test type: {test_case['test_type']}")
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    n=1,
                )
                
                result = response["choices"][0]["message"]["content"].strip()
                operation_results = extract_operation_results(result, test_case['test_type'])
                
                # Validate that required items are extracted based on operation type
                validation_passed = True
                if test_case['test_type'] == 'add':
                    if 'selected_item' not in operation_results:
                        print(f"ADD operation missing required 'selected_item'. Available keys: {list(operation_results.keys())}")
                        validation_passed = False
                elif test_case['test_type'] == 'delete':
                    if 'removed_item' not in operation_results:
                        print(f"DELETE operation missing required 'removed_item'. Available keys: {list(operation_results.keys())}")
                        validation_passed = False
                elif test_case['test_type'] == 'update':
                    missing_items = []
                    if 'removed_item' not in operation_results:
                        missing_items.append('removed_item')
                    if 'selected_item' not in operation_results:
                        missing_items.append('selected_item')
                    
                    if missing_items:
                        print(f"UPDATE operation missing required items: {missing_items}")
                        print(f"Available keys: {list(operation_results.keys())}")
                        validation_passed = False
                
                if not validation_passed:
                    print(f"Attempt {attempt + 1}/{max_retries}: Validation failed, retrying...")
                    print(f"Response snippet: {result[:200]}...")
                    continue  # Retry this attempt
                
                # 将结果中的新ID转换回原始ID
                bundle_items = test_case['bundle_items']
                candidate_items = test_case['candidate_items']
                
                def convert_new_id_to_original(new_id, title):
                    """根据新ID和title找到对应的原始ID"""
                    # 先在bundle中查找
                    if new_id < len(bundle_items):
                        original_id, original_title = bundle_items[new_id]
                        if original_title == title:
                            return original_id
                    
                    # 再在candidate中查找
                    if new_id < len(candidate_items):
                        original_id, original_title = candidate_items[new_id]
                        if original_title == title:
                            return original_id
                    
                    # 如果没找到，尝试通过title在所有items中查找
                    for original_id, original_title in bundle_items + candidate_items:
                        if original_title == title:
                            return original_id
                    
                    return None
                
                # 转换final_bundle中的ID
                refined_bundle_original = []
                for new_id, title in operation_results.get('final_bundle', []):
                    original_id = convert_new_id_to_original(new_id, title)
                    if original_id is not None:
                        refined_bundle_original.append((original_id, title))
                
                # 转换selected_item和removed_item中的ID
                if 'selected_item' in operation_results:
                    new_id, title = operation_results['selected_item']
                    original_id = convert_new_id_to_original(new_id, title)
                    if original_id is not None:
                        operation_results['selected_item'] = (original_id, title)
                
                if 'removed_item' in operation_results:
                    new_id, title = operation_results['removed_item']
                    original_id = convert_new_id_to_original(new_id, title)
                    if original_id is not None:
                        operation_results['removed_item'] = (original_id, title)
                
                refined_bundle = refined_bundle_original
                
                # 提取LLM预测的操作类型
                predicted_operation = extract_operation(result)

                print(f"Prompt:\n{prompt}")
                print("--------------------------------")
                print(f"Response:\n{result}")
                
                if operation_results:  # 如果成功提取到操作结果
                    # 获取ground truth（根据bundle_id的完整bundle）
                    ground_truth_items = get_bundle_items([test_case['bundle_id']], bundle_item_df, item_titles_df)
                    ground_truth_ids = {item[0] for item in ground_truth_items}

                    # 获取当前bundle（作为原始bundle）
                    current_bundle_ids = set(test_case['all_items_ids'])
                    
                    # 获取预测结果
                    predicted_ids = {item[0] for item in refined_bundle}
                    
                    # 根据操作类型计算hit - 参照zeroshot.py的方式
                    is_hit = 0
                    
                    if test_case['test_type'] == 'add':
                        # ADD: GT是应该被添加的item（在ground truth中但不在current bundle中）
                        if 'selected_item' in operation_results:
                            selected_item_id, selected_item_title = operation_results['selected_item']
                            
                            # 正确的GT应该是：ground truth中存在但current bundle中不存在的item
                            ground_truth_ids = {item[0] for item in ground_truth_items}
                            current_bundle_ids = set(test_case['all_items_ids'])
                            positive_items = ground_truth_ids - current_bundle_ids
                            
                            # 如果selected item是正例（应该被添加的item），则hit
                            if selected_item_id in positive_items:
                                is_hit = 1
                                print("✅ Hit@1!")
                            else:
                                print("❌ Missed!")
                                
                    elif test_case['test_type'] == 'delete':
                        # DELETE: GT是应该被删除的item（在current bundle中但不在ground truth中）
                        if 'removed_item' in operation_results:
                            removed_item_title = operation_results['removed_item'][1]
                            # 获取应该被删除的items（current bundle中有但ground truth中没有）
                            ground_truth_titles = {item[1] for item in ground_truth_items}
                            current_bundle_titles = {item[1] for item in bundle_items}
                            items_to_delete = current_bundle_titles - ground_truth_titles
                            
                            if removed_item_title in items_to_delete:
                                is_hit = 1
                                print("✅ Hit@1!")
                            else:
                                print("❌ Missed!")
                                
                    elif test_case['test_type'] == 'update':
                        # UPDATE: 需要同时满足两个条件：
                        # 1. 成功移除了随机添加的item（current bundle中有但ground truth中没有的item）
                        # 2. 选择的item是GT bundle的正例（ground truth中有但current bundle中没有的item）
                        if 'selected_item' in operation_results and 'removed_item' in operation_results:
                            selected_item_id, selected_item_title = operation_results['selected_item']
                            removed_item_id, removed_item_title = operation_results['removed_item']
                            
                            ground_truth_ids = {item[0] for item in ground_truth_items}
                            current_bundle_ids = set(test_case['all_items_ids'])
                            
                            # 应该被移除的items：current bundle中有但ground truth中没有的items（随机添加的item）
                            items_to_remove = current_bundle_ids - ground_truth_ids
                            
                            # 应该被添加的items：ground truth中有但current bundle中没有的items（原始被移除的item）
                            items_to_add = ground_truth_ids - current_bundle_ids
                            
                            # 检查两个条件是否都满足
                            remove_correct = removed_item_id in items_to_remove
                            add_correct = selected_item_id in items_to_add
                            
                            if remove_correct and add_correct:
                                is_hit = 1
                                print("✅ Hit@1! (Both remove and add correct)")
                            else:
                                print(f"❌ Missed! Remove correct: {remove_correct}, Add correct: {add_correct}")
                    
                    # 只保留hit指标的统计
                    total_samples += 1
                    total_hits += is_hit  # 累加hit值
                    
                    results.append({
                        'test_type': test_case['test_type'],
                        'ground_truth_items': ground_truth_items,
                        'current_bundle_items': bundle_items,  # 当前bundle的items
                        'refined_bundle': refined_bundle,
                        'response': result,
                        'bundle_id': test_case['bundle_id'],
                        'is_hit': is_hit,
                        'operation_results': operation_results.copy()  # 保存操作结果
                    })
                    success = True
                    # 成功请求后也稍作延迟，避免过快请求
                    time.sleep(0.5)
                    break

            except Exception as e:
                print(f"Error: {e}")
                # 对于internal_error，等待更长时间
                if "internal_error" in str(e):
                    print(f"API internal error detected, waiting longer before retry...")
                    time.sleep(5.0)  # 等待5秒
                else:
                    time.sleep(1.5)

        if not success:
            failed_extractions += 1
            print(f"Failed to process test case {idx + 1}: Could not extract required items for {test_case['test_type']} operation after {max_retries} attempts")

    end_time = time.time()
    total_minutes = (end_time - start_time) / 60
    save_data_path = f'../result/{args.data_name}'
    os.makedirs(save_data_path, exist_ok=True)
    # Save detailed results
    output_file = os.path.join(save_data_path, f'{args.mode}_ours_{args.top_k}_results.txt')
    
    # Calculate overall metrics
    hit_rate = total_hits / len(prompts) if len(prompts) > 0 else 0  # 计算hit rate


    print(f"\nEvaluation completed for {args.mode.upper()} operation.")
    print(f"Number of neighbor bundles (k): {args.top_k}")  # 新增
    print(f"Total prompts processed: {len(prompts)}")
    print(f"Successful predictions: {total_samples}")
    print(f"Failed due to extraction issues: {failed_extractions}")
    print(f"Total evaluation time: {total_minutes:.2f} minutes")
    print(f"\nMetrics for {args.mode.upper()} Operation:")
    print(f"    Hit Rate: {hit_rate:.4f}")
    print(f"    Success Rate: {total_samples / len(prompts):.4f}")
    print(f"    Extraction Failure Rate: {failed_extractions / len(prompts):.4f}")
    print(f"Results saved to: {output_file}")
    
    # 写入总体指标到文件
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*50 + "\n")
        f.write(f"Overall Metrics for {args.mode.upper()} Operation:\n")
        f.write(f"    Total prompts processed: {len(prompts)}\n")
        f.write(f"    Successful predictions: {total_samples}\n")
        f.write(f"    Failed due to extraction issues: {failed_extractions}\n")
        f.write(f"    Total evaluation time: {total_minutes:.2f} minutes\n")
        f.write(f"    Hit Rate: {hit_rate:.4f}\n")
        f.write(f"    Success Rate: {total_samples / len(prompts):.4f}\n")
        f.write(f"    Extraction Failure Rate: {failed_extractions / len(prompts):.4f}\n")
        f.write("="*50 + "\n")

if __name__ == '__main__':
    main()