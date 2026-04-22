'''
Description: Process bundle operation test data using zero-shot LLM with YAML configuration
Author: anyiran
Date: 2025-07-13
'''
import pandas as pd
import openai
import time
import math
import re
import argparse
import os
import yaml
import random

# ========== GPT-3.5 API 配置 ==========
openai.api_key = ""
openai.api_base = ""

def load_zeroshot_config(config_path='../props/zeroshot.yaml'):
    """
    Load zeroshot configuration from YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary containing prompt templates
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file {config_path} not found. Please create the YAML configuration file.")
        return None

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
    # 3. 匹配 Removed item: 5. "xxx"
    match = re.search(r"Removed item:\s*(\d+)\.", response_text)
    if match:
        return match.group(1)
    # 4. 匹配 Selected item: 5. "xxx"
    match = re.search(r"Selected item:\s*(\d+)\.", response_text)
    if match:
        return match.group(1)
    # 5. 兼容原有格式
    match = re.search(r"\[(\d+)\]\s*[-–]", response_text)
    if match: return match.group(1)
    match = re.search(r"\b(\d+)\s*[-–]", response_text)
    if match: return match.group(1)
    # 6. 匹配纯数字（作为最后的备选方案）
    match = re.search(r"\b(\d+)\b", response_text)
    if match: return match.group(1)
    return None

def extract_update_results(response_text):
    """Extract both removed and selected items for update operations"""
    removed_match = re.search(r"Removed item:\s*(\d+)\.", response_text)
    selected_match = re.search(r"Selected item:\s*(\d+)\.", response_text)
    
    removed_id = removed_match.group(1) if removed_match else None
    selected_id = selected_match.group(1) if selected_match else None
    
    return removed_id, selected_id

def get_bundle_items(bundle_ids, bundle_item_df, item_titles_df):
    """
    Retrieves the item IDs and titles for a given list of bundle IDs.
    """
    if not bundle_ids:
        return []
    bundle_items_ids_df = bundle_item_df[bundle_item_df['bundle ID'].isin(bundle_ids)]
    bundle_items_df = bundle_items_ids_df.merge(item_titles_df, on='item ID', how='left').dropna()
    return list(zip(bundle_items_df['item ID'], bundle_items_df['titles']))

def main():
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_name', type=str, required=True, 
                        choices=['food', 'clothing', 'electronic'],
                        help='Dataset name: clothing, electronic, food')
    parser.add_argument('-m', '--mode', type=str, required=True, 
                        choices=['add', 'delete', 'update'], 
                        help='Operation mode: add, delete, or update')
    parser.add_argument('-c', '--config', type=str, default='../props/zeroshot.yaml', 
                        help='Path to zeroshot configuration YAML file')
    
    args = parser.parse_args()

    # Load zeroshot configuration
    config = load_zeroshot_config(args.config)
    if config is None:
        return
    print(f"Loaded configuration from: {args.config}")
    print(f"Running in mode: {args.mode}")

    # ========== 构建数据集路径 ==========
    dataset_path = f'../../dataset/{args.data_name}/'
    test_data_path = f'../../testdata/{args.data_name}/'
    bundle_item_path = os.path.join(dataset_path, 'bundle_item.csv').replace('\\', '/')
    item_titles_path = os.path.join(dataset_path, 'item_titles.csv').replace('\\', '/')
    
    print(f"🔹使用数据集: {args.data_name}")
    print(f"🔹数据集路径: {dataset_path}")

    # Check if files exist
    if not os.path.exists(bundle_item_path) or not os.path.exists(item_titles_path):
        print("Error: Missing bundle_item.csv or item_titles.csv in the dataset directory.")
        return

    # ========== 读取数据 ==========
    item_titles_df = pd.read_csv(item_titles_path)
    bundle_item_df = pd.read_csv(bundle_item_path)

    # Read test files
    test_type = args.mode
    
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

    # ========== 构建 prompt 并评估 ==========
    prompts = []
    test_cases = []
    
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

        # 构建 prompt - 使用与ours.py相同的逻辑
        if test_type == 'add':
            # For ADD: 当前bundle作为输入，候选集包含应该被添加的item
            bundle_lines = ',\n'.join([f'{idx}. "{item[1]}"' for idx, item in enumerate(bundle_items)])
            cand_lines = ',\n'.join([f'{i}. "{it[1]}"' for i, it in enumerate(candidate_items)])
            
            # 获取ground truth bundle来确定哪个item应该被添加
            ground_truth_items = get_bundle_items([bundle_id], bundle_item_df, item_titles_df)
            ground_truth_ids = {item[0] for item in ground_truth_items}
            current_bundle_ids = set(all_items_ids)
            
            # 应该被添加的items（在ground truth中但不在current bundle中）
            items_to_add = ground_truth_ids - current_bundle_ids
            if items_to_add and candidate_items:
                # 找到第一个应该被添加的item在候选集中的位置
                gt_pos = None
                for item_id in items_to_add:
                    try:
                        gt_pos = str(candidates_ids.index(item_id))
                        break
                    except ValueError:
                        continue
                
                if gt_pos is not None:
                    test_cases.append({
                        'test_type': test_type,
                        'bundle_id': bundle_id,
                        'gt_pos': gt_pos,
                        'bundle_items': bundle_items,
                        'candidate_items': candidate_items
                    })
                else:
                    print(f"Warning: No valid GT item found in candidates for bundle {bundle_id}")
                    continue
            else:
                print(f"Warning: No items to add or no candidates for bundle {bundle_id}")
                continue
                
        elif test_type == 'delete':
            # For DELETE: 不打乱顺序，直接使用原始bundle items
            bundle_lines = ',\n'.join([f'{idx}. "{item[1]}"' for idx, item in enumerate(bundle_items)])
            cand_lines = ""  # DELETE不需要候选集
            
            # 获取ground truth bundle来确定哪个item应该被删除
            ground_truth_items = get_bundle_items([bundle_id], bundle_item_df, item_titles_df)
            ground_truth_ids = {item[0] for item in ground_truth_items}
            current_bundle_ids = set(all_items_ids)
            
            # 应该被删除的items（在current bundle中但不在ground truth中）
            items_to_delete = current_bundle_ids - ground_truth_ids
            if items_to_delete:
                # 在bundle中找到应该被删除的item位置
                for idx, (item_id, title) in enumerate(bundle_items):
                    if item_id in items_to_delete:
                        test_cases.append({
                            'test_type': test_type,
                            'bundle_id': bundle_id,
                            'gt_pos': str(idx),
                            'bundle_items': bundle_items,  # 使用原始顺序的items
                            'candidate_items': []
                        })
                        break
            else:
                print(f"Warning: No items to delete for bundle {bundle_id}")
                continue
                
        elif test_type == 'update':
            # For UPDATE: 需要删除一个item并添加一个item
            bundle_lines = ',\n'.join([f'{idx}. "{item[1]}"' for idx, item in enumerate(bundle_items)])
            cand_lines = ',\n'.join([f'{i}. "{it[1]}"' for i, it in enumerate(candidate_items)])
            
            # 获取ground truth bundle
            ground_truth_items = get_bundle_items([bundle_id], bundle_item_df, item_titles_df)
            ground_truth_ids = {item[0] for item in ground_truth_items}
            current_bundle_ids = set(all_items_ids)
            
            # 应该被删除的items和应该被添加的items
            items_to_remove = current_bundle_ids - ground_truth_ids
            items_to_add = ground_truth_ids - current_bundle_ids
            
            if items_to_remove and items_to_add and candidate_items:
                # 找到要删除的item在bundle中的位置
                remove_pos = None
                for idx, (item_id, title) in enumerate(bundle_items):
                    if item_id in items_to_remove:
                        remove_pos = str(idx)
                        break
                
                # 找到要添加的item在候选集中的位置
                add_pos = None
                for item_id in items_to_add:
                    try:
                        add_pos = str(candidates_ids.index(item_id))
                        break
                    except ValueError:
                        continue
                
                if remove_pos is not None and add_pos is not None:
                    test_cases.append({
                        'test_type': test_type,
                        'bundle_id': bundle_id,
                        'gt_remove_pos': remove_pos,
                        'gt_add_pos': add_pos,
                        'bundle_items': bundle_items,
                        'candidate_items': candidate_items
                    })
                else:
                    print(f"Warning: Cannot find valid remove/add positions for bundle {bundle_id}")
                    continue
            else:
                print(f"Warning: Invalid update case for bundle {bundle_id}")
                continue

        # Build prompt from config
        prompt_template = config[prompt_key]
        prompt = prompt_template.format(
            bundle_lines=bundle_lines,
            cand_lines=cand_lines
        )
        prompts.append(prompt)

    # ========== 主评估循环 ==========
    max_retries = 5
    hit_count = 0
    failed = 0
    start_time = time.time()

    print(f"Total test cases: {len(prompts)}")

    for idx, (prompt, test_case) in enumerate(zip(prompts, test_cases)):
        success = False
        for attempt in range(max_retries):
            try:
                print(f"\nEvaluating prompt {idx + 1}/{len(prompts)} - Attempt {attempt + 1}")
                print(f"Test type: {test_case['test_type']}")
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    n=1,
                )

                result = response["choices"][0]["message"]["content"].strip()
                print(f"Prompt:\n{prompt}")
                print("--------------------------------")
                print(f"Response:\n{result}")
                
                # 使用与ours.py相同的评估逻辑
                if test_case['test_type'] == 'add':
                    pred_id = extract_pred_id(result)
                    if pred_id:
                        # 获取ground truth bundle来进行hit计算
                        ground_truth_items = get_bundle_items([test_case['bundle_id']], bundle_item_df, item_titles_df)
                        ground_truth_ids = {item[0] for item in ground_truth_items}
                        current_bundle_ids = set([item[0] for item in test_case['bundle_items']])
                        
                        # 应该被添加的items
                        positive_items = ground_truth_ids - current_bundle_ids
                        
                        # 检查预测的候选item是否是正确答案
                        pred_idx = int(pred_id)
                        if pred_idx < len(test_case['candidate_items']):
                            predicted_item_id = test_case['candidate_items'][pred_idx][0]
                            if predicted_item_id in positive_items:
                                hit_count += 1
                                print("✅ Hit@1!")
                            else:
                                print("❌ Missed!")
                        else:
                            print("❌ Missed! (Invalid prediction index)")
                        success = True
                        break
                    else:
                        print("Prediction ID missing, retrying...")
                        
                elif test_case['test_type'] == 'delete':
                    pred_id = extract_pred_id(result)
                    if pred_id:
                        # 获取ground truth bundle来进行hit计算
                        ground_truth_items = get_bundle_items([test_case['bundle_id']], bundle_item_df, item_titles_df)
                        ground_truth_ids = {item[0] for item in ground_truth_items}
                        current_bundle_ids = set([item[0] for item in test_case['bundle_items']])
                        
                        # 应该被删除的items
                        items_to_delete = current_bundle_ids - ground_truth_ids
                        
                        # 检查预测删除的item是否正确
                        pred_idx = int(pred_id)
                        if pred_idx < len(test_case['bundle_items']):
                            predicted_item_id = test_case['bundle_items'][pred_idx][0]
                            if predicted_item_id in items_to_delete:
                                hit_count += 1
                                print("✅ Hit@1!")
                            else:
                                print("❌ Missed!")
                        else:
                            print("❌ Missed! (Invalid prediction index)")
                        success = True
                        break
                    else:
                        print("Prediction ID missing, retrying...")
                        
                elif test_case['test_type'] == 'update':
                    removed_id, selected_id = extract_update_results(result)
                    if removed_id and selected_id:
                        # 获取ground truth bundle来进行hit计算
                        ground_truth_items = get_bundle_items([test_case['bundle_id']], bundle_item_df, item_titles_df)
                        ground_truth_ids = {item[0] for item in ground_truth_items}
                        current_bundle_ids = set([item[0] for item in test_case['bundle_items']])
                        
                        # 应该被删除和添加的items
                        items_to_remove = current_bundle_ids - ground_truth_ids
                        items_to_add = ground_truth_ids - current_bundle_ids
                        
                        # 检查删除操作
                        remove_correct = False
                        removed_idx = int(removed_id)
                        if removed_idx < len(test_case['bundle_items']):
                            removed_item_id = test_case['bundle_items'][removed_idx][0]
                            remove_correct = removed_item_id in items_to_remove
                        
                        # 检查添加操作
                        add_correct = False
                        selected_idx = int(selected_id)
                        if selected_idx < len(test_case['candidate_items']):
                            selected_item_id = test_case['candidate_items'][selected_idx][0]
                            add_correct = selected_item_id in items_to_add
                        
                        if remove_correct and add_correct:
                            hit_count += 1
                            print("✅ Hit@1! (Both remove and add correct)")
                        else:
                            print(f"❌ Missed! Remove correct: {remove_correct}, Add correct: {add_correct}")
                        success = True
                        break
                    else:
                        print("Prediction IDs missing, retrying...")

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1.5)

        if not success:
            failed += 1
            print(f"Failed to process test case {idx + 1}")

    end_time = time.time()
    total_minutes = (end_time - start_time) / 60

    # ========== 输出结果 ==========
    hit_rate_at_1 = hit_count / len(prompts) if len(prompts) > 0 else 0
    
    print(f"\nEvaluation completed for {args.mode.upper()} operation.")
    print(f"Total prompts: {len(prompts)}")
    print(f"Failed prompts: {failed}")
    print(f"Hit Rate@1: {hit_rate_at_1:.4f}")
    print(f"Total evaluation time: {total_minutes:.2f} minutes")
    save_data_path = f'../result/{args.data_name}'
    os.makedirs(save_data_path, exist_ok=True)
    # Save results
    output_file = os.path.join(save_data_path, f'{args.mode}_zeroshot_results.txt')
    with open(output_file,  'a', encoding='utf-8') as f:
        f.write(f"Configuration used: {args.config}\n")
        f.write(f"Operation mode: {args.mode.upper()}\n")
        f.write(f"Dataset: {args.data_name}\n")
        f.write("="*50 + "\n")
        f.write(f"Total prompts: {len(prompts)}\n")
        f.write(f"Failed prompts: {failed}\n")
        f.write(f"Hit Rate@1: {hit_rate_at_1:.4f}\n")
        f.write(f"Total evaluation time: {total_minutes:.2f} minutes\n")
        f.write("="*50 + "\n")
    
    print(f"Results saved to: {output_file}")

if __name__ == '__main__':
    main()