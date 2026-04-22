
import pandas as pd
import random
import yaml
import argparse
import os

def get_random_item_not_in_sets(all_items, exclude_sets):
    """获取一个不在任何exclude_sets中的随机item"""
    available_items = all_items.copy()
    for item_set in exclude_sets:
        available_items -= set(item_set)
    if not available_items:
        raise ValueError("No available items left!")
    return random.choice(list(available_items))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_name', type=str, required=True, help='数据集名称，如clothing、electronic、food')
    args = parser.parse_args()

    data_name = args.data_name
    print(f"当前使用的数据集: {data_name}")
    random.seed(42)
    
    # 构建数据路径
    data_path = f'dataset/{data_name}/'
    test_data_path = f'testdata/{data_name}/'
    os.makedirs(test_data_path, exist_ok=True)
    
    # 读取数据
    df = pd.read_csv(os.path.join(data_path, 'bundle_item.csv'))
    
    # 获取所有 bundle_id 和 items
    bundle_to_items = df.groupby('bundle ID')['item ID'].apply(list).to_dict()
    
    # 筛选出item数量大于等于3的bundle
    valid_bundles = [bid for bid, items in bundle_to_items.items() if len(items) >= 3]
    num_valid_bundles = len(valid_bundles)
    all_items = set(df['item ID'].unique())
    
    # 计算总样本数（20%的valid bundles）
    sample_size = max(4, int(0.2 * num_valid_bundles))
    
    print(f"总bundle数量: {len(bundle_to_items)}")
    print(f"有效bundle数量 (>= 3 items): {num_valid_bundles}")
    print(f"采样数量(20%): {sample_size}")
    
    # 随机选取bundle_ids
    sampled_bundles = random.sample(valid_bundles, sample_size)
    
    # 每个bundle都用于生成三种类型的测试数据
    type_bundles = {
        'add': sampled_bundles,
        'delete': sampled_bundles,
        'update': sampled_bundles
    }
    
    # 为三种不同类型准备数据结构
    test_types = {
        'add': [],      # 增：bundle-1, 1pos+19neg
        'delete': [],   # 删：bundle+1, 无候选集
        'update': []  # 更新：bundle-1+1, 1pos+19neg
    }
    
    # 生成每种类型的测试数据
    for test_type, bundle_ids in type_bundles.items():
        for bundle_id in bundle_ids:
            items = bundle_to_items[bundle_id]
            original_items = items.copy()
            
            if test_type == 'add':
                # 选择bundle的最后一个item作为正例
                positive_item = items[-1]  # 最后一个item
                # 移除最后一个item后的bundle
                bundle_without_item = items[:-1]  # 除了最后一个item之外的所有items
            
            # 负例池：从全集中排除当前bundle的items
            negative_pool = list(all_items - set(items))
            
            if test_type == 'add':
                # bundle-1, 1pos+19neg
                neg_items = random.sample(negative_pool, 19)
                candidates = [positive_item] + neg_items
                random.shuffle(candidates)
                test_types['add'].append({
                    'bundle_id': bundle_id,
                    'bundle_items': bundle_without_item,  # 使用移除一个item后的bundle
                    'candidates': candidates
                })
                
            elif test_type == 'delete':
                # bundle+1, 无候选集
                # 随机选择一个新item添加到bundle中
                new_item = get_random_item_not_in_sets(all_items, [set(items)])
                bundle_plus_one = items + [new_item]  # 在原bundle后添加新item
                
                test_types['delete'].append({
                    'bundle_id': bundle_id,
                    'bundle_items': bundle_plus_one,  # 使用bundle+1作为bundle内容
                    'candidates': []  # 删除操作不需要候选集
                })
                
            elif test_type == 'update':
                # 选择bundle的最后一个item作为要被替换的item（与ADD操作保持一致）
                removed_item = items[-1]  # 最后一个item
                
                # 移除最后一个item后的bundle
                bundle_without_removed = items[:-1]  # 除了最后一个item之外的所有items
                
                # 为bundle添加一个新的随机item（确保最终长度与原始bundle相同）
                new_item = get_random_item_not_in_sets(all_items, [set(items)])
                bundle_modified = bundle_without_removed + [new_item]  # 添加新item
                
                # 生成候选集（包含被移除的最后一个item作为正例）
                neg_items_update = random.sample(negative_pool, 19)
                candidates = [removed_item] + neg_items_update  # 使用被移除的最后一个item作为正例
                random.shuffle(candidates)
                
                test_types['update'].append({
                    'bundle_id': bundle_id,
                    'bundle_items': bundle_modified,  # 使用修改后的bundle
                    'candidates': candidates
                })
                
    
    # 写入文件并输出统计信息
    total_generated = 0
    for test_type, data in test_types.items():
        output_file = os.path.join(test_data_path, f'{test_type}_test.txt')
        with open(output_file, 'w') as f:
            for entry in data:
                bundle_str = ' '.join(map(str, entry['bundle_items']))
                candidates_str = ' '.join(map(str, entry['candidates']))
                f.write(f"{entry['bundle_id']}\t{bundle_str}\t{candidates_str}\n")
        total_generated += len(data)
        print(f"已生成{test_type}测试数据：{output_file}")
        print(f"{test_type}数据集大小: {len(data)} ({len(data)/sample_size*100:.1f}%)")
    
    print(f"\n总共生成数据量: {total_generated}")
    print(f"占有效bundle比例: {total_generated/num_valid_bundles*100:.1f}%")

if __name__ == '__main__':
    main()