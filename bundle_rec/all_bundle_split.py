#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行BI划分的完整流程
1. 先运行UB划分（如果还没有运行）
2. 然后运行BI划分
"""

import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ub_split import split_user_bundle_data
from bi_split_with_cold_start import split_bundle_item_with_cold_start

def main():
    # 示例：划分electronic数据集
    input_file = '/root/autodl-tmp/datasets/iFashion/user_bundle_all.txt'
    output_base_dir = '/root/autodl-tmp/Data/All_bundle_data'  # 最上层文件夹名
    original_data_dir = '/root/autodl-tmp/datasets'  # 用于复制其他文件
    dataset_name = 'iFashion'
    data_dir = os.path.join(output_base_dir, dataset_name)
    
    # 检查是否需要先运行UB划分
    cold_start_bundle_file = os.path.join(data_dir, 'cold_start_bundles.txt')
    if not os.path.exists(cold_start_bundle_file):
        print("=" * 60)
        print("第一步: 运行UB划分...")
        print("=" * 60)
        
        # 如果输入文件不存在，尝试合并train和test
        if not os.path.exists(input_file):
            train_file = '/root/autodl-tmp/datasets/iFashion/user_bundle_train.txt'
            test_file = '/root/autodl-tmp/datasets/iFashion/user_bundle_test.txt'
            if os.path.exists(train_file) and os.path.exists(test_file):
                import pandas as pd
                train_df = pd.read_csv(train_file, sep='\t', header=None, names=['user_id', 'bundle_id'])
                test_df = pd.read_csv(test_file, sep='\t', header=None, names=['user_id', 'bundle_id'])
                combined_df = pd.concat([train_df, test_df], ignore_index=True)
                input_file = '/root/autodl-tmp/datasets/iFashion/user_bundle_all.txt'
                combined_df.to_csv(input_file, sep='\t', index=False, header=False)
                print(f"已创建合并文件: {input_file}")
        
        # 运行UB划分
        result = split_user_bundle_data(
            input_file=input_file,
            output_base_dir=output_base_dir,
            dataset_name=dataset_name,
            original_data_dir=original_data_dir,
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
            test_cold_start_ratio=0.3,
            val_warm_start_ratio=1.0,
            cold_start_strategy='low_interaction',
            random_state=123
        )
        print("\nUB划分完成!")
    else:
        print("=" * 60)
        print("检测到已存在UB划分结果，跳过UB划分步骤")
        print("=" * 60)
    
    # 运行BI划分
    print("\n" + "=" * 60)
    print("第二步: 运行BI划分...")
    print("=" * 60)
    
    result = split_bundle_item_with_cold_start(
        dataset_name=dataset_name,
        data_dir=data_dir,
        random_state=123
    )
    
    print("\n" + "=" * 60)
    print("所有划分完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()

