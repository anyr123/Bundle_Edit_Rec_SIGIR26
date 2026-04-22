

import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split

def split_user_bundle_data(
    input_file,
    output_base_dir,
    dataset_name,
    original_data_dir=None,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    random_state=123
):
    """
    纯冷启动的数据划分：先按 bundle 7:1:2 划分，再根据 bundle 归属划分其所有交互。
    
    Args:
        input_file: 输入的user_bundle数据文件路径（全量数据）
        output_base_dir: 输出基础目录（会在其下创建dataset_name文件夹）
        dataset_name: 数据集名称
        original_data_dir: 原始数据目录，用于复制其他文件（如bundle_item.txt等）
        train_ratio: 训练集 bundle 比例，默认0.7
        val_ratio: 验证集 bundle 比例，默认0.1
        test_ratio: 测试集 bundle 比例，默认0.2（train_ratio + val_ratio + test_ratio 应该等于1）
        random_state: 随机种子
    """
    # 验证比例
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "train_ratio + val_ratio + test_ratio 必须等于1"
    
    # 构建输出目录（与Original_data结构一致）
    output_dir = os.path.join(output_base_dir, dataset_name)
    
    print(f"开始划分 {dataset_name} 数据集")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"按 bundle 划分比例: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
    print("-" * 60)
    
    # 读取全量数据
    print(f"正在读取数据: {input_file}")
    df = pd.read_csv(input_file, sep='\t', header=None, names=['user_id', 'bundle_id'])
    print(f"总交互数: {len(df)}")
    all_bundles = df['bundle_id'].unique()
    print(f"总bundle数: {len(all_bundles)}")
    print(f"总user数: {len(df['user_id'].unique())}")

    # 步骤1: 按 bundle 进行 7:1:2 划分（即纯冷启动：测试集 bundle 在训练/验证集中完全不可见）
    print("\n按 bundle 划分（纯冷启动）...")
    # 先划分出训练 bundle
    train_bundles, temp_bundles = train_test_split(
        all_bundles,
        test_size=val_ratio + test_ratio,
        random_state=random_state
    )

    # 再在剩余 bundle 中按比例继续划分验证和测试
    # 这里使用 val_ratio/(val_ratio+test_ratio) 来保持整体 7:1:2
    temp_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_bundles, test_bundles = train_test_split(
        temp_bundles,
        test_size=1 - temp_val_ratio,
        random_state=random_state
    )

    train_bundles_set = set(train_bundles)
    val_bundles_set = set(val_bundles)
    test_bundles_set = set(test_bundles)

    print(f"  训练集 bundle 数: {len(train_bundles_set)}")
    print(f"  验证集 bundle 数: {len(val_bundles_set)}")
    print(f"  测试集 bundle 数: {len(test_bundles_set)}")

    # 步骤2: 根据 bundle 归属划分交互
    train_df = df[df['bundle_id'].isin(train_bundles_set)].copy()
    val_df = df[df['bundle_id'].isin(val_bundles_set)].copy()
    test_df = df[df['bundle_id'].isin(test_bundles_set)].copy()

    print("\n根据 bundle 归属得到的交互数:")
    print(f"  训练集交互数: {len(train_df)}")
    print(f"  验证集交互数: {len(val_df)}")
    print(f"  测试集交互数: {len(test_df)}")

    # 统计最终比例（按交互数）
    total_final = len(train_df) + len(val_df) + len(test_df)
    actual_train_ratio = len(train_df) / total_final if total_final > 0 else 0
    actual_val_ratio = len(val_df) / total_final if total_final > 0 else 0
    actual_test_ratio = len(test_df) / total_final if total_final > 0 else 0
    
    print("\n" + "=" * 60)
    print("划分结果统计:")
    print("-" * 60)
    print(f"训练集: {len(train_df)} 条交互 ({actual_train_ratio:.2%}, 目标: {train_ratio:.2%})")
    print(f"验证集: {len(val_df)} 条交互 ({actual_val_ratio:.2%}, 目标: {val_ratio:.2%})")
    print(f"测试集: {len(test_df)} 条交互 ({actual_test_ratio:.2%}, 目标: {test_ratio:.2%})")
    print(f"总计: {total_final} 条交互")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存user_bundle文件（与Original_data命名一致）
    train_file = os.path.join(output_dir, 'user_bundle_train.txt')
    val_file = os.path.join(output_dir, 'user_bundle_tune.txt')
    test_file = os.path.join(output_dir, 'user_bundle_test.txt')
    
    train_df[['user_id', 'bundle_id']].to_csv(
        train_file, sep='\t', index=False, header=False
    )
    val_df[['user_id', 'bundle_id']].to_csv(
        val_file, sep='\t', index=False, header=False
    )
    test_df[['user_id', 'bundle_id']].to_csv(
        test_file, sep='\t', index=False, header=False
    )
    
    print(f"\nuser_bundle文件已保存:")
    print(f"  训练集: {train_file}")
    print(f"  验证集: {val_file}")
    print(f"  测试集: {test_file}")
    
    # 保存冷启动bundle列表到文件
    # 在纯冷启动设定下，测试集中出现的 bundle 视为冷启动 bundle
    cold_start_bundles = sorted(list(test_bundles_set))
    cold_start_bundle_file = os.path.join(output_dir, 'cold_start_bundles.txt')
    with open(cold_start_bundle_file, 'w') as f:
        for bundle_id in cold_start_bundles:
            f.write(f"{bundle_id}\n")
    print(f"\n冷启动bundle列表已保存: {cold_start_bundle_file}")
    print(f"  共 {len(cold_start_bundles)} 个冷启动bundle（即仅出现在测试集中的 bundle）")
    
    # 复制其他文件（如果提供了原始数据目录）
    if original_data_dir is not None:
        original_dataset_dir = os.path.join(original_data_dir, dataset_name)
        if os.path.exists(original_dataset_dir):
            print(f"\n正在从 {original_dataset_dir} 复制其他文件...")
            files_to_copy = [
                'bundle_item.txt',
                'user_item.txt',
                f'{dataset_name}_data_size.txt'
            ]
            
            for filename in files_to_copy:
                src_file = os.path.join(original_dataset_dir, filename)
                dst_file = os.path.join(output_dir, filename)
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dst_file)
                    print(f"  已复制: {filename}")
                else:
                    print(f"  警告: 文件不存在，跳过: {filename}")
        else:
            print(f"\n警告: 原始数据目录不存在: {original_dataset_dir}")
    
    print("=" * 60)
    
    return {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'cold_start_bundles': set(cold_start_bundles)
    }


if __name__ == '__main__':
    dataset_name = 'iFashion'
    input_file = f'/root/autodl-tmp/datasets/{dataset_name}/user_bundle_all.txt'
    # input_file = f'/root/autodl-tmp/Data/merged/user_bundle_all.txt'
    output_base_dir = '/root/autodl-tmp/Data/cold_bundle_data'  # 最上层文件夹名
    original_data_dir = '/root/autodl-tmp/datasets'  # 用于复制其他文件

    # 如果输入文件不存在，尝试使用合并后的全量数据
    if not os.path.exists(input_file):
        # 尝试合并train、tune和test作为全量数据
        train_file = f'/root/autodl-tmp/datasets/{dataset_name}/user_bundle_train.txt'
        tune_file = f'/root/autodl-tmp/datasets/{dataset_name}/user_bundle_tune.txt'
        test_file = f'/root/autodl-tmp/datasets/{dataset_name}/user_bundle_test.txt'
        
        dataframes = []
        if os.path.exists(train_file):
            train_df = pd.read_csv(train_file, sep='\t', header=None, names=['user_id', 'bundle_id'])
            dataframes.append(train_df)
            print(f"已读取训练集: {len(train_df)} 条")
        if os.path.exists(tune_file):
            tune_df = pd.read_csv(tune_file, sep='\t', header=None, names=['user_id', 'bundle_id'])
            dataframes.append(tune_df)
            print(f"已读取验证集: {len(tune_df)} 条")
        if os.path.exists(test_file):
            test_df = pd.read_csv(test_file, sep='\t', header=None, names=['user_id', 'bundle_id'])
            dataframes.append(test_df)
            print(f"已读取测试集: {len(test_df)} 条")
        
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            input_file = f'/root/autodl-tmp/datasets/{dataset_name}/user_bundle_all.txt'
            combined_df.to_csv(input_file, sep='\t', index=False, header=False)
            print(f"已创建合并文件: {input_file}，共 {len(combined_df)} 条数据")
    
    result = split_user_bundle_data(
        input_file=input_file,
        output_base_dir=output_base_dir,
        dataset_name=dataset_name,
        original_data_dir=original_data_dir,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        random_state=123
    )
    
    print("\n划分完成!")

