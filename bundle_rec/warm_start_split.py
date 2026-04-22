import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split

def split_user_bundle_data_warm_start(
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
    暖启动的数据划分：按交互（user-bundle对）进行7:1:2随机划分。
    在暖启动场景中，验证集和测试集中的bundle可能在训练集中出现过。
    
    Args:
        input_file: 输入的user_bundle数据文件路径（全量数据）
        output_base_dir: 输出基础目录（会在其下创建dataset_name文件夹）
        dataset_name: 数据集名称
        original_data_dir: 原始数据目录，用于复制其他文件（如bundle_item.txt等）
        train_ratio: 训练集比例，默认0.7
        val_ratio: 验证集比例，默认0.1
        test_ratio: 测试集比例，默认0.2（train_ratio + val_ratio + test_ratio 应该等于1）
        random_state: 随机种子
    """
    # 验证比例
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "train_ratio + val_ratio + test_ratio 必须等于1"
    
    # 构建输出目录
    output_dir = os.path.join(output_base_dir, dataset_name)
    
    print(f"开始划分 {dataset_name} 数据集（暖启动）")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"按交互划分比例: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
    print("-" * 60)
    
    # 读取全量数据
    print(f"正在读取数据: {input_file}")
    df = pd.read_csv(input_file, sep='\t', header=None, names=['user_id', 'bundle_id'])
    print(f"总交互数: {len(df)}")
    print(f"总bundle数: {len(df['bundle_id'].unique())}")
    print(f"总user数: {len(df['user_id'].unique())}")

    # 步骤1: 按交互进行7:1:2划分（暖启动：验证集和测试集中的bundle可能在训练集中出现）
    print("\n按交互随机划分（暖启动）...")
    
    # 先划分出训练集
    train_df, temp_df = train_test_split(
        df,
        test_size=val_ratio + test_ratio,
        random_state=random_state
    )
    
    # 再在剩余数据中按比例划分验证集和测试集
    temp_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - temp_val_ratio,
        random_state=random_state
    )

    print("\n划分结果（交互数）:")
    print(f"  训练集交互数: {len(train_df)}")
    print(f"  验证集交互数: {len(val_df)}")
    print(f"  测试集交互数: {len(test_df)}")

    # 统计最终比例
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

    # 统计bundle信息
    train_bundles = set(train_df['bundle_id'].unique())
    val_bundles = set(val_df['bundle_id'].unique())
    test_bundles = set(test_df['bundle_id'].unique())

    print("\nBundle统计:")
    print(f"  训练集 bundle 数: {len(train_bundles)}")
    print(f"  验证集 bundle 数: {len(val_bundles)}")
    print(f"  测试集 bundle 数: {len(test_bundles)}")

    # 统计暖启动bundle（在训练集中出现过的bundle）
    val_warm_bundles = val_bundles & train_bundles
    test_warm_bundles = test_bundles & train_bundles
    val_cold_bundles = val_bundles - train_bundles
    test_cold_bundles = test_bundles - train_bundles

    print(f"\n暖启动bundle统计:")
    print(f"  验证集中的暖启动bundle: {len(val_warm_bundles)} ({len(val_warm_bundles)/len(val_bundles)*100:.1f}%)")
    print(f"  验证集中的冷启动bundle: {len(val_cold_bundles)} ({len(val_cold_bundles)/len(val_bundles)*100:.1f}%)")
    print(f"  测试集中的暖启动bundle: {len(test_warm_bundles)} ({len(test_warm_bundles)/len(test_bundles)*100:.1f}%)")
    print(f"  测试集中的冷启动bundle: {len(test_cold_bundles)} ({len(test_cold_bundles)/len(test_bundles)*100:.1f}%)")

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
    # 在暖启动设定下，测试集中未在训练集出现的bundle视为冷启动bundle
    cold_start_bundles = sorted(list(test_cold_bundles))
    cold_start_bundle_file = os.path.join(output_dir, 'cold_start_bundles.txt')
    with open(cold_start_bundle_file, 'w') as f:
        for bundle_id in cold_start_bundles:
            f.write(f"{bundle_id}\n")
    print(f"\n冷启动bundle列表已保存: {cold_start_bundle_file}")
    print(f"  共 {len(cold_start_bundles)} 个冷启动bundle（在测试集中出现但训练集中未出现）")
    
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
        'cold_start_bundles': test_cold_bundles,
        'warm_start_bundles': test_warm_bundles
    }


if __name__ == '__main__':
    dataset_name = 'iFashion'
    input_file = f'/root/autodl-tmp/datasets/{dataset_name}/user_bundle_all.txt'
    output_base_dir = f'/root/autodl-tmp/Data/warm_bundle_data'
    original_data_dir = '/root/autodl-tmp/datasets'

    # 如果输入文件不存在，尝试合并现有数据
    if not os.path.exists(input_file):
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
            os.makedirs(os.path.dirname(input_file), exist_ok=True)
            combined_df.to_csv(input_file, sep='\t', index=False, header=False)
            print(f"已创建合并文件: {input_file}，共 {len(combined_df)} 条数据")
    
    result = split_user_bundle_data_warm_start(
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