'''
Description: 统一的模型运行脚本 - bundle add任务
Author: anyiran
Date: 2025-09-21
LastEditTime: 2025-09-21
'''

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='运行指定的模型进行bundle add任务')
    parser.add_argument('-m', '--model', type=str, required=True,
                        choices=['tsf', 'mean_vae', 'itemknn', 'concat_vae', 'bprmf'],
                        help='选择要运行的模型: tsf, mean_vae, itemknn, concat_vae, bprmf')
    parser.add_argument('-d', '--dataset', type=str, default='clothing',
                        choices=['food', 'clothing', 'electronic'],
                        help='选择数据集: food, clothing, electronic')
    
    args = parser.parse_args()
    
    # 模型文件映射
    model_files = {
        'tsf': 'tsf_add.py',
        'mean_vae': 'mean_vae_add.py', 
        'itemknn': 'itemknn_add.py',
        'concat_vae': 'concat_vae_add.py',
        'bprmf': 'bprmf_add.py'
    }
    
    model_file = model_files[args.model]
    
    # 检查模型文件是否存在
    if not os.path.exists(model_file):
        print(f"错误：模型文件 {model_file} 不存在！")
        sys.exit(1)
    
    print(f"运行模型: {args.model}")
    print(f"数据集: {args.dataset}")
    print(f"模型文件: {model_file}")
    print("-" * 50)
    
    # 构建命令并执行
    command = f"python {model_file} -d {args.dataset}"
    print(f"执行命令: {command}")
    
    # 执行模型
    exit_code = os.system(command)
    
    if exit_code == 0:
        print(f"\n{args.model} 模型运行完成！")
    else:
        print(f"\n{args.model} 模型运行失败，退出码: {exit_code}")
        sys.exit(1)

if __name__ == "__main__":
    main()
