'''
Description: 统一的delete任务运行脚本，支持所有模型
Author: anyiran
Date: 2025-09-21
Usage: python run_delete.py -d clothing -s tsf
'''

import argparse
import os
import sys
import subprocess
import time

def main():
    parser = argparse.ArgumentParser(description='运行bundle delete任务的统一脚本')
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['food', 'clothing', 'electronic'],
                        help='选择数据集: food, clothing, electronic')
    parser.add_argument('-s', '--strategy', type=str, required=True,
                        choices=['tsf', 'mean_vae', 'concat_vae', 'itemknn', 'bprmf'],
                        help='选择模型策略: tsf, mean_vae, concat_vae, itemknn, bprmf')
    parser.add_argument('--train-only', action='store_true',
                        help='只进行训练，不进行测试')
    parser.add_argument('--eval-only', action='store_true', 
                        help='只进行测试，不进行训练（仅支持部分模型）')
    
    args = parser.parse_args()
    
    print(f"[START] 开始运行Delete任务")
    print(f"[CONFIG] 数据集: {args.dataset}")
    print(f"[CONFIG] 模型策略: {args.strategy}")
    print("="*50)
    
    # 构建对应的脚本文件名
    script_mapping = {
        'tsf': 'tsf_delete.py',
        'mean_vae': 'mean_vae_delete.py', 
        'concat_vae': 'concat_vae_delete.py',
        'itemknn': 'itemknn_delete.py',
        'bprmf': 'bprmf_delete.py'
    }
    
    script_file = script_mapping[args.strategy]
    
    # 检查脚本文件是否存在
    if not os.path.exists(script_file):
        print(f"[ERROR] 找不到脚本文件 {script_file}")
        sys.exit(1)
    
    # 构建命令行参数
    cmd = ['python', script_file, '-d', args.dataset]
    
    # 添加可选参数（仅对支持的模型）
    if args.strategy in ['concat_vae', 'itemknn']:
        if args.train_only:
            cmd.append('--train-only')
        elif args.eval_only:
            cmd.append('--eval-only')
    elif args.train_only or args.eval_only:
        print(f"[WARNING] {args.strategy}模型不支持--train-only或--eval-only参数，将运行完整流程")
    
    print(f"[EXEC] 执行命令: {' '.join(cmd)}")
    print("="*50)
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 运行对应的脚本
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        # 记录结束时间
        end_time = time.time()
        total_time = (end_time - start_time) / 60
        
        print("\n" + "="*50)
        if result.returncode == 0:
            print(f"[SUCCESS] {args.strategy.upper()}模型运行成功！")
            print(f"[TIME] 总耗时: {total_time:.2f} 分钟")
            
            # 显示结果文件位置
            result_dir = f'../result/{args.dataset}'
            result_file = f'{result_dir}/{args.strategy}_delete_{args.dataset}.txt'
            
            if os.path.exists(result_file):
                print(f"[OUTPUT] 结果文件: {result_file}")
                
                # 尝试读取并显示结果摘要
                try:
                    with open(result_file, 'r') as f:
                        content = f.read()
                        print(f"[RESULT] 结果摘要:")
                        print(content)
                except:
                    pass
            
        else:
            print(f"[ERROR] {args.strategy.upper()}模型运行失败！")
            print(f"[ERROR] 退出码: {result.returncode}")
            sys.exit(result.returncode)
        
    except KeyboardInterrupt:
        print(f"\n[INTERRUPT] 用户中断了程序执行")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 运行过程中发生错误: {e}")
        sys.exit(1)
    
    print("="*50)
    print("[FINISH] 任务完成！")

if __name__ == '__main__':
    main()