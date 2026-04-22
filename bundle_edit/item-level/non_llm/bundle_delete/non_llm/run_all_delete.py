'''
Description: 批量运行所有delete任务模型并生成对比报告
Author: anyiran
Date: 2025-09-21
Usage: python run_all_delete.py -d clothing
'''

import argparse
import os
import sys
import subprocess
import time
import pandas as pd
from datetime import datetime

def parse_result_file(result_file, model_name):
    """解析结果文件，提取关键指标"""
    if not os.path.exists(result_file):
        return None
    
    try:
        with open(result_file, 'r') as f:
            content = f.read()
        
        # 提取Hit@1指标
        hit_rate = None
        hit_count = None
        total_bundles = None
        success_rate = None
        time_cost = None
        
        lines = content.split('\n')
        for line in lines:
            if 'Hit@1:' in line:
                hit_rate = float(line.split('Hit@1:')[1].strip())
            elif 'Hit count:' in line:
                hit_count = int(line.split('Hit count:')[1].strip())
            elif 'Total bundles:' in line:
                total_bundles = int(line.split('Total bundles:')[1].strip())
            elif 'Success rate:' in line:
                success_rate = line.split('Success rate:')[1].strip()
            elif 'Time:' in line and 'minutes' in line:
                time_str = line.split('Time:')[1].replace('minutes', '').strip()
                time_cost = float(time_str)
        
        return {
            'Model': model_name,
            'Hit@1': hit_rate,
            'Hit_Count': hit_count,
            'Total_Bundles': total_bundles,
            'Success_Rate': success_rate,
            'Time_Minutes': time_cost
        }
    except Exception as e:
        print(f"WARNING: 解析结果文件 {result_file} 失败: {e}")
        return None

def run_model(dataset, strategy):
    """运行单个模型"""
    print(f"\n[INFO] 正在运行 {strategy.upper()} 模型...")
    
    script_mapping = {
        'tsf': 'tsf_delete.py',
        'mean_vae': 'mean_vae_delete.py', 
        'concat_vae': 'concat_vae_delete.py',
        'itemknn': 'itemknn_delete.py',
        'bprmf': 'bprmf_delete.py'
    }
    
    script_file = script_mapping[strategy]
    
    if not os.path.exists(script_file):
        print(f"[ERROR] 找不到脚本文件 {script_file}")
        return False, None
    
    cmd = ['python', script_file, '-d', dataset]
    
    start_time = time.time()
    
    try:
        # 运行模型，捕获输出但不显示（避免输出过多）
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"[SUCCESS] {strategy.upper()} 运行成功！耗时: {(end_time - start_time)/60:.2f} 分钟")
            return True, (end_time - start_time) / 60
        else:
            print(f"[ERROR] {strategy.upper()} 运行失败！")
            print(f"错误信息: {result.stderr}")
            return False, None
            
    except Exception as e:
        print(f"[ERROR] 运行 {strategy.upper()} 时发生错误: {e}")
        return False, None

def generate_comparison_report(dataset, results):
    """生成对比报告"""
    if not results:
        print("[ERROR] 没有有效的结果数据")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 按Hit@1降序排列
    df = df.sort_values('Hit@1', ascending=False)
    
    print("\n" + "="*80)
    print(f"[REPORT] {dataset.upper()} 数据集 - Delete任务模型对比报告")
    print("="*80)
    
    # 打印表格
    print(f"{'模型':<12} {'Hit@1':<8} {'命中数':<8} {'总数':<8} {'成功率':<12} {'耗时(分钟)':<10}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        model = row['Model']
        hit_rate = f"{row['Hit@1']:.4f}" if row['Hit@1'] is not None else "N/A"
        hit_count = str(row['Hit_Count']) if row['Hit_Count'] is not None else "N/A"
        total = str(row['Total_Bundles']) if row['Total_Bundles'] is not None else "N/A"
        success_rate = row['Success_Rate'] if row['Success_Rate'] is not None else "N/A"
        time_cost = f"{row['Time_Minutes']:.2f}" if row['Time_Minutes'] is not None else "N/A"
        
        print(f"{model:<12} {hit_rate:<8} {hit_count:<8} {total:<8} {success_rate:<12} {time_cost:<10}")
    
    print("="*80)
    
    # 找出最佳模型
    best_model = df.iloc[0]
    print(f"[BEST] 最佳模型: {best_model['Model'].upper()}")
    print(f"[BEST] 最高Hit@1: {best_model['Hit@1']:.4f}")
    
    # 保存详细报告到文件
    report_file = f'../result/{dataset}/delete_comparison_report.txt'
    os.makedirs(f'../result/{dataset}', exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"Bundle Delete Task - 模型对比报告\n")
        f.write(f"数据集: {dataset}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'模型':<12} {'Hit@1':<8} {'命中数':<8} {'总数':<8} {'成功率':<12} {'耗时(分钟)':<10}\n")
        f.write("-" * 80 + "\n")
        
        for _, row in df.iterrows():
            model = row['Model']
            hit_rate = f"{row['Hit@1']:.4f}" if row['Hit@1'] is not None else "N/A"
            hit_count = str(row['Hit_Count']) if row['Hit_Count'] is not None else "N/A"
            total = str(row['Total_Bundles']) if row['Total_Bundles'] is not None else "N/A"
            success_rate = row['Success_Rate'] if row['Success_Rate'] is not None else "N/A"
            time_cost = f"{row['Time_Minutes']:.2f}" if row['Time_Minutes'] is not None else "N/A"
            
            f.write(f"{model:<12} {hit_rate:<8} {hit_count:<8} {total:<8} {success_rate:<12} {time_cost:<10}\n")
        
        f.write("="*80 + "\n")
        f.write(f"最佳模型: {best_model['Model'].upper()}\n")
        f.write(f"最高Hit@1: {best_model['Hit@1']:.4f}\n")
    
    print(f"[OUTPUT] 详细报告已保存到: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='批量运行所有delete任务模型')
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['food', 'clothing', 'electronic'],
                        help='选择数据集: food, clothing, electronic')
    parser.add_argument('-m', '--models', type=str, nargs='+',
                        choices=['tsf', 'mean_vae', 'concat_vae', 'itemknn', 'bprmf'],
                        default=['tsf', 'mean_vae', 'concat_vae', 'itemknn', 'bprmf'],
                        help='选择要运行的模型（默认运行所有模型）')
    parser.add_argument('--skip-existing', action='store_true',
                        help='跳过已有结果文件的模型')
    
    args = parser.parse_args()
    
    print(f"[START] 批量运行Delete任务模型")
    print(f"[CONFIG] 数据集: {args.dataset}")
    print(f"[CONFIG] 模型列表: {', '.join(args.models)}")
    print("="*80)
    
    # 结果文件映射（更新文件名格式）
    result_file_mapping = {
        'tsf': f'../result/{args.dataset}/tsf_delete_{args.dataset}.txt',
        'mean_vae': f'../result/{args.dataset}/mean_vae_delete_{args.dataset}.txt',
        'concat_vae': f'../result/{args.dataset}/concat_vae_delete_{args.dataset}.txt',
        'itemknn': f'../result/{args.dataset}/itemknn_delete_{args.dataset}.txt',
        'bprmf': f'../result/{args.dataset}/bprmf_delete_{args.dataset}.txt'
    }
    
    total_start_time = time.time()
    results = []
    
    for model in args.models:
        result_file = result_file_mapping[model]
        
        # 检查是否跳过已存在的结果
        if args.skip_existing and os.path.exists(result_file):
            print(f"[SKIP] 跳过 {model.upper()}（结果文件已存在）")
            # 仍然解析现有结果
            result_data = parse_result_file(result_file, model.upper())
            if result_data:
                results.append(result_data)
            continue
        
        # 运行模型
        success, time_cost = run_model(args.dataset, model)
        
        if success:
            # 解析结果
            result_data = parse_result_file(result_file, model.upper())
            if result_data:
                results.append(result_data)
        
        # 短暂休息，避免资源冲突
        time.sleep(2)
    
    total_end_time = time.time()
    total_time = (total_end_time - total_start_time) / 60
    
    print(f"\n[COMPLETE] 所有模型运行完成，总耗时: {total_time:.2f} 分钟")
    
    # 生成对比报告
    generate_comparison_report(args.dataset, results)
    
    print("\n[FINISH] 批量任务完成！")

if __name__ == '__main__':
    main()
