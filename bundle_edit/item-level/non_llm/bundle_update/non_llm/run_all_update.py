'''
Description: 批量运行所有update任务模型并生成对比报告
Author: anyiran
Date: 2025-09-22
Usage: python run_all_update.py -d clothing
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
        
        # 提取关键指标
        overall_hit_rate = None
        success_hit_rate = None
        hits = None
        total_samples = None
        successful_count = None
        failed_count = None
        delete_accuracy = None
        add_accuracy = None
        random_baseline = None
        time_cost = None
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if 'Overall Hit Rate:' in line:
                # 提取格式如 "Overall Hit Rate: 0.1234 (12.34%)"
                parts = line.split('Overall Hit Rate:')[1].strip()
                overall_hit_rate = float(parts.split()[0])
            elif 'Success Hit Rate:' in line:
                # 提取格式如 "Success Hit Rate: 0.1234 (12.34%)"
                parts = line.split('Success Hit Rate:')[1].strip()
                success_hit_rate = float(parts.split()[0])
            elif line.startswith('Hits:') or line.startswith('Overall hits:'):
                # 提取格式如 "Hits: 123" 或 "Overall hits: 123"
                if 'Hits:' in line:
                    hits = int(line.split('Hits:')[1].strip())
                else:
                    hits = int(line.split('Overall hits:')[1].strip())
            elif 'Total samples:' in line:
                total_samples = int(line.split('Total samples:')[1].strip())
            elif 'Successfully processed:' in line:
                successful_count = int(line.split('Successfully processed:')[1].strip())
            elif 'Failed:' in line:
                failed_count = int(line.split('Failed:')[1].strip())
            elif 'Delete accuracy:' in line or 'Delete Accuracy:' in line:
                # 提取格式如 "Delete accuracy: 0.1234 (12/100)"
                if 'Delete accuracy:' in line:
                    parts = line.split('Delete accuracy:')[1].strip()
                else:
                    parts = line.split('Delete Accuracy:')[1].strip()
                delete_accuracy = float(parts.split()[0])
            elif 'Add accuracy:' in line or 'Add Accuracy:' in line:
                # 提取格式如 "Add accuracy: 0.1234 (12/100)"
                if 'Add accuracy:' in line:
                    parts = line.split('Add accuracy:')[1].strip()
                else:
                    parts = line.split('Add Accuracy:')[1].strip()
                add_accuracy = float(parts.split()[0])
            elif 'Random baseline:' in line:
                # 提取格式如 "Random baseline: 0.1234 (12.34%)"
                parts = line.split('Random baseline:')[1].strip()
                random_baseline = float(parts.split()[0])
            elif ('Training + Validation time:' in line or 
                  'Training + Evaluation time:' in line or 
                  'Time:' in line) and 'minutes' in line:
                # 提取时间信息
                if 'Training + Validation time:' in line:
                    time_str = line.split('Training + Validation time:')[1].replace('minutes', '').strip()
                elif 'Training + Evaluation time:' in line:
                    time_str = line.split('Training + Evaluation time:')[1].replace('minutes', '').strip()
                else:
                    time_str = line.split('Time:')[1].replace('minutes', '').strip()
                try:
                    time_cost = float(time_str)
                except:
                    pass
        
        #
        # 对于ItemKNN，可能使用不同的格式
        if model_name.upper() == 'ITEMKNN':
            for line in lines:
                line = line.strip()
                if 'Hit@1:' in line:  # 改为新格式
                    parts = line.split('Hit@1:')[1].strip()
                    overall_hit_rate = float(parts)
                    success_hit_rate = overall_hit_rate
                elif 'Hit count:' in line:  # 改为新格式
                    hits = int(line.split('Hit count:')[1].strip())
                elif 'Total bundles:' in line:  # 改为新格式
                    total_samples = int(line.split('Total bundles:')[1].strip())
                elif 'Success rate:' in line:  # 解析成功率
                    rate_str = line.split('Success rate:')[1].strip()
                    if '/' in rate_str:
                        successful_count = int(rate_str.split('/')[0])
        
        # 使用success_hit_rate作为主要指标，如果没有则使用overall_hit_rate
        main_hit_rate = success_hit_rate if success_hit_rate is not None else overall_hit_rate
        
        return {
            'Model': model_name,
            'Hit_Rate': main_hit_rate,
            'Overall_Hit_Rate': overall_hit_rate,
            'Success_Hit_Rate': success_hit_rate,
            'Hits': hits,
            'Total_Samples': total_samples,
            'Successfully_Processed': successful_count,
            'Failed': failed_count,
            'Delete_Accuracy': delete_accuracy,
            'Add_Accuracy': add_accuracy,
            'Random_Baseline': random_baseline,
            'Time_Minutes': time_cost
        }
    except Exception as e:
        print(f"WARNING: 解析结果文件 {result_file} 失败: {e}")
        return None

def run_model(dataset, strategy):
    """运行单个模型"""
    print(f"\n[INFO] 正在运行 {strategy.upper()} 模型...")
    
    script_mapping = {
        'tsf': 'tsf_update.py',
        'mean_vae': 'mean_vae_update.py', 
        'concat_vae': 'concat_vae_update.py',
        'itemknn': 'itemknn_update.py',
        'bprmf': 'bprmf_update.py'
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
    
    # 按主要Hit Rate降序排列
    df = df.sort_values('Hit_Rate', ascending=False)
    
    # 在控制台输出简洁的表格
    print("\n" + "="*80)
    print(f"Bundle Update Task - 模型对比报告")
    print(f"数据集: {dataset}")
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 打印表头
    print(f"{'模型':<12} {'Hit@1':<8} {'命中数':<8} {'总数':<8} {'成功率':<12} {'耗时(分钟)':<12}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        model = row['Model']
        hit_rate = f"{row['Hit_Rate']:.4f}" if row['Hit_Rate'] is not None else "N/A"
        hits = str(row['Hits']) if row['Hits'] is not None else "N/A"
        total = str(row['Total_Samples']) if row['Total_Samples'] is not None else "N/A"
        
        # 计算成功率显示
        if row['Successfully_Processed'] is not None and row['Total_Samples'] is not None:
            success_rate = f"{row['Successfully_Processed']}/{row['Total_Samples']}"
        else:
            success_rate = "N/A"
        
        time_cost = f"{row['Time_Minutes']:.2f}" if row['Time_Minutes'] is not None else "N/A"
        
        print(f"{model:<12} {hit_rate:<8} {hits:<8} {total:<8} {success_rate:<12} {time_cost:<12}")
    
    print("="*80)
    
    # 找出最佳模型
    best_model = df.iloc[0]
    print(f"最佳模型: {best_model['Model'].upper()}")
    print(f"最高Hit@1: {best_model['Hit_Rate']:.4f}")
    print()
    
    # 保存详细报告到文件
    report_file = f'../result/{dataset}/update_comparison_report.txt'
    os.makedirs(f'../result/{dataset}', exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"Bundle Update Task - 模型对比报告\n")
        f.write(f"数据集: {dataset}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # 使用与控制台相同的格式输出到文件
        f.write(f"{'模型':<12} {'Hit@1':<8} {'命中数':<8} {'总数':<8} {'成功率':<12} {'耗时(分钟)':<12}\n")
        f.write("-" * 80 + "\n")
        
        for _, row in df.iterrows():
            model = row['Model']
            hit_rate = f"{row['Hit_Rate']:.4f}" if row['Hit_Rate'] is not None else "N/A"
            hits = str(row['Hits']) if row['Hits'] is not None else "N/A"
            total = str(row['Total_Samples']) if row['Total_Samples'] is not None else "N/A"
            
            # 计算成功率显示
            if row['Successfully_Processed'] is not None and row['Total_Samples'] is not None:
                success_rate = f"{row['Successfully_Processed']}/{row['Total_Samples']}"
            else:
                success_rate = "N/A"
            
            time_cost = f"{row['Time_Minutes']:.2f}" if row['Time_Minutes'] is not None else "N/A"
            
            f.write(f"{model:<12} {hit_rate:<8} {hits:<8} {total:<8} {success_rate:<12} {time_cost:<12}\n")
        
        f.write("="*80 + "\n")
        f.write(f"最佳模型: {best_model['Model'].upper()}\n")
        f.write(f"最高Hit@1: {best_model['Hit_Rate']:.4f}\n\n\n")
        
        # 添加详细分析部分
        f.write("详细分析:\n")
        f.write("="*80 + "\n")
        f.write("任务说明:\n")
        f.write("Update任务 = Delete任务 + Add任务的组合\n")
        f.write("- Delete: 从当前bundle中删除最不合适的item\n")
        f.write("- Add: 从候选集中添加最合适的item\n")
        f.write("- Hit@1: 删除和添加都正确才算命中\n\n")
        
        for _, row in df.iterrows():
            f.write(f"{row['Model'].upper()}:\n")
            if row['Hit_Rate'] is not None:
                f.write(f"  - 整体命中率: {row['Hit_Rate']:.4f} ({row['Hit_Rate']*100:.2f}%)\n")
            if row['Delete_Accuracy'] is not None and row['Add_Accuracy'] is not None:
                f.write(f"  - 删除准确率: {row['Delete_Accuracy']:.4f} ({row['Delete_Accuracy']*100:.2f}%)\n")
                f.write(f"  - 添加准确率: {row['Add_Accuracy']:.4f} ({row['Add_Accuracy']*100:.2f}%)\n")
            if row['Random_Baseline'] is not None and row['Hit_Rate'] is not None:
                improvement = ((row['Hit_Rate'] - row['Random_Baseline']) / row['Random_Baseline'] * 100) if row['Random_Baseline'] > 0 else 0
                f.write(f"  - 相对于随机基准的提升: {improvement:.2f}%\n")
            if row['Time_Minutes'] is not None:
                f.write(f"  - 训练+验证时间: {row['Time_Minutes']:.2f} 分钟\n")
            f.write("\n")
    
    print(f"详细报告已保存到: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='批量运行所有update任务模型')
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
    
    print(f"批量运行Update任务模型")
    print(f"数据集: {args.dataset}")
    print(f"模型列表: {', '.join(args.models)}")
    print("="*80)
    
    # 结果文件映射
    result_file_mapping = {
        'tsf': f'../result/{args.dataset}/tsf_update_{args.dataset}_results.txt',
        'mean_vae': f'../result/{args.dataset}/mean_vae_update_{args.dataset}_results.txt',
        'concat_vae': f'../result/{args.dataset}/concat_vae_update_{args.dataset}_results.txt',
        'itemknn': f'../result/{args.dataset}/itemknn_update_{args.dataset}_results.txt',
        'bprmf': f'../result/{args.dataset}/bprmf_update_{args.dataset}_results.txt'
    }
    
    total_start_time = time.time()
    results = []
    
    for model in args.models:
        result_file = result_file_mapping[model]
        
        # 检查是否跳过已存在的结果
        if args.skip_existing and os.path.exists(result_file):
            print(f"跳过 {model.upper()}（结果文件已存在）")
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
    
    print(f"\n所有模型运行完成，总耗时: {total_time:.2f} 分钟")
    
    # 生成对比报告
    generate_comparison_report(args.dataset, results)
    
    print("\n批量任务完成！")

if __name__ == '__main__':
    main()