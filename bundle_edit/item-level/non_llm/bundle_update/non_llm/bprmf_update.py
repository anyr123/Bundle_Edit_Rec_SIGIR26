'''
Description: BPRMF模型用于bundle update任务 - 两阶段实现
第一阶段DELETE：找出bundle中最不合适的item
第二阶段ADD：在候选集中选择最合适的item
Author: anyiran
Date: 2025-09-21
LastEditTime: 2025-09-22 14:47:05
'''

import pandas as pd
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle
import os
import argparse
from collections import defaultdict
from numpy.linalg import norm
import time

# -------------------------------
# 命令行参数解析
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default='clothing', 
                    choices=['food', 'clothing', 'electronic'],
                    help='选择数据集: food, clothing, electronic')

args = parser.parse_args()

# 构建数据集路径
data_path = '../../dataset'
dataset_path = os.path.join(data_path, args.d).replace('\\', '/')
bundle_item_path = os.path.join(dataset_path, 'bundle_item.csv').replace('\\', '/')

print(f"使用数据集: {args.d}")
print(f"数据集路径: {dataset_path}")

# 确保结果目录存在
result_dir = f'../result/{args.d}'
os.makedirs(result_dir, exist_ok=True)

# 测试数据路径
test_data_path = f'../../testdata/{args.d}/update_test.txt'

# -------------------------------
# Step 1: 数据加载和处理
# -------------------------------
print("加载数据和分析测试格式...")

# 读取完整的bundle-item数据
df = pd.read_csv(bundle_item_path, sep=',')
df.columns = ['bundle_id', 'item_id']

# 构建完整的bundle字典用于获取ground truth
complete_bundle_dict = df.groupby('bundle_id')['item_id'].apply(list).to_dict()

# 分析update测试数据格式
test_data_analysis = []
gt_items = set()

with open(test_data_path, 'r') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('\t')
        if len(parts) >= 3:
            bundle_id = int(parts[0])
            current_bundle = list(map(int, parts[1].split()))
            candidate_ids = list(map(int, parts[2].split()))
            
            # 获取完整的ground truth bundle
            if bundle_id in complete_bundle_dict:
                ground_truth_bundle = complete_bundle_dict[bundle_id]
                
                # 找到需要删除的items（在current中但不在ground truth中）
                items_to_remove = [item for item in current_bundle if item not in ground_truth_bundle]
                # 找到需要添加的items（在ground truth中但不在current中，且在候选集中）
                items_to_add = [item for item in ground_truth_bundle 
                               if item not in current_bundle and item in candidate_ids]
                
                test_data_analysis.append({
                    'bundle_id': bundle_id,
                    'current_bundle': current_bundle,
                    'candidate_ids': candidate_ids,
                    'ground_truth_bundle': ground_truth_bundle,
                    'items_to_remove': items_to_remove,
                    'items_to_add': items_to_add
                })
                
                # 收集ground truth items（用于避免数据泄露）
                gt_items.update(items_to_remove)
                gt_items.update(items_to_add)
                
                # 打印前5个样本分析
                if line_num <= 5:
                    print(f"样本 {line_num}:")
                    print(f"  Bundle ID: {bundle_id}")
                    print(f"  当前bundle: {current_bundle}")
                    print(f"  候选集数量: {len(candidate_ids)}")
                    print(f"  完整bundle: {ground_truth_bundle}")
                    print(f"  需要删除: {items_to_remove}")
                    print(f"  需要添加: {items_to_add}")
                    print()

print(f"总测试样本数: {len(test_data_analysis)}")
print(f"Ground Truth items数量: {len(gt_items)}")

# -------------------------------
# Step 2: 构建训练数据（避免数据泄露）
# -------------------------------
# 获取所有bundle和item
all_items = df['item_id'].unique().tolist()

# 筛选出有效bundle（至少3个items）
bundle_to_items = df.groupby('bundle_id')['item_id'].apply(list).to_dict()
valid_bundles = [bid for bid, items in bundle_to_items.items() if len(items) >= 3]

# 获取测试bundle IDs
test_bundle_ids = set([sample['bundle_id'] for sample in test_data_analysis])

# 训练集 = 有效bundle - 测试集bundle
train_bundles = [bid for bid in valid_bundles if bid not in test_bundle_ids]

print(f"数据划分统计:")
print(f"  总bundle数量: {len(bundle_to_items)}")
print(f"  有效bundle数量: {len(valid_bundles)}")
print(f"  测试bundle数量: {len(test_bundle_ids)}")
print(f"  训练bundle数量: {len(train_bundles)}")

# 构建训练数据
train_df = df[df['bundle_id'].isin(train_bundles)]
bundle2items = train_df.groupby('bundle_id')['item_id'].apply(list).to_dict()

# 验证数据泄露
overlap = set(train_bundles).intersection(test_bundle_ids)
print(f"训练集与测试集重叠: {len(overlap)} (应该为0)")

start_time = time.time()

# -------------------------------
# Step 3: 构建三元组和训练BPRMF
# -------------------------------
triplets = []
for bundle, items in bundle2items.items():
    for pos_item in items:
        neg_item = random.choice([i for i in all_items if i not in items])
        triplets.append((bundle, pos_item, neg_item))

bundle2id = {b: idx for idx, b in enumerate(sorted(train_df['bundle_id'].unique()))}
item2id = {i: idx for idx, i in enumerate(sorted(df['item_id'].unique()))}
id2item = {v: k for k, v in item2id.items()}

print(f"生成的三元组数量: {len(triplets)}")
print(f"Bundle映射数量: {len(bundle2id)}")
print(f"Item映射数量: {len(item2id)}")

class BPRDataset(Dataset):
    def __init__(self, triplets):
        self.data = triplets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        b, pos, neg = self.data[idx]
        return bundle2id[b], item2id[pos], item2id[neg]

class BPRMF(nn.Module):
    def __init__(self, n_bundles, n_items, dim=64):
        super().__init__()
        self.bundle_emb = nn.Embedding(n_bundles, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.xavier_uniform_(self.bundle_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, bundle, pos_item, neg_item):
        b = self.bundle_emb(bundle)
        pos = self.item_emb(pos_item)
        neg = self.item_emb(neg_item)
        pos_score = (b * pos).sum(dim=1)
        neg_score = (b * neg).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()
        return loss

dataset = BPRDataset(triplets)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

model = BPRMF(len(bundle2id), len(item2id), dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("开始训练 BPRMF...")
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        b, pos, neg = [x for x in batch]
        loss = model(b, pos, neg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

print("训练完成！")

# 获取训练好的item embeddings
item_embeddings = model.item_emb.weight.detach().cpu().numpy()
print(f"Item embedding形状: {item_embeddings.shape}")

# -------------------------------
# Step 4: DELETE阶段 - 离群item检测函数
# -------------------------------
def detect_outlier_item(bundle_items, item_embeddings, item2id):
    """
    检测bundle中的离群item（最不合适的item）
    方法：计算每个item与其他items的平均相似度，选择相似度最低的
    """
    valid_items = [item for item in bundle_items if item in item2id]
    
    if len(valid_items) < 2:
        return None
    
    anomaly_scores = {}
    
    for target_item in valid_items:
        target_idx = item2id[target_item]
        target_emb = item_embeddings[target_idx].reshape(1, -1)
        
        # 计算与bundle中其他items的相似度
        other_items = [item for item in valid_items if item != target_item]
        if not other_items:
            anomaly_scores[target_item] = 0.5
            continue
        
        other_indices = [item2id[item] for item in other_items]
        other_embs = item_embeddings[other_indices]
        
        # 计算余弦相似度
        similarities = cosine_similarity(target_emb, other_embs).flatten()
        
        # 异常分数 = 1 - 平均相似度（越不相似越异常）
        avg_similarity = similarities.mean()
        anomaly_score = 1 - avg_similarity
        
        anomaly_scores[target_item] = anomaly_score
    
    # 返回异常分数最高的item（最不合适的）
    return max(anomaly_scores.keys(), key=lambda x: anomaly_scores[x])

# -------------------------------
# Step 5: ADD阶段 - 候选item选择函数
# -------------------------------
def select_best_candidate(bundle_items, candidate_ids, item_embeddings, item2id):
    """
    从候选集中选择最适合bundle的item
    方法：计算候选item与bundle的相似度，选择相似度最高的
    """
    valid_bundle_items = [item for item in bundle_items if item in item2id]
    valid_candidates = [item for item in candidate_ids if item in item2id and item not in bundle_items]
    
    if not valid_bundle_items or not valid_candidates:
        # 如果没有有效的选择，返回第一个候选项
        return candidate_ids[0] if candidate_ids else None
    
    # 计算bundle的中心embedding
    bundle_indices = [item2id[item] for item in valid_bundle_items]
    bundle_embs = item_embeddings[bundle_indices]
    bundle_center = bundle_embs.mean(axis=0).reshape(1, -1)
    
    best_item = None
    best_score = -np.inf
    
    for candidate in valid_candidates:
        candidate_idx = item2id[candidate]
        candidate_emb = item_embeddings[candidate_idx].reshape(1, -1)
        
        # 方法1：与bundle中心的余弦相似度
        center_similarity = cosine_similarity(candidate_emb, bundle_center)[0, 0]
        
        # 方法2：与bundle中各item的平均相似度
        item_similarities = cosine_similarity(candidate_emb, bundle_embs).flatten()
        avg_item_similarity = item_similarities.mean()
        
        # 方法3：与bundle中各item的最大相似度
        max_item_similarity = item_similarities.max()
        
        # 综合评分：加权组合多种相似度
        combined_score = (0.5 * center_similarity + 
                         0.3 * avg_item_similarity + 
                         0.2 * max_item_similarity)
        
        if combined_score > best_score:
            best_score = combined_score
            best_item = candidate
    
    return best_item if best_item else candidate_ids[0]

# -------------------------------
# Step 6: 评估Bundle Update任务
# -------------------------------
print("开始评估Bundle Update任务...")

total_samples = len(test_data_analysis)
hits = 0
successful_count = 0
failed_count = 0

delete_hits = 0
add_hits = 0

results = []

print(f"开始处理 {total_samples} 个测试样本...")

for idx, sample in enumerate(test_data_analysis):
    try:
        bundle_id = sample['bundle_id']
        current_bundle = sample['current_bundle']
        candidate_ids = sample['candidate_ids']
        ground_truth_bundle = sample['ground_truth_bundle']
        items_to_remove = sample['items_to_remove']
        items_to_add = sample['items_to_add']
        
        # 如果没有需要更新的内容，跳过
        if not items_to_remove and not items_to_add:
            failed_count += 1
            continue
        
        # Phase 1: DELETE阶段 - 检测需要删除的item
        predicted_remove = None
        if len(current_bundle) > 1 and items_to_remove:
            predicted_remove = detect_outlier_item(current_bundle, item_embeddings, item2id)
        
        # Phase 2: ADD阶段 - 选择最佳候选item
        # 先从current bundle中移除预测的删除item
        updated_bundle = [item for item in current_bundle if item != predicted_remove] if predicted_remove else current_bundle
        
        predicted_add = select_best_candidate(updated_bundle, candidate_ids, item_embeddings, item2id)
        
        # 评估结果
        remove_hit = predicted_remove in items_to_remove if predicted_remove and items_to_remove else (not items_to_remove)
        add_hit = predicted_add in items_to_add if predicted_add and items_to_add else (not items_to_add)
        
        # 整体命中：删除和添加都正确
        overall_hit = remove_hit and add_hit
        
        if overall_hit:
            hits += 1
        if remove_hit:
            delete_hits += 1
        if add_hit:
            add_hits += 1
        
        successful_count += 1
        
        # 保存详细结果
        results.append({
            'bundle_id': bundle_id,
            'current_bundle': current_bundle,
            'ground_truth_bundle': ground_truth_bundle,
            'items_to_remove': items_to_remove,
            'items_to_add': items_to_add,
            'predicted_remove': predicted_remove,
            'predicted_add': predicted_add,
            'remove_hit': remove_hit,
            'add_hit': add_hit,
            'overall_hit': overall_hit
        })
        
        # 打印前10个结果
        if successful_count <= 10:
            print(f"\nBundle {bundle_id}:")
            print(f"  当前bundle: {current_bundle}")
            print(f"  完整bundle: {ground_truth_bundle}")
            print(f"  应删除: {items_to_remove}")
            print(f"  应添加: {items_to_add}")
            print(f"  预测删除: {predicted_remove} {'✅' if remove_hit else '❌'}")
            print(f"  预测添加: {predicted_add} {'✅' if add_hit else '❌'}")
            print(f"  整体结果: {'✅正确' if overall_hit else '❌错误'}")
        
    except Exception as e:
        print(f"Bundle {bundle_id} 处理失败: {e}")
        failed_count += 1
        continue

# 计算总时间
end_time = time.time()
total_minutes = (end_time - start_time) / 60

# -------------------------------
# Step 7: 输出结果并保存
# -------------------------------
print("\n" + "="*80)
print("BPRMF Bundle Update任务验证结果")
print("="*80)

if successful_count > 0:
    overall_hit_rate = hits / total_samples
    success_hit_rate = hits / successful_count
    delete_accuracy = delete_hits / successful_count
    add_accuracy = add_hits / successful_count
    
    # 随机基准计算
    random_hits = 0
    np.random.seed(42)
    for result in results:
        # 随机删除一个item
        random_remove = np.random.choice(result['current_bundle']) if result['current_bundle'] else None
        # 随机选择一个候选item
        available_candidates = [item for item in result['ground_truth_bundle'] if item not in result['current_bundle']]
        random_add = np.random.choice(available_candidates) if available_candidates else None
        
        random_remove_hit = random_remove in result['items_to_remove'] if random_remove and result['items_to_remove'] else (not result['items_to_remove'])
        random_add_hit = random_add in result['items_to_add'] if random_add and result['items_to_add'] else (not result['items_to_add'])
        
        if random_remove_hit and random_add_hit:
            random_hits += 1
    
    random_hit_rate = random_hits / successful_count
    
    result_text = f"""BPRMF Bundle Update Task Results - {args.d}
Dataset: {args.d}
Total samples: {total_samples}
Successfully processed: {successful_count}
Failed: {failed_count}
Overall hits: {hits}

Performance Metrics:
Overall Hit Rate: {overall_hit_rate:.4f} ({overall_hit_rate*100:.2f}%)
Success Hit Rate: {success_hit_rate:.4f} ({success_hit_rate*100:.2f}%)
Delete Accuracy: {delete_accuracy:.4f} ({delete_accuracy*100:.2f}%)
Add Accuracy: {add_accuracy:.4f} ({add_accuracy*100:.2f}%)
Processing Rate: {successful_count/total_samples*100:.1f}%

Baseline Comparison:
Random baseline: {random_hit_rate:.4f} ({random_hit_rate*100:.2f}%)
Improvement: {((success_hit_rate - random_hit_rate) / random_hit_rate * 100 if random_hit_rate > 0 else 0):.2f}%

Training Details:
Training + Evaluation time: {total_minutes:.2f} minutes
Embedding dimension: 64
Training epochs: {epochs}
Batch size: 128
Learning rate: 0.01
Trained bundles: {len(bundle2id)}
Trained items: {len(item2id)}

Task Description:
Update = Delete + Add (两阶段都正确才算命中)
Delete: 基于embedding相似度检测最不合适的item
Add: 基于embedding相似度选择最适合的候选item
Model: BPRMF (Bayesian Personalized Ranking Matrix Factorization)"""

    print(result_text)
    
    # 保存结果到文件
    result_file = os.path.join(result_dir, f'bprmf_update_{args.d}_results.txt')
    with open(result_file, 'w') as f:
        f.write(result_text)
    
    # # 保存详细结果
    # detailed_results_file = os.path.join(result_dir, f'bprmf_update_{args.d}_detailed.txt')
    # with open(detailed_results_file, 'w') as f:
    #     f.write("Bundle_ID\tCurrent_Bundle\tGround_Truth\tItems_To_Remove\tItems_To_Add\tPredicted_Remove\tPredicted_Add\tRemove_Hit\tAdd_Hit\tOverall_Hit\n")
    #     for result in results:
    #         f.write(f"{result['bundle_id']}\t")
    #         f.write(f"{' '.join(map(str, result['current_bundle']))}\t")
    #         f.write(f"{' '.join(map(str, result['ground_truth_bundle']))}\t")
    #         f.write(f"{' '.join(map(str, result['items_to_remove']))}\t")
    #         f.write(f"{' '.join(map(str, result['items_to_add']))}\t")
    #         f.write(f"{result['predicted_remove']}\t")
    #         f.write(f"{result['predicted_add']}\t")
    #         f.write(f"{result['remove_hit']}\t")
    #         f.write(f"{result['add_hit']}\t")
    #         f.write(f"{result['overall_hit']}\n")
    
    # print(f"结果已保存到: {result_file}")
    # print(f"详细结果已保存到: {detailed_results_file}")
    
else:
    print("没有成功处理任何样本，请检查数据格式")

print("="*80)