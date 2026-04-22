'''
Description: BPRMF模型用于bundle add任务 - 完整版（训练+验证）
Author: anyiran
Date: 2025-09-17
LastEditTime: 2025-09-21 14:42:10
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
test_data_path = f'../../testdata/{args.d}/add_test.txt'

# -------------------------------
# Step 1: 修正后的数据划分逻辑
# -------------------------------
df = pd.read_csv(bundle_item_path, sep=',')
df.columns = ['bundle_id', 'item_id']

# 获取所有bundle和item
all_items = df['item_id'].unique().tolist()

# 1. 先筛选出item数量>=3的有效bundle
bundle_to_items = df.groupby('bundle_id')['item_id'].apply(list).to_dict()
valid_bundles = [bid for bid, items in bundle_to_items.items() if len(items) >= 3]

print(f"总bundle数量: {len(bundle_to_items)}")
print(f"有效bundle数量 (>=3 items): {len(valid_bundles)}")

# 2. 读取测试集的bundle_id（从add_test.txt获取）
test_bundle_ids = set()
if os.path.exists(test_data_path):
    with open(test_data_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                test_bundle_ids.add(int(parts[0]))

print(f"测试集bundle数量: {len(test_bundle_ids)}")

# 3. 训练集 = 有效bundle - 测试集bundle
train_bundles = [bid for bid in valid_bundles if bid not in test_bundle_ids]

print(f"训练集bundle数量: {len(train_bundles)}")
print(f"数据一致性检查: 测试集是否都在有效bundle中: {test_bundle_ids.issubset(set(valid_bundles))}")

# 4. 构建训练数据
train_df = df[df['bundle_id'].isin(train_bundles)]
bundle2items = train_df.groupby('bundle_id')['item_id'].apply(list).to_dict()

print(f"训练集覆盖的bundle数量: {len(bundle2items)}")
print(f"训练集总交互数量: {len(train_df)}")

# 验证数据泄露
overlap = set(train_bundles).intersection(test_bundle_ids)
print(f"训练集与测试集重叠bundle数量: {len(overlap)} (应该为0)")

if len(overlap) > 0:
    print(f"警告：发现数据泄露！重叠的bundle: {list(overlap)[:5]}...")
else:
    print("数据划分正确，无泄露")

start_time = time.time()

# -------------------------------
# Step 2: 构造三元组
# -------------------------------
triplets = []
for bundle, items in bundle2items.items():
    for pos_item in items:
        neg_item = random.choice([i for i in all_items if i not in items])
        triplets.append((bundle, pos_item, neg_item))

bundle2id = {b: idx for idx, b in enumerate(sorted(train_df['bundle_id'].unique()))}
item2id = {i: idx for idx, i in enumerate(sorted(df['item_id'].unique()))}
id2item = {v: k for k, v in item2id.items()}

# -------------------------------
# Step 3: 构造 Dataset
# -------------------------------
class BPRDataset(Dataset):
    def __init__(self, triplets):
        self.data = triplets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        b, pos, neg = self.data[idx]
        return bundle2id[b], item2id[pos], item2id[neg]

dataset = BPRDataset(triplets)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# -------------------------------
# Step 4: BPRMF 模型训练
# -------------------------------
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

model = BPRMF(len(bundle2id), len(item2id), dim=64)
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

# -------------------------------
# Step 5: 加载候选集并评估
# -------------------------------
print("开始评估 bundle 补全任务 ...")

# 获取训练好的item embeddings
item_embeddings = model.item_emb.weight.detach().cpu().numpy()

# 加载全量 bundle-item 字典
bundle_item_df = pd.read_csv(bundle_item_path, sep=',')
bundle_item_df.columns = ['bundle_id', 'item_id']  # 确保列名一致
bundle_dict = bundle_item_df.groupby('bundle_id')['item_id'].apply(list).to_dict()

# 加载add_test.txt文件
add_test_data = []
with open(test_data_path, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) != 3:
            continue
        bundle_id = int(parts[0])
        bundle_items = list(map(int, parts[1].split()))
        candidate_ids = list(map(int, parts[2].split()))
        add_test_data.append((bundle_id, bundle_items, candidate_ids))

print(f"读取到 {len(add_test_data)} 个测试样本")

# 数据泄露检查：统计测试bundle与训练bundle的重叠情况
test_bundle_ids_list = [bundle_id for bundle_id, _, _ in add_test_data]
train_set = set(train_bundles)
test_set = set(test_bundle_ids_list)
overlap = train_set.intersection(test_set)

print(f"数据泄露检查:")
print(f"  训练bundle数量: {len(train_set)}")
print(f"  测试bundle数量: {len(test_set)}")
print(f"  重叠bundle数量: {len(overlap)} ({len(overlap)/len(test_set)*100:.1f}%)")

if len(overlap) > 0:
    print(f"  检测到数据泄露：{len(overlap)}个测试bundle在训练集中出现过")
    print(f"  处理策略：保留bundle，但移除候选集中训练时见过的items")
else:
    print(f"  无数据泄露：测试bundle与训练bundle完全分离")

# -------------------------------
# Step 6: Eval: Hit@K and NDCG@K（修正版）
# -------------------------------
def hit_at_k(rank_list, ground_truth, k):
    return int(ground_truth in rank_list[:k])

def ndcg_at_k(rank_list, ground_truth, k):
    if ground_truth in rank_list[:k]:
        rank = rank_list.index(ground_truth)
        return 1.0 / np.log2(rank + 2)
    return 0.0

top_k = 1

# 修正：使用总样本数作为分母，跳过的样本计为0分
total_samples = len(add_test_data)
hit_count = 0  # 成功hit的总数
ndcg_sum = 0.0  # NDCG分数总和

# 统计各种情况的数量
leaked_count = 0  # 需要清理候选集的bundle数量
gt_leaked_count = 0  # Ground Truth在训练中见过而跳过的bundle数量
filtered_too_small_count = 0  # 候选集过小而跳过的bundle数量
missing_bundle_count = 0  # 在bundle_dict中找不到的bundle数量
missing_gt_count = 0  # 找不到ground truth的bundle数量
missing_mapping_count = 0  # 缺少item mapping的bundle数量
successful_count = 0  # 成功处理的bundle数量

item_id_map = item2id
reverse_item_id_map = {v: k for k, v in item_id_map.items()}

print(f"开始逐个处理 {total_samples} 个测试样本...")

for idx, (bundle_id, bundle_items, candidate_ids) in enumerate(add_test_data):
    
    # 初始化当前样本的得分（默认为0，即miss）
    current_hit = 0
    current_ndcg = 0.0
    skip_reason = ""
    
    # 检查1: bundle是否在完整数据中存在
    if bundle_id not in bundle_dict:
        missing_bundle_count += 1
        skip_reason = f"Bundle {bundle_id} 不在完整数据中"
    else:
        complete_bundle = bundle_dict[bundle_id]
        
        # 找到ground truth: 在完整bundle中但不在当前bundle_items中的item
        ground_truth_items = [item for item in complete_bundle if item not in bundle_items]
        
        # 找到在候选集中的ground truth items
        ground_truth_in_candidates = [item for item in ground_truth_items if item in candidate_ids]
        
        # 检查2: 是否有有效的ground truth
        if not ground_truth_in_candidates:
            missing_gt_count += 1
            skip_reason = f"Bundle {bundle_id} 没有有效的ground truth"
        else:
            # 选择第一个作为ground truth
            ground_truth_item = ground_truth_in_candidates[0]
            
            # 检查3: 处理数据泄露情况
            if bundle_id in train_bundles:
                leaked_count += 1
                # 获取训练时该bundle包含的所有items
                seen_items = set(bundle2items.get(bundle_id, []))
                
                # 从候选集中移除训练时见过的items
                original_candidates = candidate_ids.copy()
                candidate_ids = [item for item in candidate_ids if item not in seen_items]
                
                # 检查3a: ground truth是否也被移除了
                if ground_truth_item in seen_items:
                    gt_leaked_count += 1
                    skip_reason = f"Bundle {bundle_id} 的ground truth在训练时已见过"
                # 检查3b: 候选集是否太小
                elif len(candidate_ids) < 5:  # 至少需要5个候选item
                    filtered_too_small_count += 1
                    skip_reason = f"Bundle {bundle_id} 过滤后候选集太小({len(candidate_ids)})"
            
            # 检查4: ground truth是否仍在处理后的候选集中
            if not skip_reason and ground_truth_item not in candidate_ids:
                missing_gt_count += 1
                skip_reason = f"Bundle {bundle_id} ground truth不在候选集中"
            
            # 检查5: 是否有有效的item mapping
            if not skip_reason:
                try:
                    # 使用bundle_items作为context计算embedding
                    context_indices = [item_id_map[i] for i in bundle_items if i in item_id_map]
                    # 对处理后的候选items进行预测
                    candidate_indices = [item_id_map[i] for i in candidate_ids if i in item_id_map]
                    ground_truth_idx = item_id_map[ground_truth_item]
                    
                    if not context_indices or not candidate_indices:
                        missing_mapping_count += 1
                        skip_reason = f"Bundle {bundle_id} 缺少有效的item mapping"
                except KeyError:
                    missing_mapping_count += 1
                    skip_reason = f"Bundle {bundle_id} KeyError in item mapping"
            
            # 如果通过所有检查，进行实际预测
            if not skip_reason:
                successful_count += 1
                
                # 计算bundle_embedding：bundle中item的embedding平均值
                bundle_embedding = item_embeddings[context_indices].mean(axis=0)

                # 对候选集中的每个item计算与bundle_embedding的点积
                scores = []
                for cand_idx in candidate_indices:
                    cand_item_id = reverse_item_id_map[cand_idx]
                    
                    # 跳过context中已有的items（避免数据泄露）
                    if cand_item_id in bundle_items:
                        continue
                        
                    cand_item_embedding = item_embeddings[cand_idx]
                    
                    # 计算点积作为相似度分数
                    dot_product = np.dot(bundle_embedding, cand_item_embedding)
                    scores.append((cand_item_id, dot_product))

                # 按分数排序
                ranked_items = sorted(scores, key=lambda x: -x[1])
                ranked_ids = [item_id for item_id, _ in ranked_items]

                # 计算指标
                current_hit = hit_at_k(ranked_ids, ground_truth_item, top_k)
                current_ndcg = ndcg_at_k(ranked_ids, ground_truth_item, top_k)
    
    # 累加到总分数中（跳过的样本贡献0分）
    hit_count += current_hit
    ndcg_sum += current_ndcg

# 计算总训练和验证时间
end_time = time.time()
total_minutes = (end_time - start_time) / 60

# -------------------------------
# Step 7: 输出结果并保存到文件
# -------------------------------
success_rate = f"{successful_count}/{len(add_test_data)}"
hit_rate = hit_count / len(add_test_data) if len(add_test_data) > 0 else 0

result_text = f"""BPRMF Add Task Results - {args.d}
Hit@1: {hit_rate:.4f}
Hit count: {hit_count}
Total bundles: {len(add_test_data)}
Success rate: {success_rate}
Time: {total_minutes:.2f} minutes"""

print("\n" + "="*50)
print(result_text)
print("="*50)

# 保存结果到文件
result_file = os.path.join(result_dir, f'bprmf_add_{args.d}_results.txt')
with open(result_file, 'w') as f:
    f.write(result_text)

print(f"结果已保存到: {result_file}")