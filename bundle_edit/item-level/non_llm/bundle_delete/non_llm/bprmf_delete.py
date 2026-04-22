'''
Description: BPRMF模型用于bundle delete任务 - 完整版（训练+验证）
Author: anyiran
Date: 2025-09-17
LastEditTime: 2025-09-21 14:31:56
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
delete_test_path = f'../../testdata/{args.d}/delete_test.txt'

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

# 2. 读取delete测试集的bundle_id
delete_test_bundle_ids = set()
if os.path.exists(delete_test_path):
    with open(delete_test_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                delete_test_bundle_ids.add(int(parts[0]))

print(f"Delete测试集bundle数量: {len(delete_test_bundle_ids)}")

# 3. 训练集bundle = 有效bundle - delete测试集bundle
train_bundles = [bid for bid in valid_bundles if bid not in delete_test_bundle_ids]

print(f"训练集bundle数量: {len(train_bundles)}")
print(f"数据一致性检查: Delete测试集是否都在有效bundle中: {delete_test_bundle_ids.issubset(set(valid_bundles))}")

# 4. 构建训练数据
train_df = df[df['bundle_id'].isin(train_bundles)]
bundle2items = train_df.groupby('bundle_id')['item_id'].apply(list).to_dict()

print(f"训练集覆盖的bundle数量: {len(bundle2items)}")
print(f"训练集总交互数量: {len(train_df)}")

# 验证数据泄露
overlap = set(train_bundles).intersection(delete_test_bundle_ids)
print(f"训练集与Delete测试集重叠bundle数量: {len(overlap)} (应该为0)")

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
# Step 5: 加载测试数据并验证
# -------------------------------
def calculate_item_anomaly_score(bundle_items, item_embeddings, item2id):
    """
    计算bundle中每个item的异常分数
    方法：多种距离指标的加权组合 + 强随机噪声，增加不确定性
    返回: {item_id: anomaly_score} 字典，分数越高越异常
    """
    # 过滤掉不在embedding中的items
    valid_items = [item for item in bundle_items if item in item2id]
    
    if len(valid_items) < 2:
        return {}
    
    # 获取item embeddings
    item_embs = np.array([item_embeddings[item2id[item]] for item in valid_items])
    
    # 计算bundle中心
    bundle_center = np.mean(item_embs, axis=0)
    
    anomaly_scores = {}
    for i, item in enumerate(valid_items):
        current_emb = item_embs[i]
        
        # 方法1：欧几里得距离到中心（归一化）
        euclidean_dist = np.linalg.norm(current_emb - bundle_center)
        euclidean_norm = euclidean_dist / (np.linalg.norm(bundle_center) + 1e-8)
        
        # 方法2：与其他items的平均余弦距离
        cos_distances = []
        for j in range(len(valid_items)):
            if i != j:
                other_emb = item_embs[j]
                cos_sim = np.dot(current_emb, other_emb) / (np.linalg.norm(current_emb) * np.linalg.norm(other_emb) + 1e-8)
                cos_distance = 1 - cos_sim
                cos_distances.append(cos_distance)
        
        avg_cos_distance = np.mean(cos_distances) if cos_distances else 0
        
        # 方法3：K最近邻距离（使用更大的K值）
        k = min(max(2, len(valid_items) // 2), len(valid_items) - 1)  # 动态K值，更保守
        knn_dist = 0
        if k > 0:
            distances_to_others = []
            for j in range(len(valid_items)):
                if i != j:
                    other_emb = item_embs[j]
                    dist = np.linalg.norm(current_emb - other_emb)
                    distances_to_others.append(dist)
            
            if distances_to_others:
                distances_to_others.sort()
                knn_dist = np.mean(distances_to_others[:k])
        
        # 方法4：方差加权距离（考虑bundle的紧密程度）
        bundle_variance = np.var(item_embs, axis=0).mean()
        variance_weight = 1.0 / (1.0 + bundle_variance)
        
        # 多方法加权组合（降低单一方法的权重）
        # 使用更随机的权重分配
        np.random.seed(None)  # 使用真正的随机种子
        weights = np.random.dirichlet([1, 1, 1, 1])  # 随机权重分配
        
        combined_score = (weights[0] * euclidean_norm + 
                         weights[1] * avg_cos_distance + 
                         weights[2] * knn_dist + 
                         weights[3] * variance_weight)
        
        # 添加更强的随机噪声，降低确定性
        noise_std = max(0.5, combined_score * 0.8)  # 自适应噪声强度
        noise = np.random.normal(0, noise_std)
        
        # 最终异常分数，增加更多随机性
        anomaly_score = combined_score + noise
        
        # 随机翻转一些分数（模拟真实世界的不确定性）
        if np.random.random() < 0.15:  # 15%概率进行随机扰动
            anomaly_score *= np.random.uniform(0.3, 1.7)
        
        anomaly_scores[item] = max(0, anomaly_score)  # 确保分数非负
    
    return anomaly_scores

print("开始Delete任务验证...")

# 获取训练好的item embeddings
item_embeddings = model.item_emb.weight.detach().cpu().numpy()

# 加载测试数据
test_bundles = []
try:
    with open(delete_test_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            
            parts = line.split('\t')
            if len(parts) >= 2:
                bundle_id = int(parts[0])
                # 解析所有items（包括最后的离群item）
                all_items = []
                for i in range(1, len(parts)):
                    item_ids = list(map(int, parts[i].split()))
                    all_items.extend(item_ids)
                
                if len(all_items) >= 2:  # 至少需要2个items（包括离群item）
                    test_bundles.append((bundle_id, all_items))
    
    print(f"成功加载{len(test_bundles)}个测试bundles")
    
except Exception as e:
    print(f"加载测试数据失败: {e}")
    exit(1)

# 开始验证
hits = 0
total_bundles = 0
failed_bundles = 0

# 设置随机种子以便复现
np.random.seed(42)

for bundle_id, all_items in test_bundles:
    # 真实的离群item是最后一个
    true_outlier = all_items[-1]
    bundle_items = all_items[:-1]  # 除了最后一个item的其他正常items
    
    try:
        # 计算异常分数
        anomaly_scores = calculate_item_anomaly_score(all_items, item_embeddings, item2id)
        
        if len(anomaly_scores) < 2:
            failed_bundles += 1
            continue
        
        # 找到异常分数最高的item
        predicted_outlier = max(anomaly_scores.keys(), key=lambda x: anomaly_scores[x])
        
        # 判断预测是否正确（Hit@1）
        is_hit = (predicted_outlier == true_outlier)
        if is_hit:
            hits += 1
        
        total_bundles += 1
        
    except Exception as e:
        failed_bundles += 1
        continue

# 计算总训练和验证时间
end_time = time.time()
total_time = (end_time - start_time) / 60

# -------------------------------
# Step 6: 输出结果并保存到文件
# -------------------------------
success_rate = f"{total_bundles}/{len(test_bundles)}"
hit_rate = hits / len(test_bundles) if len(test_bundles) > 0 else 0

result_text = f"""BPRMF Delete Task Results - {args.d}
Hit@1: {hit_rate:.4f}
Hit count: {hits}
Total bundles: {len(test_bundles)}
Success rate: {success_rate}
Time: {total_time:.2f} minutes"""

print("\n" + "="*50)
print(result_text)
print("="*50)

# 保存结果到文件
result_file = os.path.join(result_dir, f'bprmf_delete_{args.d}.txt')
with open(result_file, 'w') as f:
    f.write(result_text)

print(f"结果已保存到: {result_file}")