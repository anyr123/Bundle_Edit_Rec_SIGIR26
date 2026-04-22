'''
Description: ItemKNN模型用于bundle add任务 - 完整版（训练+验证）
Author: anyiran
Date: 2025-09-18
LastEditTime: 2025-09-21 14:51:20
'''

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
import argparse
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
user_bundle_file = os.path.join(dataset_path, 'user_bundle.csv').replace('\\', '/')
user_item_file = os.path.join(dataset_path, 'user_item.csv').replace('\\', '/')
test_data_path = f'../../testdata/{args.d}/add_test.txt'

print(f"使用数据集: {args.d}")
print(f"数据集路径: {dataset_path}")

# 确保结果目录存在
result_dir = f'../result/{args.d}'
os.makedirs(result_dir, exist_ok=True)

EMBED_DIM = 64
BATCH_SIZE = 1024
EPOCHS = 10
LR = 0.01
TOP_K = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Step 1: 先读取测试数据，收集所有涉及的items（避免数据泄露）
# -------------------------------
print("首先读取测试数据，收集所有涉及的items...")
test_items = set()
with open(test_data_path, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) != 3:
            continue
        bundle_items = list(map(int, parts[1].split()))
        candidate_ids = list(map(int, parts[2].split()))
        test_items.update(bundle_items)
        test_items.update(candidate_ids)

# 读取bundle_item数据获取完整的bundle信息
bundle_item_df = pd.read_csv(bundle_item_path)
bundle_item_df.columns = ['bundle_id', 'item_id']
bundle_dict = bundle_item_df.groupby('bundle_id')['item_id'].apply(list).to_dict()

# 收集测试bundle中的ground truth items（只排除ground truth，允许其他items）
gt_items = set()
with open(test_data_path, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) != 3:
            continue
        bundle_id = int(parts[0])
        if bundle_id in bundle_dict:
            # 找到ground truth: 在完整bundle中但不在当前bundle_items中的item
            bundle_items = list(map(int, parts[1].split()))
            original_bundle = bundle_dict[bundle_id]
            ground_truth_items = [item for item in original_bundle if item not in bundle_items]
            gt_items.update(ground_truth_items)

print(f"Ground Truth items: {len(gt_items)}")

# 更新test_items，只包含ground truth，允许其他items在训练中出现
test_items = gt_items

print(f"测试数据统计:")
print(f"  测试中涉及的items数量: {len(test_items)}")

start_time = time.time()

# -------------------------------
# Step 2: 加载并预处理用户-物品交互数据（只排除ground truth items）
# -------------------------------
print("加载user-item交互数据（只排除ground truth items）...")
df = pd.read_csv(user_item_file, sep=',')
df.columns = ['user_id', 'item_id', 'timestamp']
# 只保留需要的列
df = df[['user_id', 'item_id']]

# 只过滤掉ground truth items，允许其他测试items在训练中出现
original_interactions = len(df)
df = df[~df['item_id'].isin(test_items)]
filtered_interactions = len(df)

print(f"数据过滤统计:")
print(f"  原始交互记录数: {original_interactions}")
print(f"  过滤后交互记录数: {filtered_interactions}")
print(f"  过滤掉的ground truth记录数: {original_interactions - filtered_interactions}")

if len(df) == 0:
    print("错误：过滤后没有训练数据！")
    exit(1)

user2id = {u: i for i, u in enumerate(df['user_id'].unique())}
item2id = {i: j for j, i in enumerate(df['item_id'].unique())}

df['user'] = df['user_id'].map(user2id)
df['item'] = df['item_id'].map(item2id)

num_users = len(user2id)
num_items = len(item2id)

print(f"最终训练数据统计:")
print(f"  用户数量: {num_users}")
print(f"  物品数量: {num_items}")
print(f"  交互记录数: {len(df)}")

# -------------------------------
# Step 3: 构建 BPRDataset
# -------------------------------
class BPRDataset(Dataset):
    def __init__(self, df, num_items):
        self.user_item_dict = df.groupby('user')['item'].apply(set).to_dict()
        self.users = list(self.user_item_dict.keys())
        self.num_items = num_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_items = list(self.user_item_dict[user])
        pos_item = random.choice(pos_items)

        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in self.user_item_dict[user]:
                break

        return torch.LongTensor([user])[0], torch.LongTensor([pos_item])[0], torch.LongTensor([neg_item])[0]

# -------------------------------
# Step 4: 定义 BPRMF 模型
# -------------------------------
class BPRMF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, user, pos_item, neg_item):
        u = self.user_embedding(user)
        pos = self.item_embedding(pos_item)
        neg = self.item_embedding(neg_item)

        pos_score = torch.sum(u * pos, dim=1)
        neg_score = torch.sum(u * neg, dim=1)

        loss = -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score)))
        return loss

    def get_item_embeddings(self):
        return self.item_embedding.weight.detach().cpu().numpy()

# -------------------------------
# Step 5: 训练模型
# -------------------------------
print("训练BPRMF模型...")
dataset = BPRDataset(df, num_items)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = BPRMF(num_users, num_items, EMBED_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for user, pos, neg in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        user = user.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        loss = model(user, pos, neg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}")

print("训练完成！")

# 获取item embeddings
item_embedding = model.get_item_embeddings()

# -------------------------------
# Step 6: 重新读取测试数据并评估
# -------------------------------
def hit_at_k(rank_list, ground_truth, k):
    return int(ground_truth in rank_list[:k])

def ndcg_at_k(rank_list, ground_truth, k):
    if ground_truth in rank_list[:k]:
        rank = rank_list.index(ground_truth)
        return 1.0 / np.log2(rank + 2)
    return 0.0

print("重新读取测试数据...")
test_data = []
with open(test_data_path, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) != 3:
            continue
        bundle_id = int(parts[0])
        bundle_items = list(map(int, parts[1].split()))
        candidate_ids = list(map(int, parts[2].split()))
        test_data.append((bundle_id, bundle_items, candidate_ids))

print(f"加载了 {len(test_data)} 个测试样本")

# 反向映射（仅包含训练过的items）
reverse_item_id_map = {v: k for k, v in item2id.items()}

# 获取训练时见过的所有items（已排除测试items）
trained_items = set(item2id.keys())

print("开始评估 bundle 补全任务...")

hit_count = 0
total_samples = len(test_data)
successful_count = 0

for bundle_id, bundle_items, candidate_ids in test_data:
    # 获取完整的原始bundle
    if bundle_id not in bundle_dict:
        continue
    
    original_bundle = bundle_dict[bundle_id]
    
    # 找到ground truth: 在完整bundle中但不在当前bundle_items中的item
    ground_truth_items = [item for item in original_bundle if item not in bundle_items]
    
    # 找到在候选集中的ground truth items
    ground_truth_in_candidates = [item for item in ground_truth_items if item in candidate_ids]
    
    # 如果没有有效的ground truth，跳过
    if not ground_truth_in_candidates:
        continue
    
    # 选择第一个作为ground truth
    ground_truth_item = ground_truth_in_candidates[0]
    
    # 将bundle items和候选items转换为embeddings索引（只保留训练过的items）
    bundle_indices = [item2id[item] for item in bundle_items if item in item2id]
    candidate_indices = [item2id[item] for item in candidate_ids if item in item2id]
    
    # 必须至少有一个bundle item在训练中见过
    if not bundle_indices:
        continue
    
    # 如果所有候选items都不在训练中，使用随机分数
    if not candidate_indices:
        scores = [(item, 0.05 + 0.10 * np.random.random()) for item in candidate_ids]
        ranked_items = sorted(scores, key=lambda x: x[1], reverse=True)
        ranked_ids = [item_id for item_id, _ in ranked_items]
        
        hit = hit_at_k(ranked_ids, ground_truth_item, TOP_K)
        hit_count += hit
        successful_count += 1
        continue
    
    # 获取bundle items的embeddings（作为上下文）
    bundle_embs = item_embedding[bundle_indices]
    
    # 对每个候选item计算与bundle items的相似度
    scores = []
    
    # 处理在训练中见过的候选items
    for cand_idx in candidate_indices:
        cand_emb = item_embedding[cand_idx].reshape(1, -1)
        # 计算与bundle中所有items的余弦相似度
        sims = cosine_similarity(cand_emb, bundle_embs).flatten()
        
        # 相似度计算：主要依赖平均相似度
        avg_sim = sims.mean()
        max_sim = sims.max()
        min_sim = sims.min()
        
        # 相似度组合
        combined_sim = 0.8 * avg_sim + 0.15 * max_sim + 0.05 * min_sim
        scores.append((reverse_item_id_map[cand_idx], combined_sim))
    
    # 对于不在训练中的候选items，给中等的随机分数
    trained_candidate_items = set(reverse_item_id_map[idx] for idx in candidate_indices)
    for item in candidate_ids:
        if item not in trained_candidate_items:
            random_score = 0.05 + 0.10 * np.random.random()
            scores.append((item, random_score))
    
    # 按相似度排序
    ranked_items = sorted(scores, key=lambda x: x[1], reverse=True)
    ranked_ids = [item_id for item_id, _ in ranked_items]
    
    hit = hit_at_k(ranked_ids, ground_truth_item, TOP_K)
    hit_count += hit
    successful_count += 1

# 计算总训练和验证时间
end_time = time.time()
total_minutes = (end_time - start_time) / 60

# -------------------------------
# Step 7: 输出结果并保存到文件
# -------------------------------
success_rate = f"{successful_count}/{total_samples}"
hit_rate = hit_count / total_samples if total_samples > 0 else 0

result_text = f"""ItemKNN Add Task Results - {args.d}
Hit@1: {hit_rate:.4f}
Hit count: {hit_count}
Total bundles: {total_samples}
Success rate: {success_rate}
Time: {total_minutes:.2f} minutes"""

print("\n" + "="*50)
print(result_text)
print("="*50)

# 保存结果到文件
result_file = os.path.join(result_dir, f'itemknn_add_{args.d}_results.txt')
with open(result_file, 'w') as f:
    f.write(result_text)

print(f"结果已保存到: {result_file}")