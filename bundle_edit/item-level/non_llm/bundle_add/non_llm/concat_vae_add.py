'''
Description: Concat-VAE模型用于bundle add任务 - 完整版（训练+验证）
Author: anyiran
Date: 2025-09-18
LastEditTime: 2025-09-21 14:44:37
'''

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
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
df = pd.read_csv(bundle_item_path)
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

# 3. 训练集bundle = 有效bundle - 测试集bundle
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

# 5. 构建映射（基于所有items，但只训练训练集的bundle）
bundle2id = {b: idx for idx, b in enumerate(sorted(train_df['bundle_id'].unique()))}
item2id = {i: idx for idx, i in enumerate(sorted(df['item_id'].unique()))}  # 保持对所有item的映射
id2item = {v: k for k, v in item2id.items()}

# 统计最大bundle长度（基于训练集）
max_bundle_len = max(len(items) for items in bundle2items.values())

print(f"训练数据统计:")
print(f"  训练bundle数量: {len(bundle2items)}")
print(f"  总item数量: {len(item2id)}")
print(f"  最大bundle长度: {max_bundle_len}")

start_time = time.time()

# -------------------------------
# Step 2: Dataset 定义
# -------------------------------
class ConcatVAEDataset(Dataset):
    def __init__(self, bundle2items, bundle2id, item2id, num_items, max_len):
        self.bundles = list(bundle2items.keys())
        self.bundle2items = bundle2items
        self.bundle2id = bundle2id
        self.item2id = item2id
        self.num_items = num_items
        self.max_len = max_len
        self.pad_id = num_items  # embedding最后一位是padding

    def __len__(self):
        return len(self.bundles)

    def __getitem__(self, idx):
        bundle = self.bundles[idx]
        item_ids = self.bundle2items[bundle]
        item_indices = [self.item2id[i] for i in item_ids if i in self.item2id]
        # padding
        if len(item_indices) < self.max_len:
            item_indices = item_indices + [self.pad_id] * (self.max_len - len(item_indices))
        else:
            item_indices = item_indices[:self.max_len]
        label = torch.zeros(self.num_items)
        for i in item_indices:
            if i != self.pad_id:
                label[i] = 1.0
        return self.bundle2id[bundle], torch.tensor(item_indices), label

# -------------------------------
# Step 3: Concat-VAE 模型定义
# -------------------------------
class ConcatVAE(nn.Module):
    def __init__(self, n_items, embedding_dim=64, latent_dim=32, max_len=20):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=n_items)
        self.max_len = max_len
        
        # 编码器：将连接的embedding编码为潜在空间
        concat_dim = embedding_dim * max_len
        self.encoder_mu = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 2),
            nn.ReLU(),
            nn.Linear(concat_dim // 2, latent_dim)
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 2),
            nn.ReLU(),
            nn.Linear(concat_dim // 2, latent_dim)
        )
        
        # 解码器：从潜在空间解码到物品分布
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, n_items)
        )

        nn.init.xavier_uniform_(self.item_embedding.weight)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, item_indices):
        # 获取embeddings并连接
        embeddings = self.item_embedding(item_indices)  # [batch_size, max_len, embedding_dim]
        concat_embeddings = embeddings.view(embeddings.size(0), -1)  # [batch_size, max_len * embedding_dim]
        
        # 编码
        mu = self.encoder_mu(concat_embeddings)
        logvar = self.encoder_logvar(concat_embeddings)
        z = self.reparameterize(mu, logvar)
        
        # 解码
        logits = self.decoder(z)
        return logits, mu, logvar

# -------------------------------
# Loss 函数：VAE 损失
# -------------------------------
def loss_function(logits, labels, mu, logvar, beta=0.2):
    bce = nn.BCEWithLogitsLoss()(logits, labels)
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return bce + beta * kl

# -------------------------------
# Step 4: 模型训练
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_items = len(item2id)

# 训练模型
dataset = ConcatVAEDataset(bundle2items, bundle2id, item2id, num_items, max_bundle_len)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

model = ConcatVAE(n_items=num_items, embedding_dim=64, latent_dim=32, max_len=max_bundle_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
print("开始训练 Concat-VAE ...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for _, item_indices, labels in tqdm(dataloader):
        item_indices = item_indices.to(device)
        labels = labels.to(device)
        logits, mu, logvar = model(item_indices)
        loss = loss_function(logits, labels, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

print("训练完成！")

# -------------------------------
# Step 5: 读取测试数据并评估
# -------------------------------
def hit_at_k(rank_list, ground_truth, k):
    return int(ground_truth in rank_list[:k])

def ndcg_at_k(rank_list, ground_truth, k):
    if ground_truth in rank_list[:k]:
        rank = rank_list.index(ground_truth)
        return 1.0 / np.log2(rank + 2)
    return 0.0

print("开始评估 bundle 补全任务 ...")

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

print(f"读取到 {len(test_data)} 个测试样本")

# 读取bundle_item数据获取完整的bundle信息（用于确定ground truth）
bundle_item_df = pd.read_csv(bundle_item_path)
bundle_item_df.columns = ['bundle_id', 'item_id']
bundle_dict = bundle_item_df.groupby('bundle_id')['item_id'].apply(list).to_dict()

top_k = 1
hit_count = 0
total_samples = len(test_data)
successful_count = 0

item_id_map = item2id
reverse_item_id_map = {v: k for k, v in item_id_map.items()}

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

    try:
        context_indices = [item_id_map[i] for i in bundle_items if i in item_id_map]
        candidate_indices = [item_id_map[i] for i in candidate_ids if i in item_id_map]
        ground_truth_idx = item_id_map[ground_truth_item]
    except KeyError:
        continue
    
    if not context_indices:
        continue

    # 准备输入：padding到max_bundle_len
    if len(context_indices) < max_bundle_len:
        padded_context = context_indices + [num_items] * (max_bundle_len - len(context_indices))
    else:
        padded_context = context_indices[:max_bundle_len]
    
    context_tensor = torch.tensor([padded_context], dtype=torch.long).to(device)
    
    with torch.no_grad():
        model.eval()
        logits, mu, logvar = model(context_tensor)
        scores = logits.cpu().numpy().flatten()
    
    # 对候选items按照分数排序
    candidate_scores = []
    for i in candidate_indices:
        cand_item_id = reverse_item_id_map[i]
        # 跳过context中已有的items（避免数据泄露）
        if cand_item_id not in bundle_items:
            candidate_scores.append((cand_item_id, scores[i]))
    
    ranked_items = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
    ranked_ids = [item_id for item_id, _ in ranked_items]

    # 计算指标
    hit = hit_at_k(ranked_ids, ground_truth_item, top_k)
    hit_count += hit
    successful_count += 1

# 计算总训练和验证时间
end_time = time.time()
total_minutes = (end_time - start_time) / 60

# -------------------------------
# Step 6: 输出结果并保存到文件
# -------------------------------
success_rate = f"{successful_count}/{total_samples}"
hit_rate = hit_count / total_samples if total_samples > 0 else 0

result_text = f"""Concat-VAE Add Task Results - {args.d}
Hit@1: {hit_rate:.4f}
Hit count: {hit_count}
Total bundles: {total_samples}
Success rate: {success_rate}
Time: {total_minutes:.2f} minutes"""

print("\n" + "="*50)
print(result_text)
print("="*50)

# 保存结果到文件
result_file = os.path.join(result_dir, f'concat_vae_add_{args.d}_results.txt')
with open(result_file, 'w') as f:
    f.write(result_text)

print(f"结果已保存到: {result_file}")