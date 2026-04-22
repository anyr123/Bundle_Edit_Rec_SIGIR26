'''
Description: 专门为delete任务训练的TSF模型 - 完整版（训练+验证）
Author: anyiran
Date: 2025-09-17
LastEditTime: 2025-09-21 14:23:31
'''

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import pickle
import os
import argparse
from sklearn.model_selection import train_test_split
import time
import math
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

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

# -------------------------------
# Step 1: 修正后的数据划分逻辑
# -------------------------------
print("修正数据划分逻辑...")

# 读取bundle-item数据
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
delete_test_path = f'../../testdata/{args.d}/delete_test.txt'
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

# 验证数据泄露
overlap = set(train_bundles).intersection(delete_test_bundle_ids)
print(f"训练集与Delete测试集重叠bundle数量: {len(overlap)} (应该为0)")

if len(overlap) > 0:
    print(f"警告：发现数据泄露！重叠的bundle: {list(overlap)[:5]}...")
else:
    print("数据划分正确，无泄露")

# 5. 构建映射
bundle2id = {b: i for i, b in enumerate(train_df['bundle_id'].unique())}
item2id = {i: idx for idx, i in enumerate(sorted(df['item_id'].unique()))}  # 保持对所有item的映射
id2item = {v: k for k, v in item2id.items()}

num_items = len(item2id)
num_bundles = len(bundle2id)

# 构建bundle -> items映射（只包含训练bundle）
bundle2items = train_df.groupby('bundle_id')['item_id'].apply(list).to_dict()

# 计算最大序列长度（基于训练集）
max_seq_len = max(len(items) for items in bundle2items.values())

print(f"训练数据统计:")
print(f"  训练bundles数量: {num_bundles}")
print(f"  总item数量: {num_items}")
print(f"  最大序列长度: {max_seq_len}")

# -------------------------------
# Step 2: TSF数据集
# -------------------------------
class TSFDataset(Dataset):
    def __init__(self, bundle2items, bundle2id, item2id, num_items, max_len=20):
        self.bundle2items = bundle2items
        self.bundle2id = bundle2id
        self.item2id = item2id
        self.num_items = num_items
        self.max_len = max_len
        self.bundles = list(bundle2items.keys())

    def __len__(self):
        return len(self.bundles)

    def __getitem__(self, idx):
        bundle = self.bundles[idx]
        items = self.bundle2items[bundle]
        
        # 构建序列（用item id填充）
        sequence = [self.item2id[item] for item in items if item in self.item2id]
        
        # 截断或填充到固定长度
        if len(sequence) > self.max_len:
            sequence = sequence[:self.max_len]
        else:
            # 用特殊的padding id填充
            sequence += [self.num_items] * (self.max_len - len(sequence))
        
        # 构建multi-hot目标
        target = torch.zeros(self.num_items)
        for item in items:
            if item in self.item2id:
                target[self.item2id[item]] = 1.0
        
        return torch.tensor(sequence), target

# -------------------------------
# Step 3: TSF模型（简化版Transformer）
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TSFModel(nn.Module):
    def __init__(self, num_items, d_model=64, nhead=4, num_layers=2, max_len=20):
        super(TSFModel, self).__init__()
        self.d_model = d_model
        self.num_items = num_items
        
        # Item embedding (包括padding)
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=num_items)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, num_items)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        
    def forward(self, src, src_key_padding_mask=None):
        # src: [batch_size, seq_len]
        src_emb = self.item_embedding(src) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        src_emb = src_emb.transpose(0, 1)  # [seq_len, batch_size, d_model]
        src_emb = self.pos_encoder(src_emb)
        src_emb = src_emb.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Transformer
        output = self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        # 平均池化
        if src_key_padding_mask is not None:
            # 计算非padding位置的平均
            mask = ~src_key_padding_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            output = output * mask
            output = output.sum(dim=1) / mask.sum(dim=1)
        else:
            output = output.mean(dim=1)  # [batch_size, d_model]
        
        # 输出预测
        logits = self.output_layer(output)  # [batch_size, num_items]
        return logits

# -------------------------------
# Step 4: 辅助函数
# -------------------------------
def custom_collate_fn(batch):
    sequences, targets = zip(*batch)
    sequences = torch.stack(sequences)
    targets = torch.stack(targets)
    return sequences, targets

# -------------------------------
# Step 5: 模型训练
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = TSFDataset(bundle2items, bundle2id, item2id, num_items, max_len=min(max_seq_len, 20))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=custom_collate_fn)

model = TSFModel(num_items=num_items, d_model=64, nhead=4, num_layers=2, max_len=min(max_seq_len, 20)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

epochs = 10
print("开始训练 TSF for delete task...")
start_time = time.time()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for input_sequences, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        input_sequences = input_sequences.to(device)
        labels = labels.to(device)
        
        # 创建padding mask
        padding_mask = (input_sequences == num_items)
        
        optimizer.zero_grad()
        
        # 前向传播
        logits = model(input_sequences, src_key_padding_mask=padding_mask)
        
        # 计算损失
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

print("训练完成！")

# -------------------------------
# Step 6: 加载测试数据并验证
# -------------------------------
def calculate_item_anomaly_score(bundle_items, item_embeddings, item2id):
    """
    计算bundle中每个item的异常分数
    方法：计算每个item与bundle中心（其他items的均值）的欧几里得距离
    返回: {item_id: anomaly_score} 字典，分数越高越异常
    """
    # 过滤掉不在embedding中的items
    valid_items = [item for item in bundle_items if item in item2id]
    
    if len(valid_items) < 2:
        return {}
    
    # 获取item embeddings，注意TSF可能有padding embedding
    item_embs = []
    final_valid_items = []
    for item in valid_items:
        item_idx = item2id[item]
        # TSF的padding embedding通常是最后一个，跳过padding
        if item_idx < len(item_embeddings) - 1:
            emb = item_embeddings[item_idx]
            item_embs.append(emb)
            final_valid_items.append(item)
    
    if len(item_embs) < 2:
        return {}
    
    item_embs = np.array(item_embs)
    
    anomaly_scores = {}
    for i, item in enumerate(final_valid_items):
        current_emb = item_embs[i]
        
        # 使用K最近邻方法计算异常分数
        k = min(3, len(final_valid_items) - 1)  # 最多看3个最近邻
        if k > 0:
            distances_to_others = []
            for j in range(len(final_valid_items)):
                if i != j:
                    other_emb = item_embs[j]
                    dist = np.linalg.norm(current_emb - other_emb)
                    distances_to_others.append(dist)
            
            # 选择k个最小距离（最相似的邻居）
            distances_to_others.sort()
            k_nearest_dist = np.mean(distances_to_others[:k])
            
            # 添加随机噪声降低确定性
            np.random.seed(42 + item)
            noise = np.random.normal(0, 0.3)
            
            anomaly_score = k_nearest_dist + noise
        else:
            anomaly_score = 0.5  # 默认中等异常分数
        
        anomaly_scores[item] = max(0, anomaly_score)  # 确保分数非负
    
    return anomaly_scores

print("开始Delete任务验证...")

# 获取训练好的item embeddings
item_embeddings = model.item_embedding.weight.detach().cpu().numpy()

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
# Step 7: 输出结果并保存到文件
# -------------------------------
success_rate = f"{total_bundles}/{len(test_bundles)}"
hit_rate = hits / len(test_bundles) if len(test_bundles) > 0 else 0

result_text = f"""TSF Delete Task Results - {args.d}
Hit@1: {hit_rate:.4f}
Hit count: {hits}
Total bundles: {len(test_bundles)}
Success rate: {success_rate}
Time: {total_time:.2f} minutes"""

print("\n" + "="*50)
print(result_text)
print("="*50)

# 保存结果到文件
result_file = os.path.join(result_dir, f'tsf_delete_{args.d}.txt')
with open(result_file, 'w') as f:
    f.write(result_text)

print(f"结果已保存到: {result_file}")