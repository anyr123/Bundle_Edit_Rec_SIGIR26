'''
Description: TSF模型用于bundle update任务 - 包含模型训练+验证
基于bundle add任务的完整实现，结合delete和add功能
Author: anyiran
Date: 2025-09-21
LastEditTime: 2025-09-21 16:35:17
'''

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import os
import math
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

# 测试数据路径（修改为update测试数据）
test_data_path = f'../../testdata/{args.d}/update_test.txt'

# -------------------------------
# Step 1: 数据划分逻辑（参考add任务）
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

# 2. 读取测试集的bundle_id（从update_test.txt获取）
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
max_bundle_len = min(max(len(items) for items in bundle2items.values()), 10)  # 限制最大长度

print(f"训练数据统计:")
print(f"  训练bundle数量: {len(bundle2items)}")
print(f"  总item数量: {len(item2id)}")
print(f"  最大bundle长度: {max_bundle_len}")

start_time = time.time()

# -------------------------------
# Step 2: TSF 模型定义
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class TSF(nn.Module):
    def __init__(self, n_items, embedding_dim=32, nhead=4, num_layers=2, max_len=10):
        super(TSF, self).__init__()
        
        # 基础参数
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        
        # 物品嵌入层
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=n_items)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len)
        
        # Transformer 编码器
        self.transformer_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layers, 
            num_layers=num_layers
        )
        
        # 输出层
        self.output_layer = nn.Linear(embedding_dim, n_items)
        
        # 初始化
        nn.init.xavier_uniform_(self.item_embedding.weight[:-1])  # 除了padding
        nn.init.xavier_uniform_(self.output_layer.weight)

    def create_padding_mask(self, x, pad_idx):
        """创建padding mask"""
        return (x == pad_idx)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        batch_size, seq_len = x.size()
        
        # 嵌入
        embedded = self.item_embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # 位置编码
        embedded = self.pos_encoding(embedded)
        
        # 创建padding mask
        padding_mask = self.create_padding_mask(x, self.item_embedding.num_embeddings - 1)
        
        # Transformer编码
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
        
        # 全局平均池化（忽略padding位置）
        mask = (~padding_mask).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1)  # [batch_size, embedding_dim]
        
        # 输出
        logits = self.output_layer(pooled)  # [batch_size, n_items]
        return logits

# -------------------------------
# Step 3: Dataset 定义
# -------------------------------
class TSFDataset(Dataset):
    def __init__(self, bundle2items, bundle2id, item2id, num_items, max_len):
        self.bundles = list(bundle2items.keys())
        self.bundle2items = bundle2items
        self.bundle2id = bundle2id
        self.item2id = item2id
        self.num_items = num_items
        self.max_len = max_len
        self.pad_id = num_items

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
        
        # 创建标签
        label = torch.zeros(self.num_items)
        for i in item_indices:
            if i != self.pad_id:
                label[i] = 1.0
        
        return torch.tensor(item_indices, dtype=torch.long), label

# -------------------------------
# Step 4: 模型训练
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_items = len(item2id)

# 训练模型
dataset = TSFDataset(bundle2items, bundle2id, item2id, num_items, max_bundle_len)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

model = TSF(n_items=num_items, embedding_dim=32, nhead=4, num_layers=2, max_len=max_bundle_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

epochs = 10
print("开始训练 TSF ...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for item_indices, labels in tqdm(dataloader):
        item_indices = item_indices.to(device)
        labels = labels.to(device)
        
        logits = model(item_indices)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

print("训练完成！")

# 获取item embeddings
item_embeddings = model.item_embedding.weight.detach().cpu().numpy()

# -------------------------------
# Step 5: 离群item检测函数（用于delete）
# -------------------------------
def calculate_item_anomaly_score_tsf(bundle_items, item_embeddings, item2id):
    """
    使用TSF embeddings计算bundle中每个item的异常分数
    方法：基于K最近邻距离 + 随机噪声
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

# -------------------------------
# Step 6: 候选item选择函数（用于add）
# -------------------------------
def select_best_candidate_tsf(bundle_items, candidate_ids, model, item2id, id2item, max_len, device):
    """
    使用TSF模型从候选集中选择最适合bundle的item
    方法：使用TSF Transformer模型进行预测
    返回: 最佳候选item的id
    """
    # 过滤有效的bundle items和候选items
    valid_bundle_items = [item for item in bundle_items if item in item2id]
    valid_candidates = [item for item in candidate_ids if item in item2id and item not in bundle_items]
    
    if not valid_bundle_items or not valid_candidates:
        return None
    
    # 准备输入：padding到max_len
    context_indices = [item2id[item] for item in valid_bundle_items]
    if len(context_indices) < max_len:
        padded_context = context_indices + [len(item2id)] * (max_len - len(context_indices))
    else:
        padded_context = context_indices[:max_len]
    
    context_tensor = torch.tensor([padded_context], dtype=torch.long).to(device)
    
    with torch.no_grad():
        model.eval()
        logits = model(context_tensor)
        scores = logits.cpu().numpy().flatten()
    
    # 找到得分最高的候选item
    best_item = None
    best_score = -np.inf
    
    for candidate in valid_candidates:
        candidate_idx = item2id[candidate]
        if scores[candidate_idx] > best_score:
            best_score = scores[candidate_idx]
            best_item = candidate
    
    return best_item

# -------------------------------
# Step 7: 读取测试数据并评估Update任务
# -------------------------------
print("开始评估 bundle update任务...")

test_data = []
with open(test_data_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:  # 跳过空行
            continue
        
        parts = line.split('\t')
        if len(parts) >= 3:
            bundle_id = int(parts[0])
            bundle_items = list(map(int, parts[1].split()))
            candidate_ids = list(map(int, parts[2].split()))
            test_data.append((bundle_id, bundle_items, candidate_ids))

print(f"读取到 {len(test_data)} 个测试样本")

# 读取完整的bundle数据获取ground truth
bundle_item_df = pd.read_csv(bundle_item_path)
bundle_item_df.columns = ['bundle_id', 'item_id']
bundle_dict = bundle_item_df.groupby('bundle_id')['item_id'].apply(list).to_dict()

total_samples = len(test_data)
hits = 0
successful_count = 0
failed_count = 0

results = []

print(f"开始逐个处理 {total_samples} 个测试样本...")

for idx, (bundle_id, bundle_items, candidate_ids) in enumerate(test_data):
    
    try:
        # 获取ground truth bundle
        if bundle_id not in bundle_dict:
            print(f"Bundle {bundle_id} 不在完整数据中，跳过")
            failed_count += 1
            continue
        
        ground_truth_bundle = bundle_dict[bundle_id]
        
        # 找到ground truth：应该删除的item和应该添加的item
        items_to_remove = [item for item in bundle_items if item not in ground_truth_bundle]
        items_to_add = [item for item in ground_truth_bundle if item not in bundle_items and item in candidate_ids]
        
        if not items_to_remove and not items_to_add:
            print(f"Bundle {bundle_id} 无需更新，跳过")
            failed_count += 1
            continue
        
        # Phase 1: 使用TSF Delete检测离群item
        predicted_remove = None
        if len(bundle_items) > 1 and items_to_remove:
            anomaly_scores = calculate_item_anomaly_score_tsf(bundle_items, item_embeddings, item2id)
            if anomaly_scores:
                predicted_remove = max(anomaly_scores.keys(), key=lambda x: anomaly_scores[x])
        
        # Phase 2: 使用TSF Add选择最佳候选item
        # 从bundle中移除预测的删除item后进行候选选择
        updated_bundle = [item for item in bundle_items if item != predicted_remove] if predicted_remove else bundle_items
        predicted_add = select_best_candidate_tsf(updated_bundle, candidate_ids, model, item2id, id2item, max_bundle_len, device)
        
        # 评估结果
        remove_hit = predicted_remove in items_to_remove if predicted_remove and items_to_remove else (not items_to_remove)
        add_hit = predicted_add in items_to_add if predicted_add and items_to_add else (not items_to_add)
        
        # 整体命中：删除和添加都正确
        is_hit = remove_hit and add_hit
        
        if is_hit:
            hits += 1
        
        successful_count += 1
        
        # 保存详细结果
        results.append({
            'bundle_id': bundle_id,
            'original_bundle': bundle_items,
            'ground_truth_bundle': ground_truth_bundle,
            'items_to_remove': items_to_remove,
            'items_to_add': items_to_add,
            'predicted_remove': predicted_remove,
            'predicted_add': predicted_add,
            'remove_hit': remove_hit,
            'add_hit': add_hit,
            'overall_hit': is_hit
        })
        
        # 打印前10个结果的详细信息
        if successful_count <= 10:
            print(f"\nBundle {bundle_id}:")
            print(f"  原始bundle: {bundle_items}")
            print(f"  真实bundle: {ground_truth_bundle}")
            print(f"  应删除: {items_to_remove}")
            print(f"  应添加: {items_to_add}")
            print(f"  预测删除: {predicted_remove} {'✅' if remove_hit else '❌'}")
            print(f"  预测添加: {predicted_add} {'✅' if add_hit else '❌'}")
            print(f"  整体结果: {'✅正确' if is_hit else '❌错误'}")
        
    except Exception as e:
        print(f"Bundle {bundle_id} 处理失败: {e}")
        failed_count += 1
        continue

# 计算总训练和验证时间
end_time = time.time()
total_minutes = (end_time - start_time) / 60

# -------------------------------
# Step 8: 输出结果并保存到文件
# -------------------------------
print("\n" + "="*80)
print("TSF Bundle Update任务验证结果")
print("="*80)

if successful_count > 0:
    hit_rate = hits / total_samples
    success_hit_rate = hits / successful_count
    
    # 分别统计删除和添加的准确率
    remove_hits = sum(1 for r in results if r['remove_hit'])
    add_hits = sum(1 for r in results if r['add_hit'])
    
    # 随机基准对比
    random_hits = 0
    np.random.seed(42)
    for result in results:
        # 随机删除一个item
        random_remove = np.random.choice(result['original_bundle']) if result['original_bundle'] else None
        # 随机选择一个候选item
        candidates = [item for item in result['ground_truth_bundle'] if item not in result['original_bundle']]
        random_add = np.random.choice(candidates) if candidates else None
        
        random_remove_hit = random_remove in result['items_to_remove'] if random_remove and result['items_to_remove'] else (not result['items_to_remove'])
        random_add_hit = random_add in result['items_to_add'] if random_add and result['items_to_add'] else (not result['items_to_add'])
        
        if random_remove_hit and random_add_hit:
            random_hits += 1
    
    random_hit_rate = random_hits / successful_count
    
    result_text = f"""TSF Update Task Results - {args.d}
Dataset: {args.d}
Total samples: {total_samples}
Successfully processed: {successful_count}
Failed: {failed_count}
Hits: {hits}
Overall Hit Rate: {hit_rate:.4f} ({hit_rate*100:.2f}%)
Success Hit Rate: {success_hit_rate:.4f} ({success_hit_rate*100:.2f}%)
Valid processing rate: {successful_count/total_samples*100:.1f}%

Detailed Statistics:
Delete accuracy: {remove_hits/successful_count:.4f} ({remove_hits}/{successful_count})
Add accuracy: {add_hits/successful_count:.4f} ({add_hits}/{successful_count})
Random baseline: {random_hit_rate:.4f} ({random_hit_rate*100:.2f}%)
Relative improvement: {((success_hit_rate - random_hit_rate) / random_hit_rate * 100 if random_hit_rate > 0 else 0):.2f}%

Training + Validation time: {total_minutes:.2f} minutes

Model Details:
- Embedding dimension: 32
- Transformer heads: 4
- Transformer layers: 2
- Max sequence length: {max_bundle_len}
- Training epochs: {epochs}
- Trained bundles: {len(bundle2items)}
- Total items: {num_items}"""

    print(result_text)
    
    # 保存结果到文件
    result_file = os.path.join(result_dir, f'tsf_update_{args.d}_results.txt')
    with open(result_file, 'w') as f:
        f.write(result_text)
    
    # # 保存详细结果
    # detailed_results_file = os.path.join(result_dir, f'tsf_update_{args.d}_detailed.txt')
    # with open(detailed_results_file, 'w') as f:
    #     f.write("Bundle_ID\tOriginal_Bundle\tGround_Truth\tPredicted_Remove\tPredicted_Add\tRemove_Hit\tAdd_Hit\tOverall_Hit\n")
    #     for result in results:
    #         f.write(f"{result['bundle_id']}\t")
    #         f.write(f"{' '.join(map(str, result['original_bundle']))}\t")
    #         f.write(f"{' '.join(map(str, result['ground_truth_bundle']))}\t")
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
print("说明:")
print("Update任务 = Delete任务 + Add任务的组合")
print("Delete: 使用K最近邻距离检测并删除离群item")
print("Add: 使用TSF Transformer模型预测最佳候选item")
print("Hit Rate: 删除和添加都正确才算命中")
print("模型: 单一TSF模型同时处理delete和add任务")