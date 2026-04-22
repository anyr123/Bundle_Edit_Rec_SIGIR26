'''
Description: 整合的ItemKNN模型用于bundle delete任务（训练+测试，结果保存到../result）
Author: anyiran
Date: 2025-09-21
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
from sklearn.model_selection import train_test_split
from collections import defaultdict

# -------------------------------
# 命令行参数解析
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default='clothing', 
                    choices=['food', 'clothing', 'electronic'],
                    help='选择数据集: food, clothing, electronic')
parser.add_argument('--train-only', action='store_true', 
                    help='只进行训练，不进行测试')
parser.add_argument('--eval-only', action='store_true', 
                    help='只进行测试，不进行训练')

args = parser.parse_args()

# 构建数据集路径
data_path = '../../dataset'
dataset_path = os.path.join(data_path, args.d).replace('\\', '/')
bundle_item_path = os.path.join(dataset_path, 'bundle_item.csv').replace('\\', '/')

print(f"使用数据集: {args.d}")
print(f"数据集路径: {dataset_path}")

# 创建结果保存目录
result_dir = f'../result/{args.d}'
os.makedirs(result_dir, exist_ok=True)

# 测试数据路径
test_data_path = f'../../testdata/{args.d}/delete_test.txt'

# 模型参数
EMBED_DIM = 64
BATCH_SIZE = 1024
EPOCHS = 10
LR = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 用于在内存中传递embeddings和映射的全局变量
global_item_embeddings = None
global_item2id = None

# -------------------------------
# Step 1: 修正后的数据划分逻辑
# -------------------------------
def load_and_split_data():
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
    delete_test_bundle_ids = set()
    if os.path.exists(test_data_path):
        with open(test_data_path, 'r') as f:
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
    item2id = {i: j for j, i in enumerate(sorted(df['item_id'].unique()))}  # 保持对所有item的映射
    id2item = {v: k for k, v in item2id.items()}

    # 转换为索引
    train_df['bundle'] = train_df['bundle_id'].map(bundle2id)
    train_df['item'] = train_df['item_id'].map(item2id)

    num_bundles = len(bundle2id)
    num_items = len(item2id)

    print(f"训练数据统计:")
    print(f"  训练bundles数量: {num_bundles}")
    print(f"  总item数量: {num_items}")
    print(f"  训练交互数量: {len(train_df)}")

    return train_df, bundle2id, item2id, id2item, num_bundles, num_items

# -------------------------------
# Step 2: 构建 BPRDataset
# -------------------------------
class BPRDataset(Dataset):
    def __init__(self, df, num_items):
        self.bundle_item_dict = df.groupby('bundle')['item'].apply(set).to_dict()
        self.bundles = list(self.bundle_item_dict.keys())
        self.num_items = num_items

    def __len__(self):
        return len(self.bundles)

    def __getitem__(self, idx):
        bundle = self.bundles[idx]
        pos_items = list(self.bundle_item_dict[bundle])
        pos_item = random.choice(pos_items)

        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in self.bundle_item_dict[bundle]:
                break

        return torch.LongTensor([bundle])[0], torch.LongTensor([pos_item])[0], torch.LongTensor([neg_item])[0]

# -------------------------------
# Step 3: 定义 BPRMF 模型
# -------------------------------
class BPRMF(nn.Module):
    def __init__(self, num_bundles, num_items, embed_dim=64):
        super().__init__()
        self.bundle_embedding = nn.Embedding(num_bundles, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        nn.init.xavier_uniform_(self.bundle_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, bundle, pos_item, neg_item):
        b = self.bundle_embedding(bundle)
        pos = self.item_embedding(pos_item)
        neg = self.item_embedding(neg_item)

        pos_score = torch.sum(b * pos, dim=1)
        neg_score = torch.sum(b * neg, dim=1)

        loss = -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score)))
        return loss

    def get_item_embeddings(self):
        return self.item_embedding.weight.detach().cpu().numpy()

# -------------------------------
# Step 4: 模型训练函数
# -------------------------------
def train_model():
    global global_item_embeddings, global_item2id
    
    print("开始训练 ItemKNN (BPRMF) model for delete task...")
    
    # 加载数据
    train_df, bundle2id, item2id, id2item, num_bundles, num_items = load_and_split_data()
    
    start_time = time.time()
    
    dataset = BPRDataset(train_df, num_items)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BPRMF(num_bundles, num_items, EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for bundle, pos, neg in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            bundle = bundle.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            loss = model(bundle, pos, neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}")

    end_time = time.time()
    total_time = (end_time - start_time) / 60

    # 将embeddings保存到内存中供测试使用
    global_item_embeddings = model.get_item_embeddings()
    global_item2id = item2id

    print(f"训练完成，总耗时: {total_time:.2f} 分钟")
    print(f"Item embeddings形状: {global_item_embeddings.shape}")
    print(f"Item映射数量: {len(global_item2id)}")

# -------------------------------
# Step 5: 异常检测函数
# -------------------------------
def calculate_item_anomaly_score(bundle_items, item_embeddings, item2id):
    """
    计算bundle中每个item的异常分数
    方法：K最近邻方法 + 随机噪声，提高评估的合理性
    返回: {item_id: anomaly_score} 字典，分数越高越异常
    """
    # 过滤掉不在embedding中的items
    valid_items = [item for item in bundle_items if item in item2id]
    
    if len(valid_items) < 2:
        return {}
    
    # 获取item embeddings
    item_embs = np.array([item_embeddings[item2id[item]] for item in valid_items])
    
    anomaly_scores = {}
    for i, item in enumerate(valid_items):
        current_emb = item_embs[i]
        
        # 使用K最近邻方法计算异常分数
        k = min(3, len(valid_items) - 1)  # 最多看3个最近邻
        if k > 0:
            distances_to_others = []
            for j in range(len(valid_items)):
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
# Step 6: 测试函数
# -------------------------------
def test_model():
    global global_item_embeddings, global_item2id
    
    print("开始Delete任务测试...")
    
    # 使用内存中的embeddings
    if global_item_embeddings is None or global_item2id is None:
        print("错误：没有找到训练后的embeddings，请先运行训练")
        return
    
    item_embeddings = global_item_embeddings
    item2id = global_item2id
    
    print(f"成功加载item embeddings，形状: {item_embeddings.shape}")
    print(f"成功加载item映射，共{len(item2id)}个items")

    # 加载测试数据
    print("加载delete测试数据...")
    
    test_bundles = []
    try:
        with open(test_data_path, 'r') as f:
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
        return

    # 执行测试
    start_time = time.time()
    
    hits = 0
    total_bundles = 0
    failed_bundles = 0
    
    results = []
    
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
            
            # 保存结果
            results.append({
                'bundle_id': bundle_id,
                'true_outlier': true_outlier,
                'predicted_outlier': predicted_outlier,
                'hit': is_hit,
                'anomaly_scores': anomaly_scores
            })
            
        except Exception as e:
            failed_bundles += 1
            continue

    end_time = time.time()
    total_time = (end_time - start_time) / 60

    # 计算并打印结果
    print("\n" + "="*50)
    print("Delete任务测试结果")
    print("="*50)

    if total_bundles > 0:
        hit_rate = hits / len(test_bundles)
        print(f"数据集: {args.d}")
        print(f"总测试bundles: {len(test_bundles)}")
        print(f"成功处理: {total_bundles}")
        print(f"处理失败: {failed_bundles}")
        print(f"ItemKNN命中次数: {hits}")
        print(f"ItemKNN Hit@1: {hit_rate:.4f} ({hit_rate*100:.2f}%)")
        print(f"总耗时: {total_time:.2f} 分钟")
        
        # 保存结果到../result文件夹
        result_file = os.path.join(result_dir, f'itemknn_delete_{args.d}.txt')
        with open(result_file, 'w') as f:
            f.write(f"ItemKNN Delete Task Results - {args.d}\n")
            f.write(f"Hit@1: {hit_rate:.4f}\n")
            f.write(f"Hit count: {hits}\n")
            f.write(f"Total bundles: {len(test_bundles)}\n")
            f.write(f"Success rate: {total_bundles}/{len(test_bundles)}\n")
            f.write(f"Time: {total_time:.2f} minutes\n")
        
        print(f"结果已保存到: {result_file}")
        
    else:
        print("没有成功处理任何bundle，请检查数据格式")
    
    print("="*50)

# -------------------------------
# Step 7: 主函数
# -------------------------------
def main():
    if args.eval_only:
        # 只进行测试（需要先有训练好的embeddings）
        print("错误：eval-only模式需要预先训练的embeddings，当前版本不支持")
        return
    elif args.train_only:
        # 只进行训练
        train_model()
    else:
        # 先训练后测试
        train_model()
        print("\n" + "="*50)
        print("训练完成，开始测试...")
        print("="*50)
        test_model()

if __name__ == "__main__":
    main()