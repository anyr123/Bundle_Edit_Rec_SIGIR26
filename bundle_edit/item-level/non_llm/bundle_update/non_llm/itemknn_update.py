'''
Description: ItemKNN模型在bundle update任务中的验证代码（使用BPRMF生成embeddings）
Author: anyiran
Date: 2025-09-19
LastEditTime: 2025-09-22 16:34:18
'''

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import pickle
import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time

# -------------------------------
# 命令行参数解析
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default='clothing', 
                    choices=['food', 'clothing', 'electronic'],
                    help='选择数据集: food, clothing, electronic')

args = parser.parse_args()

print(f"使用数据集: {args.d}")

# -------------------------------
# 路径设置
# -------------------------------
# 构建数据集路径
data_path = '../../dataset'
dataset_path = os.path.join(data_path, args.d).replace('\\', '/')
bundle_item_path = os.path.join(dataset_path, 'bundle_item.csv').replace('\\', '/')
user_bundle_file = os.path.join(dataset_path, 'user_bundle.csv').replace('\\', '/')
user_item_file = os.path.join(dataset_path, 'user_item.csv').replace('\\', '/')

# 测试数据路径
test_data_path = f'../../testdata/{args.d}/update_test.txt'

print(f"Bundle-item数据: {bundle_item_path}")
print(f"User-item数据: {user_item_file}")
print(f"测试数据: {test_data_path}")

# 确保结果目录存在
result_dir = f'../result/{args.d}'
os.makedirs(result_dir, exist_ok=True)

# BPRMF参数设置
EMBED_DIM = 64
BATCH_SIZE = 1024
EPOCHS = 10
LR = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# -------------------------------
# Step 1: 先读取测试数据，收集所有涉及的items（避免数据泄露）
# -------------------------------
print("首先读取测试数据，收集所有涉及的items...")
test_items = set()
with open(test_data_path, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
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
        if len(parts) >= 3:
            bundle_id = int(parts[0])
            if bundle_id in bundle_dict:
                # 找到ground truth: 在完整bundle中但不在当前bundle_items中的item
                bundle_items = list(map(int, parts[1].split()))
                original_bundle = bundle_dict[bundle_id]
                ground_truth_items = [item for item in original_bundle if item not in bundle_items]
                gt_items.update(ground_truth_items)

print(f" Ground Truth items: {len(gt_items)}")

# 更新test_items，只包含ground truth，允许其他items在训练中出现
test_items = gt_items

print(f" 测试数据统计:")
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

print(f" 数据过滤统计:")
print(f"  原始交互记录数: {original_interactions}")
print(f"  过滤后交互记录数: {filtered_interactions}")
print(f"  过滤掉的ground truth记录数: {original_interactions - filtered_interactions}")

if len(df) == 0:
    print(" 错误：过滤后没有训练数据！")
    exit(1)

user2id = {u: i for i, u in enumerate(df['user_id'].unique())}
item2id = {i: j for j, i in enumerate(df['item_id'].unique())}

df['user'] = df['user_id'].map(user2id)
df['item'] = df['item_id'].map(item2id)

num_users = len(user2id)
num_items = len(item2id)

print(f" 最终训练数据统计:")
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
# Step 5: 训练BPRMF模型生成item embeddings
# -------------------------------
print("训练BPRMF模型生成item embeddings...")
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
    print(f" Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}")

print(" BPRMF训练完成！")

# 获取item embeddings
item_embeddings = model.get_item_embeddings()
print(f" 生成的item embeddings维度: {item_embeddings.shape}")

# 可选：保存生成的embeddings以便后续使用
save_embeddings = True
if save_embeddings:
    try:
        os.makedirs('generated_embeddings', exist_ok=True)
        np.save(f'generated_embeddings/itemknn_update_embedding_{args.d}.npy', item_embeddings)
        with open(f'generated_embeddings/itemknn_update_mapping_{args.d}.pkl', 'wb') as f:
            pickle.dump(item2id, f)
        print(f" Embeddings已保存到 generated_embeddings/ 目录")
    except Exception as e:
        print(f" 保存embeddings失败: {e}")

# -------------------------------
# Step 6: 加载测试数据
# -------------------------------
print("加载update测试数据...")

test_bundles = []
try:
    with open(test_data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            
            parts = line.split('\t')
            if len(parts) >= 3:
                bundle_id = int(parts[0])
                # 解析bundle items（包含需要更新的outlier item）
                bundle_with_outlier = list(map(int, parts[1].split()))
                # 解析候选items
                candidate_items = list(map(int, parts[2].split()))
                
                if len(bundle_with_outlier) >= 2 and len(candidate_items) >= 1:
                    # 假设outlier是最后一个item（根据update_test.txt的格式）
                    outlier_item = bundle_with_outlier[-1]
                    bundle_items = bundle_with_outlier[:-1]  # 除了outlier的其他items
                    test_bundles.append((bundle_id, bundle_items, outlier_item, candidate_items))
    
    print(f" 成功加载{len(test_bundles)}个测试bundles")
    
except Exception as e:
    print(f" 加载测试数据失败: {e}")
    exit(1)

# -------------------------------
# Step 7: Bundle Update任务验证
# -------------------------------
def calculate_replacement_score(bundle_items, candidate_item, outlier_item, item_embeddings, item2id, reverse_item_id_map):
    """
    计算候选item作为outlier替代品的得分
    方法：
    1. 计算候选item与bundle中其他items的相似度
    2. 计算outlier与bundle中其他items的相似度
    3. 比较两者，候选item与bundle更相似则得分更高
    
    返回: score 分数越高越适合作为替代品
    """
    # 过滤掉不在embedding中的items（即在训练中见过的items）
    valid_bundle_items = [item for item in bundle_items if item in item2id]
    
    if len(valid_bundle_items) == 0:
        return 0.0
    
    if candidate_item not in item2id:
        # 未训练的候选item给予随机分数
        np.random.seed(42 + candidate_item)
        return 0.1 + 0.2 * np.random.random()
    
    # 获取embeddings
    candidate_idx = item2id[candidate_item]
    candidate_emb = item_embeddings[candidate_idx].reshape(1, -1)
    
    bundle_indices = [item2id[item] for item in valid_bundle_items]
    bundle_embs = item_embeddings[bundle_indices]
    
    # 计算候选item与bundle items的相似度
    candidate_similarities = cosine_similarity(candidate_emb, bundle_embs).flatten()
    candidate_avg_sim = candidate_similarities.mean()
    candidate_max_sim = candidate_similarities.max()
    candidate_min_sim = candidate_similarities.min()
    
    # 候选item的综合相似度分数（与bundle的匹配度）
    candidate_score = 0.8 * candidate_avg_sim + 0.15 * candidate_max_sim + 0.05 * candidate_min_sim
    
    # 如果outlier也在embedding中，计算其与bundle的相似度作为对比
    if outlier_item in item2id:
        outlier_idx = item2id[outlier_item]
        outlier_emb = item_embeddings[outlier_idx].reshape(1, -1)
        outlier_similarities = cosine_similarity(outlier_emb, bundle_embs).flatten()
        outlier_avg_sim = outlier_similarities.mean()
        outlier_max_sim = outlier_similarities.max()
        outlier_min_sim = outlier_similarities.min()
        
        outlier_score = 0.8 * outlier_avg_sim + 0.15 * outlier_max_sim + 0.05 * outlier_min_sim
        
        # 最终分数：候选item的相似度 - outlier的相似度 + 基础分数
        # 这样可以优先选择比outlier更适合的候选item
        final_score = candidate_score - outlier_score + 0.5
    else:
        # 如果outlier不在embedding中，直接使用候选item的相似度分数
        final_score = candidate_score
    
    return max(0.0, final_score)  # 确保分数非负

print("开始Bundle Update任务验证...")

hits = 0
random_hits = 0  # 随机选择的命中次数
total_bundles = 0
failed_bundles = 0

results = []

# 设置随机种子以便复现
np.random.seed(42)

# 反向映射（仅包含训练过的items）
reverse_item_id_map = {v: k for k, v in item2id.items()}

# 获取训练时见过的所有items（已排除测试items）
trained_items = set(item2id.keys())

for bundle_id, bundle_items, outlier_item, candidate_items in test_bundles:
    try:
        # 确定ground truth（原始bundle的完整item列表）
        if bundle_id in bundle_dict:
            original_bundle = bundle_dict[bundle_id]
            # 假设ground truth是原始bundle中不在当前bundle_items中且在候选items中的item
            ground_truth_candidates = [item for item in original_bundle 
                                     if item not in bundle_items and item in candidate_items]
            if ground_truth_candidates:
                ground_truth = ground_truth_candidates[0]  # 选择第一个匹配的
            else:
                # 如果找不到明确的ground truth，跳过这个bundle
                failed_bundles += 1
                continue
        else:
            # 如果没有bundle信息，跳过
            failed_bundles += 1
            continue
        
        # 计算每个候选item的替代分数
        candidate_scores = {}
        for candidate in candidate_items:
            score = calculate_replacement_score(bundle_items, candidate, outlier_item, 
                                              item_embeddings, item2id, reverse_item_id_map)
            candidate_scores[candidate] = score
        
        if len(candidate_scores) == 0:
            print(f" Bundle {bundle_id}: 没有有效的候选items，跳过")
            failed_bundles += 1
            continue
        
        # 找到得分最高的候选item
        predicted_best = max(candidate_scores.keys(), key=lambda x: candidate_scores[x])
        
        # 判断预测是否正确（Hit@1）
        is_hit = (predicted_best == ground_truth)
        if is_hit:
            hits += 1
        
        # 随机选择一个候选item作为基准对比
        random_candidate = np.random.choice(candidate_items)
        is_random_hit = (random_candidate == ground_truth)
        if is_random_hit:
            random_hits += 1
        
        total_bundles += 1
        
        # 保存结果
        results.append({
            'bundle_id': bundle_id,
            'bundle_items': bundle_items,
            'outlier_item': outlier_item,
            'ground_truth': ground_truth,
            'predicted_best': predicted_best,
            'hit': is_hit,
            'candidate_scores': candidate_scores
        })
        
        # 打印前10个结果的详细信息
        if total_bundles <= 10:
            print(f"\nBundle {bundle_id}:")
            print(f"  Bundle items: {bundle_items}")
            print(f"  Outlier item: {outlier_item}")
            print(f"  Ground truth: {ground_truth}")
            print(f"  预测最佳替代: {predicted_best}")
            print(f"  预测{'正确' if is_hit else '错误'}")
            # 显示top-5候选items的分数
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            print(f"  Top-5候选items:")
            for rank, (item, score) in enumerate(sorted_candidates[:5]):
                hit_flag = "正确" if item == ground_truth else "错误"
                print(f"    Rank {rank + 1}: Item {item} | Score: {score:.4f} {hit_flag}")
            print()
        
        # 如果前10个bundle都失败了，额外打印调试信息
        if total_bundles <= 3 and failed_bundles <= 10:
            print(f"调试信息 - Bundle {bundle_id}:")
            print(f"  候选items数量: {len(candidate_items)}")
            print(f"  在embedding中的候选items: {len([c for c in candidate_items if c in item2id])}")
            print(f"  原始bundle信息: {bundle_dict.get(bundle_id, '未找到')}")
            print()
        
    except Exception as e:
        print(f"Bundle {bundle_id}处理失败: {e}")
        failed_bundles += 1
        continue

end_time = time.time()
total_time = (end_time - start_time) / 60

# -------------------------------
# Step 8: 计算并打印结果
# -------------------------------
print("\n" + "="*50)
print("Bundle Update任务验证结果")
print("="*50)

if total_bundles > 0:
    hit_rate = hits / total_bundles
    random_hit_rate = random_hits / total_bundles
    print(f"数据集: {args.d}")
    print(f"总测试bundles: {len(test_bundles)}")
    print(f"成功处理: {total_bundles}")
    print(f"处理失败: {failed_bundles}")
    print(f"ItemKNN命中次数: {hits}")
    print(f"ItemKNN Hit@1: {hit_rate:.4f} ({hit_rate*100:.2f}%)")
    print(f"随机选择命中次数: {random_hits}")
    print(f"随机Hit@1: {random_hit_rate:.4f} ({random_hit_rate*100:.2f}%)")
    print(f"提升幅度: {((hit_rate - random_hit_rate) / random_hit_rate * 100 if random_hit_rate > 0 else 0):.2f}%")
    print(f"Embedding维度: {item_embeddings.shape}")
    print(f"总耗时: {total_time:.2f} 分钟")
    
    # 保存详细结果
    # results_file = f'itemknn_update_bprmf_results_{args.d}.txt'
    # with open(results_file, 'w') as f:
    #     f.write("Bundle_ID\tBundle_Items\tOutlier_Item\tGround_Truth\tPredicted_Best\tHit\tBest_Score\n")
    #     for result in results:
    #         best_score = max(result['candidate_scores'].values())
    #         bundle_items_str = ' '.join(map(str, result['bundle_items']))
    #         f.write(f"{result['bundle_id']}\t{bundle_items_str}\t{result['outlier_item']}\t{result['ground_truth']}\t{result['predicted_best']}\t{result['hit']}\t{best_score:.4f}\n")
    
    # print(f"详细结果已保存到: {results_file}")
    
    # 保存到result目录
    result_text = f"""ITEMKNN Update Task Results - {args.d}
Hit@1: {hit_rate:.4f}
Hit count: {hits}
Total bundles: {total_bundles}
Success rate: {total_bundles}/{len(test_bundles)}
Time: {total_time:.2f} minutes"""

    result_file = os.path.join(result_dir, f'itemknn_update_{args.d}_results.txt')
    with open(result_file, 'w') as f:
        f.write(result_text)
    print(f"结果已保存到: {result_file}")
    
else:
    print("没有成功处理任何bundle，请检查数据格式和数据文件")

print("="*50)