
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
import os
from tqdm import tqdm

# ============================================================================
# 配置参数
# ============================================================================
DATA_PATH = '/root/autodl-tmp/editrec/test/Youshu/bundle_item.txt'
OUTPUT_DIR = '/root/autodl-tmp/editrec/test/Youshu'
RANDOM_SEED = 123
BPRMF_EPOCHS = 10
EMBEDDING_DIM = 64
L2_REG = 1e-2                  # L2正则系数，防止过拟合
MIN_BUNDLE_SIZE_FOR_COHERENCE = 4   # bundle小于此值时跳过coherence，直接随机选
MAX_BUNDLE_SIZE_FOR_FULL_SIM = 1000 # bundle大于此值时用近似方法，防OOM
APPROX_SAMPLE_SIZE = 500            # 大bundle近似计算时的采样数量

# 设置随机种子
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print("=" * 80)
print("BPRMF智能数据划分（修复版）")
print("=" * 80)
print(f"数据路径: {DATA_PATH}")
print(f"输出目录: {OUTPUT_DIR}")
print()

# ============================================================================
# Step 1: 读取原始数据
# ============================================================================
print("Step 1: 读取原始数据...")
df = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['bundle', 'item'])

# 修复3：统一ID类型为字符串，兼容整数/字符串/非连续ID
df['bundle'] = df['bundle'].astype(str)
df['item'] = df['item'].astype(str)

print(f"  原始数据: {len(df)} pairs, {df['bundle'].nunique()} bundles, {df['item'].nunique()} items")

bundle_items_dict = df.groupby('bundle')['item'].apply(list).to_dict()
all_items = df['item'].unique().tolist()

bundle_sizes = df.groupby('bundle').size()
print(f"  Bundle大小范围: {bundle_sizes.min()} - {bundle_sizes.max()} items")
print(f"  Bundle大小均值: {bundle_sizes.mean():.1f} items")
print(f"  小bundle数量 (size<{MIN_BUNDLE_SIZE_FOR_COHERENCE}): "
      f"{(bundle_sizes < MIN_BUNDLE_SIZE_FOR_COHERENCE).sum()}")
print(f"  大bundle数量 (size>{MAX_BUNDLE_SIZE_FOR_FULL_SIM}): "
      f"{(bundle_sizes > MAX_BUNDLE_SIZE_FOR_FULL_SIM).sum()}")
print()

# ============================================================================
# Step 2: 生成Random基准文件（Bundle-Aware方式）
# ============================================================================
print("Step 2: 生成Random基准（确定每个bundle保留多少items）...")

mask_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
random_keep_plan = {}  # {mask: {bundle_id: [kept_items]}}

for mask in mask_ratios:
    keep_plan = {}

    for bundle_id, items in bundle_items_dict.items():
        n_items = len(items)
        n_keep = max(1, int(n_items * (1 - mask)))
        kept_items = random.sample(items, n_keep)
        keep_plan[bundle_id] = kept_items

    random_keep_plan[mask] = keep_plan
    total_kept = sum(len(v) for v in keep_plan.values())
    print(f"  mask={mask}: 保留 {total_kept}/{len(df)} pairs ({(1-mask)*100:.0f}%)")

print("✓ Random基准计划已生成")
print()

# ============================================================================
# Step 3: 训练BPRMF模型学习item coherence
# ============================================================================
print("Step 3: 训练BPRMF学习item coherence...")

# 修复3：用sorted保证确定性，字符串ID也能正确映射
bundle2id = {b: idx for idx, b in enumerate(sorted(df['bundle'].unique()))}
item2id   = {i: idx for idx, i in enumerate(sorted(df['item'].unique()))}

n_bundles = len(bundle2id)
n_items_vocab = len(item2id)
print(f"  Bundles: {n_bundles}, Items: {n_items_vocab}")

# 构造BPR三元组
print("  构造训练三元组...")
triplets = []
skipped_bundles = 0

for bundle_id, items in tqdm(bundle_items_dict.items(), desc="  构造三元组"):
    items_set = set(items)
    available_negatives = [it for it in all_items if it not in items_set]

    # 修复4：负样本数量太少时跳过，避免负采样崩溃
    if len(available_negatives) < 10:
        skipped_bundles += 1
        continue

    for pos_item in items:
        neg_item = random.choice(available_negatives)
        triplets.append((
            bundle2id[bundle_id],
            item2id[pos_item],
            item2id[neg_item]
        ))

print(f"  三元组数量: {len(triplets)}")
if skipped_bundles > 0:
    print(f"  跳过负样本不足的bundle: {skipped_bundles} 个")

# Dataset
class BPRDataset(Dataset):
    def __init__(self, triplets):
        self.data = triplets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        b, pos, neg = self.data[idx]
        return b, pos, neg

dataset = BPRDataset(triplets)
dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=0)

# BPRMF模型（修复6：加入L2正则）
class BPRMF(nn.Module):
    def __init__(self, n_bundles, n_items, dim=64):
        super().__init__()
        self.bundle_emb = nn.Embedding(n_bundles, dim)
        self.item_emb   = nn.Embedding(n_items, dim)
        nn.init.xavier_uniform_(self.bundle_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, bundle, pos_item, neg_item):
        b   = self.bundle_emb(bundle)
        pos = self.item_emb(pos_item)
        neg = self.item_emb(neg_item)

        pos_score = (b * pos).sum(dim=1)
        neg_score = (b * neg).sum(dim=1)

        bpr_loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

        # L2正则：防止embedding过拟合到0
        reg_loss = (b.norm(2).pow(2) +
                    pos.norm(2).pow(2) +
                    neg.norm(2).pow(2)) / b.size(0)

        return bpr_loss + L2_REG * reg_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  使用设备: {device}")

model = BPRMF(n_bundles, n_items_vocab, dim=EMBEDDING_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(f"  开始训练 ({BPRMF_EPOCHS} epochs)...")
for epoch in range(BPRMF_EPOCHS):
    model.train()
    total_loss = 0
    for batch in dataloader:
        b, pos, neg = batch
        b   = b.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        loss = model(b, pos, neg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(f"    Epoch {epoch+1}/{BPRMF_EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

print("✓ BPRMF训练完成")
print()

# 获取并预归一化item embeddings（只做一次，修复1）
model.eval()
with torch.no_grad():
    item_embeddings = model.item_emb.weight.cpu().numpy()  # (n_items, dim)

print("  预计算归一化embeddings...")
norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
norms = np.where(norms == 0, 1e-8, norms)
normalized_embeddings = item_embeddings / norms  # (n_items, dim)
print("✓ 归一化完成")
print()

# ============================================================================
# Step 4: 使用BPRMF的coherence分数替换Random选择（向量化 + 自适应）
# ============================================================================
print("Step 4: 使用BPRMF智能选择items（向量化加速 + 自适应）...")

def select_by_coherence(items, n_keep, normalized_embeddings, item2id):
    """
    从items中选出coherence最高的n_keep个。
    
    策略：
    - items数量 <= n_keep：全部保留
    - bundle过小(<MIN_BUNDLE_SIZE_FOR_COHERENCE)：随机选（coherence无意义）
    - bundle适中：完整相似度矩阵
    - bundle过大(>MAX_BUNDLE_SIZE_FOR_FULL_SIM)：近似计算（随机子集锚点）
    """
    n = len(items)

    # 无需筛选
    if n <= n_keep:
        return items[:]

    # 修复2：小bundle直接随机选，coherence意义不大
    if n < MIN_BUNDLE_SIZE_FOR_COHERENCE:
        return random.sample(items, n_keep)

    # 获取embedding索引（过滤不在词表中的item）
    valid_mask = [it in item2id for it in items]
    item_indices = [item2id[it] for it, v in zip(items, valid_mask) if v]

    if len(item_indices) == 0:
        return random.sample(items, n_keep)

    # 取出该bundle的归一化embedding矩阵
    bundle_embs = normalized_embeddings[item_indices]  # (n_valid, dim)
    n_valid = len(item_indices)

    if n_valid <= n_keep:
        # 有效item不够，补随机
        valid_items = [it for it, v in zip(items, valid_mask) if v]
        invalid_items = [it for it, v in zip(items, valid_mask) if not v]
        need_more = n_keep - n_valid
        extra = random.sample(invalid_items, min(need_more, len(invalid_items)))
        return valid_items + extra

    # 修复3：大bundle用近似coherence，防OOM
    if n_valid > MAX_BUNDLE_SIZE_FOR_FULL_SIM:
        sample_size = min(APPROX_SAMPLE_SIZE, n_valid)
        anchor_indices = np.random.choice(n_valid, sample_size, replace=False)
        anchor_embs = bundle_embs[anchor_indices]              # (sample, dim)
        sim_matrix = bundle_embs @ anchor_embs.T               # (n_valid, sample)
        coherence_scores = sim_matrix.mean(axis=1)             # (n_valid,)
    else:
        # 完整相似度矩阵
        sim_matrix = bundle_embs @ bundle_embs.T               # (n_valid, n_valid)
        np.fill_diagonal(sim_matrix, 0)
        coherence_scores = sim_matrix.sum(axis=1) / max(n_valid - 1, 1)

    # 选出coherence最高的n_keep个（基于有效item）
    valid_items = [it for it, v in zip(items, valid_mask) if v]
    top_indices = np.argsort(-coherence_scores)[:n_keep]
    kept_items = [valid_items[i] for i in top_indices]

    return kept_items


bprmf_keep_plan = {}

for mask in mask_ratios:
    keep_plan = {}

    for bundle_id, items in tqdm(bundle_items_dict.items(),
                                  desc=f"  mask={mask}",
                                  leave=False):
        n_keep = len(random_keep_plan[mask][bundle_id])
        kept = select_by_coherence(items, n_keep, normalized_embeddings, item2id)
        keep_plan[bundle_id] = kept

    bprmf_keep_plan[mask] = keep_plan
    total_kept = sum(len(v) for v in keep_plan.values())
    print(f"  mask={mask}: 选择了 {total_kept} pairs (最coherent的)")

print("✓ BPRMF智能选择完成")
print()

# ============================================================================
# Step 5: 只保存BPRMF文件
# ============================================================================
print("Step 5: 保存文件...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for mask in mask_ratios:
    bprmf_pairs = [
        [bundle_id, item]
        for bundle_id, kept_items in bprmf_keep_plan[mask].items()
        for item in kept_items
    ]
    bprmf_df = pd.DataFrame(bprmf_pairs, columns=['bundle', 'item'])
    bprmf_df = bprmf_df.sort_values(['bundle', 'item'])
    bprmf_file = os.path.join(OUTPUT_DIR, f'BPRMF_bundle_item_mask{mask}.txt')
    bprmf_df.to_csv(bprmf_file, index=False, header=False, sep='\t')
    print(f"  mask={mask}: BPRMF: {bprmf_file} ({len(bprmf_df)} pairs)")

print("✓ 所有文件已保存")
print()

# ============================================================================
# Step 6: 对比分析
# ============================================================================
print("Step 6: 对比Random vs BPRMF...")
print()

comparison_masks = [0.3, 0.5, 0.7, 0.9]

for mask in comparison_masks:
    print(f"mask={mask} (保留 {(1-mask)*100:.0f}%):")
    print("-" * 60)

    random_set = {
        (bundle_id, item)
        for bundle_id, items in random_keep_plan[mask].items()
        for item in items
    }
    bprmf_set = {
        (bundle_id, item)
        for bundle_id, items in bprmf_keep_plan[mask].items()
        for item in items
    }

    overlap     = random_set & bprmf_set
    only_random = random_set - bprmf_set
    only_bprmf  = bprmf_set  - random_set

    print(f"  总pairs:       {len(random_set)}")
    print(f"  重叠:          {len(overlap)} ({len(overlap)/len(random_set)*100:.1f}%)")
    print(f"  仅Random保留:  {len(only_random)} ({len(only_random)/len(random_set)*100:.1f}%)")
    print(f"  仅BPRMF保留:   {len(only_bprmf)} ({len(only_bprmf)/max(len(bprmf_set),1)*100:.1f}%)")

    sample_bundles = [
        b for b in sorted(bundle_items_dict.keys())
        if len(bundle_items_dict[b]) >= MIN_BUNDLE_SIZE_FOR_COHERENCE
    ][:3]

    print(f"\n  质量对比（前3个有效bundle）:")
    for bundle_id in sample_bundles:
        original    = bundle_items_dict[bundle_id]
        random_kept = random_keep_plan[mask][bundle_id]
        bprmf_kept  = bprmf_keep_plan[mask][bundle_id]

        def avg_coherence(kept_items):
            if len(kept_items) < 2:
                return 0.0
            indices = [item2id[it] for it in kept_items if it in item2id]
            if len(indices) < 2:
                return 0.0
            embs = normalized_embeddings[indices]
            sim  = embs @ embs.T
            np.fill_diagonal(sim, 0)
            return sim.sum() / (len(indices) * (len(indices) - 1))

        random_avg  = avg_coherence(random_kept)
        bprmf_avg   = avg_coherence(bprmf_kept)
        improvement = (bprmf_avg - random_avg) / (random_avg + 1e-8) * 100

        print(f"    Bundle {bundle_id}: {len(original)} → {len(random_kept)} items")
        print(f"      Random coherence: {random_avg:.4f}")
        print(f"      BPRMF coherence:  {bprmf_avg:.4f} ({improvement:+.1f}%)")

    print()

# ============================================================================
# 总结
# ============================================================================
print("=" * 80)
print("完成！")
print("=" * 80)
print()
print("生成的文件:")
print()
print("【BPRMF方法】(智能选择最coherent的items):")
for mask in mask_ratios:
    print(f"  BPRMF_bundle_item_mask{mask}.txt")
print()
print("修复说明:")
print("  ✓ 向量化coherence计算，解决卡死问题")
print("  ✓ 小bundle自适应（iFashion等套装数据集）")
print("  ✓ 大bundle近似计算，防止OOM（NetEase等）")
print("  ✓ 负采样崩溃保护")
print("  ✓ 字符串/非连续ID兼容")
print("  ✓ L2正则防止过拟合")