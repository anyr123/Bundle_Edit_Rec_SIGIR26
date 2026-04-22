# byob_source.py
"""
BYOB: Build Your Own Bundle
基于官方源码 (https://github.com/yoongi0428/BYOB) 改造
适配 bundle generation 数据集
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Categorical
import yaml
from tqdm import tqdm
import argparse
import math
from utils.logger import Logger
from utils.functions import process_results
from utils.metrics import compute


# ===== ItemPointer Network (来自policy_net.py) =====

class ItemPointer(nn.Module):
    """
    Item Pointer Network
    使用 Pointer Network 机制选择 items
    """
    
    def __init__(self, n_user, n_item, pool_size=20, bundle_size=3,
                 embed_dim=32, hidden_dim=128, num_heads=2, num_layers=1,
                 dropout=0.1, clip=10, encoder=True, concat=False):
        super(ItemPointer, self).__init__()
        
        self.n_user = n_user
        self.n_item = n_item
        self.pool_size = pool_size
        self.bundle_size = bundle_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.encoder = encoder
        self.concat = concat
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.clip = clip
        
        self._build()
    
    def _build(self):
        # Embeddings
        self.user_embed = nn.Embedding(self.n_user, self.embed_dim)
        self.item_embed = nn.Embedding(self.n_item, self.embed_dim)
        nn.init.xavier_normal_(self.user_embed.weight, gain=1.0)
        nn.init.xavier_normal_(self.item_embed.weight, gain=1.0)
        
        # Transformer Encoders
        if self.encoder:
            encoder_layer = TransformerEncoderLayer(
                self.embed_dim, self.num_heads, self.hidden_dim, self.dropout
            )
            self.bundle_encoder = TransformerEncoder(encoder_layer, self.num_layers)
            
            encoder_layer = TransformerEncoderLayer(
                self.embed_dim, self.num_heads, self.hidden_dim, self.dropout
            )
            self.pool_encoder = TransformerEncoder(encoder_layer, self.num_layers)
        
        # Concat layer for bundle
        if self.concat:
            self.fc = nn.Linear(self.bundle_size * self.embed_dim, self.embed_dim)
        
        # Pointer Network components
        self.w1 = nn.Linear(3 * self.embed_dim, self.hidden_dim)
        self.w2 = nn.Conv1d(self.embed_dim, self.hidden_dim, 1, 1)
        self.v = nn.Parameter(torch.FloatTensor(self.hidden_dim), requires_grad=True)
        self.v.data.uniform_(-(1. / math.sqrt(self.hidden_dim)), 1. / math.sqrt(self.hidden_dim))
    
    def forward(self, user, seq, pool, bundle):
        """
        前向传播
        
        user: (N,) or (N, 1) - user ids
        seq: (N, L) - sequence of items
        pool: (N, S) - candidate pool
        bundle: (N, K) - current bundle
        
        返回:
        - logits: (N, S) - scores for each pool item
        - features: (N, 3*E) - state features for value function
        """
        batch_size, pool_size = pool.size(0), pool.size(1)
        
        # User embedding
        if user.dim() == 2:
            user = user.squeeze(dim=1)
        user_emb = self.user_embed(user)  # (N, E)
        
        # Sequence embedding (mean pooling)
        seq_emb = self.item_embed(seq)  # (N, L, E)
        seq_emb = torch.mean(seq_emb, dim=1)  # (N, E)
        
        # Bundle embedding
        bundle_emb = self.item_embed(bundle)  # (N, K, E)
        if self.encoder:
            bundle_emb = bundle_emb.permute(1, 0, 2)  # (K, N, E)
            bundle_emb = self.bundle_encoder(bundle_emb, mask=None)  # (K, N, E)
            bundle_emb = bundle_emb.permute(1, 0, 2)  # (N, K, E)
        
        if self.concat:
            bundle_emb = F.tanh(self.fc(bundle_emb.reshape(batch_size, -1)))  # (N, E)
        else:
            bundle_emb = torch.mean(bundle_emb, dim=1)  # (N, E)
        
        # Pool embedding
        pool_emb = self.item_embed(pool)  # (N, S, E)
        if self.encoder:
            pool_emb = pool_emb.permute(1, 0, 2)  # (S, N, E)
            pool_emb = self.pool_encoder(pool_emb, mask=None)  # (S, N, E)
            pool_emb = pool_emb.permute(1, 0, 2)  # (N, S, E)
        pool_emb = pool_emb.permute(0, 2, 1)  # (N, E, S)
        
        # Query (state representation)
        query = torch.cat([user_emb, seq_emb, bundle_emb], dim=-1)  # (N, 3E)
        features = query
        
        # Pointer Network attention
        query = self.w1(query)  # (N, H)
        query = query.unsqueeze(2).repeat(1, 1, pool_size)  # (N, H, S)
        reference = self.w2(pool_emb)  # (N, H, S)
        v = self.v.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)  # (N, 1, H)
        
        logits = torch.bmm(v, torch.tanh(query + reference)).squeeze(1)  # (N, S)
        
        # Clip logits
        if self.clip:
            logits = self.clip * torch.tanh(logits)
        
        return logits, features


# ===== Value Network =====

class ValueNetwork(nn.Module):
    """Value Network for critic"""
    
    def __init__(self, input_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, features):
        """
        features: (N, input_dim) - state features
        返回: value (N,)
        """
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        value = self.fc3(x).squeeze(-1)
        return value


# ===== BYOB Agent =====

class BYOBAgent:
    """
    BYOB Agent using PPO
    基于官方源码的policy和value网络
    """
    
    def __init__(self, num_items, num_users, pool_size=20, bundle_size=3,
                 embed_dim=32, hidden_dim=128, num_heads=2, num_layers=1,
                 dropout=0.1, clip=10, encoder=True, concat=False,
                 lr=1e-3, gamma=0.99, clip_param=0.2, device=None):
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.pool_size = pool_size
        self.bundle_size = bundle_size
        self.gamma = gamma
        self.clip_param = clip_param
        
        # Policy Network (ItemPointer)
        self.policy_net = ItemPointer(
            n_user=num_users,
            n_item=num_items,
            pool_size=pool_size,
            bundle_size=bundle_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            clip=clip,
            encoder=encoder,
            concat=concat
        ).to(self.device)
        
        # Value Network
        self.value_net = ValueNetwork(
            input_dim=3 * embed_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
    
    def compute_reward(self, bundle_items, gt_bundle_items):
        """计算奖励 - Jaccard similarity"""
        pred_set = set([item for item in bundle_items if item > 0])
        gt_set = set(gt_bundle_items)
        
        if len(pred_set) == 0 or len(gt_set) == 0:
            return 0.0
        
        intersection = len(pred_set & gt_set)
        union = len(pred_set | gt_set)
        
        jaccard = intersection / union if union > 0 else 0
        return jaccard
    
    def train_episode(self, user_id, session_items, gt_bundle_items, seq_len=20):
        """训练一个episode"""
        self.policy_net.train()
        self.value_net.train()
        
        # 准备数据
        user_tensor = torch.LongTensor([user_id]).to(self.device)
        
        # Sequence: 从session中随机采样或使用全部
        if len(session_items) > seq_len:
            seq_indices = np.random.choice(len(session_items), seq_len, replace=False)
            seq_items = [session_items[i] for i in seq_indices]
        else:
            seq_items = session_items + [0] * (seq_len - len(session_items))
        
        seq_tensor = torch.LongTensor(seq_items).unsqueeze(0).to(self.device)  # (1, seq_len)
        
        # Pool: 确保包含GT items
        gt_set = set(gt_bundle_items)
        session_set = set(session_items)
        
        pool_items_list = list(gt_set)[:self.bundle_size]
        remaining = list(session_set - gt_set)
        
        if len(remaining) > 0:
            num_neg = min(len(remaining), self.pool_size - len(pool_items_list))
            pool_items_list.extend(np.random.choice(remaining, num_neg, replace=False).tolist())
        
        while len(pool_items_list) < self.pool_size:
            pool_items_list.append(0)
        
        np.random.shuffle(pool_items_list)
        
        pool_tensor = torch.LongTensor(pool_items_list).unsqueeze(0).to(self.device)  # (1, pool_size)
        available_mask = torch.BoolTensor([item > 0 for item in pool_items_list]).to(self.device)
        
        # 初始化bundle
        bundle_tensor = torch.zeros(1, self.bundle_size, dtype=torch.long).to(self.device)
        
        # 收集trajectory
        log_probs = []
        rewards = []
        values = []
        
        # 生成bundle
        for step in range(self.bundle_size):
            if not available_mask.any():
                break
            
            # Forward
            logits, features = self.policy_net(
                user_tensor, seq_tensor, pool_tensor, bundle_tensor
            )
            
            logits = logits.squeeze(0)  # (pool_size,)
            logits = logits.masked_fill(~available_mask, float('-inf'))
            
            # Value
            value = self.value_net(features)
            values.append(value.squeeze(0))
            
            # Sample action
            probs = F.softmax(logits, dim=0)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
            
            # Execute action
            selected_item = pool_tensor[0, action].item()
            bundle_tensor[0, step] = selected_item
            available_mask[action] = False
            
            # Compute reward
            current_bundle = bundle_tensor[0].cpu().numpy()
            reward = self.compute_reward(current_bundle, gt_bundle_items)
            rewards.append(reward)
        
        if len(log_probs) == 0:
            return 0.0, 0
        
        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(self.device)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # PPO update - 先更新value，再更新policy
        advantages = returns - values.detach()
        
        # Value loss
        value_loss = F.mse_loss(values, returns.detach())
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=10.0)
        self.value_optimizer.step()
        
        # Policy loss
        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantages.detach()
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages.detach()
        policy_loss = -torch.min(surr1, surr2).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.policy_optimizer.step()
        
        return sum(rewards), len(bundle_tensor[0][bundle_tensor[0] > 0])
    
    def generate_bundle(self, user_id, session_items, item_to_position, seq_len=20):
        """生成bundle (inference)"""
        self.policy_net.eval()
        self.value_net.eval()
        
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id]).to(self.device)
            
            # Sequence
            if len(session_items) > seq_len:
                seq_indices = np.random.choice(len(session_items), seq_len, replace=False)
                seq_items = [session_items[i] for i in seq_indices]
            else:
                seq_items = session_items + [0] * (seq_len - len(session_items))
            
            seq_tensor = torch.LongTensor(seq_items).unsqueeze(0).to(self.device)
            
            # Pool
            pool_items_list = session_items[:self.pool_size]
            while len(pool_items_list) < self.pool_size:
                pool_items_list.append(0)
            
            pool_tensor = torch.LongTensor(pool_items_list).unsqueeze(0).to(self.device)
            available_mask = torch.BoolTensor([item > 0 for item in pool_items_list]).to(self.device)
            
            # 初始化bundle
            bundle_tensor = torch.zeros(1, self.bundle_size, dtype=torch.long).to(self.device)
            selected_positions = []
            
            # 生成
            for step in range(self.bundle_size):
                if not available_mask.any():
                    break
                
                logits, _ = self.policy_net(
                    user_tensor, seq_tensor, pool_tensor, bundle_tensor
                )
                
                logits = logits.squeeze(0)
                logits = logits.masked_fill(~available_mask, float('-inf'))
                
                # Greedy selection
                action = logits.argmax()
                
                selected_item = pool_tensor[0, action].item()
                if selected_item > 0:
                    bundle_tensor[0, step] = selected_item
                    
                    # 找到原始位置
                    if selected_item in item_to_position:
                        selected_positions.append(item_to_position[selected_item])
                
                available_mask[action] = False
        
        return selected_positions


# ===== Main Training Function =====

def prepare_data(session_items, session_bundles, train_session_ids):
    """准备训练数据"""
    all_items = set()
    for sid, items_str in session_items.items():
        all_items.update(items_str.split(','))
    
    item_to_idx = {item: idx+1 for idx, item in enumerate(sorted(all_items))}
    
    train_sessions = [sid for sid in train_session_ids if sid in session_items]
    session_to_user_idx = {sid: idx+1 for idx, sid in enumerate(train_sessions)}
    
    training_samples = []
    for session_id in train_sessions:
        if session_id not in session_bundles:
            continue
        
        items_str = session_items[session_id]
        items = items_str.split(',')
        item_indices = [item_to_idx[item] for item in items]
        
        for bundle_tuple in session_bundles[session_id]:
            bundle_items_str = bundle_tuple[-1]
            bundle_items = bundle_items_str.split(',')
            bundle_indices = [item_to_idx[item] for item in bundle_items if item in item_to_idx]
            
            if len(bundle_indices) >= 2:
                training_samples.append({
                    'session_id': session_id,
                    'user_idx': session_to_user_idx[session_id],
                    'session_items': item_indices,
                    'gt_bundle': bundle_indices
                })
    
    return training_samples, item_to_idx, session_to_user_idx


def run_byob(config, dataset_name):
    """运行BYOB"""
    logger = Logger(config['log_path'])
    logger.info('='*60)
    logger.info('BYOB: Build Your Own Bundle (Official Source)')
    logger.info('='*60)
    
    # 加载数据
    data_path = config['data_path'] + dataset_name + '/'
    temp_path = config['temp_path'] + dataset_name + '/'
    
    session_items = np.load(f'{data_path}session_items.npy', allow_pickle=True).item()
    session_bundles = np.load(f'{data_path}session_bundles_deduplication.npy', allow_pickle=True).item()
    train_set = np.load(f'{data_path}training_set.npy', allow_pickle=True).item()
    test_set = np.load(f'{data_path}test_set.npy', allow_pickle=True).item()
    
    logger.info(f'\nDataset: {dataset_name}')
    logger.info(f'  Sessions: train={len(train_set)}, test={len(test_set)}')
    
    # 准备数据
    logger.info('\nPreparing data...')
    training_samples, item_to_idx, session_to_user_idx = prepare_data(
        session_items, session_bundles, train_set.keys()
    )
    
    num_items = len(item_to_idx)
    num_users = len(session_to_user_idx)
    
    logger.info(f'  Items: {num_items}, Users: {num_users}, Samples: {len(training_samples)}')
    
    # 初始化Agent
    logger.info('\n' + '-'*60)
    logger.info('Initializing BYOB Agent')
    logger.info('-'*60)
    
    agent = BYOBAgent(
        num_items=num_items,
        num_users=num_users,
        pool_size=20,
        bundle_size=3,
        embed_dim=32,
        hidden_dim=128,
        num_heads=2,
        num_layers=1,
        dropout=0.1,
        clip=10,
        encoder=True,
        concat=False,
        lr=1e-3,
        gamma=0.99,
        clip_param=0.2
    )
    
    logger.info(f'  Device: {agent.device}')
    logger.info(f'  Config: pool=20, bundle=3, embed=32, hidden=128')
    
    # 训练
    logger.info('\n' + '-'*60)
    logger.info('Training with PPO')
    logger.info('-'*60)
    
    epochs = 20
    for epoch in range(epochs):
        np.random.shuffle(training_samples)
        
        epoch_rewards = []
        epoch_sizes = []
        
        # 限制训练样本数量加速训练
        samples_to_train = training_samples[:min(1000, len(training_samples))]
        
        progress_bar = tqdm(samples_to_train, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for sample in progress_bar:
            try:
                reward, size = agent.train_episode(
                    user_id=sample['user_idx'],
                    session_items=sample['session_items'],
                    gt_bundle_items=sample['gt_bundle'],
                    seq_len=20
                )
                
                if size > 0:
                    epoch_rewards.append(reward)
                    epoch_sizes.append(size)
                
                if len(epoch_rewards) % 50 == 0 and len(epoch_rewards) > 0:
                    progress_bar.set_postfix({
                        'reward': f"{np.mean(epoch_rewards[-50:]):.3f}",
                        'size': f"{np.mean(epoch_sizes[-50:]):.1f}"
                    })
            except Exception as e:
                continue
        
        if len(epoch_rewards) > 0:
            logger.info(f'  Epoch {epoch+1} - Reward: {np.mean(epoch_rewards):.4f}, Size: {np.mean(epoch_sizes):.2f}')
    
    # 生成bundles
    logger.info('\n' + '-'*60)
    logger.info('Generating Bundles')
    logger.info('-'*60)
    
    bundle_results = {}
    pred_sizes = []
    errors = 0
    
    for test_session_id in tqdm(test_set.keys(), desc="Generating"):
        if test_session_id not in session_items:
            continue
        
        items_str = session_items[test_session_id]
        items = items_str.split(',')
        
        if len(items) < 2:
            continue
        
        # 映射
        item_indices = [item_to_idx.get(item, 0) for item in items]
        item_to_position = {item_to_idx[item]: i for i, item in enumerate(items) if item in item_to_idx}
        
        # 找最相似user
        test_set_items = set(items)
        best_sim = 0
        best_user_idx = 1
        
        for train_sid, user_idx in list(session_to_user_idx.items())[:100]:
            train_items = set(session_items[train_sid].split(','))
            jaccard = len(test_set_items & train_items) / max(len(test_set_items | train_items), 1)
            if jaccard > best_sim:
                best_sim = jaccard
                best_user_idx = user_idx
        
        # 生成
        try:
            positions = agent.generate_bundle(best_user_idx, item_indices, item_to_position, seq_len=20)
            
            if len(positions) >= 2:
                product_list = [f'product{pos+1}' for pos in positions]
                bundle_results[test_session_id] = {'bundle1': product_list}
                pred_sizes.append(len(product_list))
        except Exception as e:
            errors += 1
            continue
    
    logger.info(f'\nResults: {len(bundle_results)}/{len(test_set)} sessions, {errors} errors')
    if len(pred_sizes) > 0:
        logger.info(f'  Avg size: {np.mean(pred_sizes):.2f}')
    
    if len(bundle_results) == 0:
        logger.error('No bundles generated!')
        return 0.0, 0.0, 0.0
    
    # 保存
    np.save(f'{temp_path}byob_bundle_res.npy', bundle_results, allow_pickle=True)
    
    # 评估
    logger.info('\n' + '-'*60)
    logger.info('Evaluation')
    logger.info('-'*60)
    
    try:
        format_res = process_results(bundle_results)
        precision, recall, coverage = compute(session_items, session_bundles, format_res)
        
        logger.info('\n' + '='*60)
        logger.info('BYOB Results:')
        logger.info(f'  Precision: {precision:.4f}')
        logger.info(f'  Recall:    {recall:.4f}')
        logger.info(f'  Coverage:  {coverage:.4f}')
        logger.info('='*60)
        # logger.info('\nPaper (Electronic):')
        # logger.info('  Precision: 0.340, Recall: 0.294, Coverage: 0.361')
        # logger.info('='*60 + '\n')
        
        return precision, recall, coverage
    except Exception as e:
        logger.error(f'Evaluation error: {e}')
        return 0.0, 0.0, 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='electronic')
    parser.add_argument('--epochs', type=int, default=20)
    opt = parser.parse_args()
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    run_byob(config, opt.dataset)