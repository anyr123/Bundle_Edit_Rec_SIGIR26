# bbpr_v2.py
import numpy as np
import yaml
from tqdm import tqdm
import argparse
from utils.logger import Logger
from utils.functions import process_results
from utils.metrics import compute

class BPRMF:
    """Bayesian Personalized Ranking Matrix Factorization"""
    
    def __init__(self, num_users, num_items, embedding_size=20, learning_rate=0.01, reg=0.01):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.reg = reg
        
        # 随机初始化embeddings
        np.random.seed(42)
        self.user_embeddings = np.random.normal(0, 0.01, (num_users, embedding_size))
        self.item_embeddings = np.random.normal(0, 0.01, (num_items, embedding_size))
        self.item_bias = np.zeros(num_items)
    
    def predict_score(self, user_idx, item_idx):
        """预测user对item的分数"""
        score = np.dot(self.user_embeddings[user_idx], self.item_embeddings[item_idx])
        score += self.item_bias[item_idx]
        return score
    
    def train(self, interactions, epochs=20, neg_samples=2):
        """
        训练BPR模型
        interactions: list of (user_idx, item_idx) tuples
        """
        print(f"Training BPRMF: {self.num_users} users, {self.num_items} items, {len(interactions)} interactions")
        
        for epoch in range(epochs):
            np.random.shuffle(interactions)
            
            for user_idx, pos_item_idx in tqdm(interactions, desc=f"Epoch {epoch+1}/{epochs}"):
                # 负采样
                for _ in range(neg_samples):
                    neg_item_idx = np.random.randint(0, self.num_items)
                    # 确保负样本不在正样本中
                    # (简化处理，可能会采样到正样本，但概率很小)
                    
                    # BPR更新
                    pos_score = self.predict_score(user_idx, pos_item_idx)
                    neg_score = self.predict_score(user_idx, neg_item_idx)
                    
                    # sigmoid
                    diff = pos_score - neg_score
                    exp_diff = np.exp(-diff)
                    sigmoid = exp_diff / (1 + exp_diff)
                    
                    # 更新user embedding
                    user_grad = sigmoid * (self.item_embeddings[pos_item_idx] - self.item_embeddings[neg_item_idx])
                    user_grad -= self.reg * self.user_embeddings[user_idx]
                    self.user_embeddings[user_idx] += self.learning_rate * user_grad
                    
                    # 更新positive item embedding
                    pos_grad = sigmoid * self.user_embeddings[user_idx]
                    pos_grad -= self.reg * self.item_embeddings[pos_item_idx]
                    self.item_embeddings[pos_item_idx] += self.learning_rate * pos_grad
                    
                    # 更新negative item embedding
                    neg_grad = -sigmoid * self.user_embeddings[user_idx]
                    neg_grad -= self.reg * self.item_embeddings[neg_item_idx]
                    self.item_embeddings[neg_item_idx] += self.learning_rate * neg_grad
                    
                    # 更新bias
                    self.item_bias[pos_item_idx] += self.learning_rate * (sigmoid - self.reg * self.item_bias[pos_item_idx])
                    self.item_bias[neg_item_idx] += self.learning_rate * (-sigmoid - self.reg * self.item_bias[neg_item_idx])


class BBPR:
    """Bundle Generation using BPRMF"""
    
    def __init__(self, bprmf_model, item_id_to_idx):
        self.bprmf = bprmf_model
        self.item_id_to_idx = item_id_to_idx
        self.idx_to_item_id = {v: k for k, v in item_id_to_idx.items()}
    
    def get_item_neighbors(self, item_idx, k=10):
        """基于embedding相似度找到最相似的k个items"""
        item_emb = self.bprmf.item_embeddings[item_idx]
        
        # 计算与所有items的余弦相似度
        similarities = np.dot(self.bprmf.item_embeddings, item_emb)
        similarities /= (np.linalg.norm(self.bprmf.item_embeddings, axis=1) * np.linalg.norm(item_emb) + 1e-8)
        
        # 获取top-k (排除自己)
        top_indices = np.argsort(similarities)[::-1][1:k+1]
        return top_indices
    
    def generate_bundle(self, user_idx, candidate_item_indices, init_size=3):
        """
        为一个user生成一个bundle
        candidate_item_indices: 候选items的索引列表
        """
        if len(candidate_item_indices) < 2:
            return []
        
        # 1. 根据预测分数选择top-init_size个items作为初始bundle
        scores = []
        for item_idx in candidate_item_indices:
            score = self.bprmf.predict_score(user_idx, item_idx)
            scores.append(score)
        
        scores = np.array(scores)
        top_k = min(init_size, len(candidate_item_indices))
        init_indices = np.argsort(scores)[::-1][:top_k]
        
        bundle = [candidate_item_indices[i] for i in init_indices]
        
        return bundle
    
    def generate_bundles_for_session(self, user_idx, session_item_ids):
        """
        为一个session生成多个bundles
        session_item_ids: session中的item id列表
        """
        # 转换为索引
        item_indices = []
        valid_item_ids = []
        for item_id in session_item_ids:
            if item_id in self.item_id_to_idx:
                item_indices.append(self.item_id_to_idx[item_id])
                valid_item_ids.append(item_id)
        
        if len(item_indices) < 2:
            return {}
        
        # 计算item之间的相似度矩阵
        n = len(item_indices)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                emb_i = self.bprmf.item_embeddings[item_indices[i]]
                emb_j = self.bprmf.item_embeddings[item_indices[j]]
                
                sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-8)
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim
        
        # 使用贪心聚类生成bundles
        bundles = []
        used_items = set()
        
        # 生成多个bundles
        while len(used_items) < len(item_indices):
            # 找到未使用的item中得分最高的
            available_indices = [i for i in range(len(item_indices)) if i not in used_items]
            if len(available_indices) < 2:
                break
            
            # 计算每个未使用item的得分
            scores = []
            for idx in available_indices:
                score = self.bprmf.predict_score(user_idx, item_indices[idx])
                scores.append(score)
            
            # 选择得分最高的作为种子
            seed_idx = available_indices[np.argmax(scores)]
            current_bundle = [seed_idx]
            used_items.add(seed_idx)
            
            # 贪心添加相似的items
            for _ in range(10):  # 最多10个items per bundle
                if len(available_indices) <= len(current_bundle):
                    break
                
                # 找到与当前bundle最相似的未使用item
                best_sim = -1
                best_idx = None
                
                for idx in available_indices:
                    if idx in used_items:
                        continue
                    
                    # 计算与bundle中所有items的平均相似度
                    avg_sim = np.mean([similarity_matrix[idx][b] for b in current_bundle])
                    
                    if avg_sim > best_sim:
                        best_sim = avg_sim
                        best_idx = idx
                
                # 如果相似度太低，停止添加
                if best_idx is None or best_sim < 0.1:
                    break
                
                current_bundle.append(best_idx)
                used_items.add(best_idx)
            
            # 只保留至少2个items的bundle
            if len(current_bundle) >= 2:
                bundle_item_ids = [valid_item_ids[i] for i in current_bundle]
                bundles.append(bundle_item_ids)
        
        # 转换为要求的格式
        result = {}
        for i, bundle_item_ids in enumerate(bundles):
            # 转换为product1, product2格式
            product_list = []
            for item_id in bundle_item_ids:
                if item_id in session_item_ids:
                    pos = session_item_ids.index(item_id) + 1
                    product_list.append(f'product{pos}')
            
            if len(product_list) >= 2:
                result[f'bundle{i+1}'] = product_list
        
        return result


def prepare_data(session_items, train_session_ids):
    """
    准备训练数据
    返回:
    - interactions: [(user_idx, item_idx), ...]
    - item_id_to_idx: {item_id: idx}
    - session_id_to_user_idx: {session_id: user_idx}
    """
    # 收集所有items
    all_items = set()
    for session_id, items_str in session_items.items():
        items = items_str.split(',')
        all_items.update(items)
    
    # 创建item映射
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(sorted(all_items))}
    
    # 创建user(session)映射 - 只用训练session
    train_sessions = {}
    for session_id in train_session_ids:
        if session_id in session_items:
            train_sessions[session_id] = session_items[session_id]
    
    session_id_to_user_idx = {session_id: idx for idx, session_id in enumerate(train_sessions.keys())}
    
    # 创建interactions
    interactions = []
    for session_id, items_str in train_sessions.items():
        user_idx = session_id_to_user_idx[session_id]
        items = items_str.split(',')
        
        for item_id in items:
            if item_id in item_id_to_idx:
                item_idx = item_id_to_idx[item_id]
                interactions.append((user_idx, item_idx))
    
    return interactions, item_id_to_idx, session_id_to_user_idx


def run_bbpr(config, dataset_name):
    """运行BBPR baseline"""
    logger = Logger(config['log_path'])
    logger.info('='*50)
    logger.info('Starting BBPR Baseline')
    logger.info('='*50)
    
    # 加载数据
    data_path = config['data_path'] + dataset_name + '/'
    temp_path = config['temp_path'] + dataset_name + '/'
    
    session_items = np.load(f'{data_path}session_items.npy', allow_pickle=True).item()
    session_bundles = np.load(f'{data_path}session_bundles_deduplication.npy', allow_pickle=True).item()
    train_set = np.load(f'{data_path}training_set.npy', allow_pickle=True).item()
    test_set = np.load(f'{data_path}test_set.npy', allow_pickle=True).item()
    
    logger.info(f'Dataset: {dataset_name}')
    logger.info(f'Total sessions: {len(session_items)}')
    logger.info(f'Train sessions: {len(train_set)}')
    logger.info(f'Test sessions: {len(test_set)}')
    
    # 准备训练数据
    logger.info('\nPreparing training data...')
    interactions, item_id_to_idx, session_id_to_user_idx = prepare_data(session_items, train_set.keys())
    
    num_users = len(session_id_to_user_idx)
    num_items = len(item_id_to_idx)
    
    logger.info(f'Number of users (train sessions): {num_users}')
    logger.info(f'Number of items: {num_items}')
    logger.info(f'Number of interactions: {len(interactions)}')
    
    # 训练BPRMF
    logger.info('\nTraining BPRMF model...')
    bprmf = BPRMF(
        num_users=num_users,
        num_items=num_items,
        embedding_size=20,
        learning_rate=0.01,
        reg=0.01
    )
    bprmf.train(interactions, epochs=20, neg_samples=2)
    
    # 生成bundles
    logger.info('\nGenerating bundles for test sessions...')
    bbpr = BBPR(bprmf, item_id_to_idx)
    
    bundle_results = {}
    
    for test_session_id in tqdm(test_set.keys()):
        # 获取session的items
        if test_session_id not in session_items:
            continue
        
        session_item_ids = session_items[test_session_id].split(',')
        
        if len(session_item_ids) < 2:
            continue
        
        # 找一个最相似的训练user
        # 基于Jaccard相似度
        test_item_set = set(session_item_ids)
        best_similarity = 0
        best_user_idx = 0
        
        for train_session_id, user_idx in session_id_to_user_idx.items():
            train_item_ids = session_items[train_session_id].split(',')
            train_item_set = set(train_item_ids)
            
            intersection = len(test_item_set & train_item_set)
            union = len(test_item_set | train_item_set)
            
            if union > 0:
                similarity = intersection / union
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_user_idx = user_idx
        
        # 生成bundles
        bundles = bbpr.generate_bundles_for_session(best_user_idx, session_item_ids)
        
        if len(bundles) > 0:
            bundle_results[test_session_id] = bundles
    
    logger.info(f'\nGenerated bundles for {len(bundle_results)}/{len(test_set)} test sessions')
    
    # 保存结果
    np.save(f'{temp_path}bbpr_bundle_res.npy', bundle_results, allow_pickle=True)
    logger.info(f'Results saved to {temp_path}bbpr_bundle_res.npy')
    
    # 评估
    logger.info('\nEvaluating results...')
    format_res = process_results(bundle_results)
    precision, recall, coverage = compute(session_items, session_bundles, format_res)
    
    logger.info('\n' + '='*50)
    logger.info('BBPR Results:')
    logger.info(f'  Precision: {precision:.4f}')
    logger.info(f'  Recall:    {recall:.4f}')
    logger.info(f'  Coverage:  {coverage:.4f}')
    logger.info('='*50)
    
    return precision, recall, coverage


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='electronic', 
                       choices=['electronic', 'clothing', 'food'])
    opt = parser.parse_args()
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    run_bbpr(config, opt.dataset)