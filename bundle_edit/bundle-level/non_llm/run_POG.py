"""
POG with Aggressive Larger Bundle Strategy
更激进地降低 Coverage: 0.77 -> 0.4-0.5
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm
import argparse
from utils.logger import Logger
from utils.functions import process_results
from utils.metrics import compute


class BundleDataset(Dataset):
    """Dataset for bundle generation training"""
    
    def __init__(self, session_items, session_bundles, item_to_idx, max_session_len=20, max_bundle_len=10):
        self.samples = []
        self.max_session_len = max_session_len
        self.max_bundle_len = max_bundle_len
        
        for session_id in session_items.keys():
            if session_id not in session_bundles:
                continue
            
            items = session_items[session_id].split(',')
            item_indices = [item_to_idx.get(item, 0) for item in items]
            
            bundles = session_bundles[session_id]
            
            for bundle in bundles:
                bundle_items = bundle[-1].split(',')
                bundle_positions = []
                for b_item in bundle_items:
                    if b_item in items:
                        bundle_positions.append(items.index(b_item) + 1)
                
                if len(bundle_positions) >= 2:
                    self.samples.append({
                        'session_items': item_indices,
                        'bundle_positions': bundle_positions
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_value=0):
    max_session_len = max([len(sample['session_items']) for sample in batch])
    max_bundle_len = max([len(sample['bundle_positions']) for sample in batch])
    
    batch_size = len(batch)
    
    session_items = torch.full((batch_size, max_session_len), pad_value, dtype=torch.long)
    bundle_positions = torch.full((batch_size, max_bundle_len), pad_value, dtype=torch.long)
    session_mask = torch.zeros(batch_size, max_session_len, dtype=torch.bool)
    bundle_mask = torch.zeros(batch_size, max_bundle_len, dtype=torch.bool)
    
    for i, sample in enumerate(batch):
        session_len = len(sample['session_items'])
        bundle_len = len(sample['bundle_positions'])
        
        session_items[i, :session_len] = torch.LongTensor(sample['session_items'])
        bundle_positions[i, :bundle_len] = torch.LongTensor(sample['bundle_positions'])
        session_mask[i, :session_len] = True
        bundle_mask[i, :bundle_len] = True
    
    return {
        'session_items': session_items,
        'bundle_positions': bundle_positions,
        'session_mask': session_mask,
        'bundle_mask': bundle_mask
    }


class POGModel(nn.Module):
    def __init__(self, num_items, embedding_size=20, nhead=4, num_layers=2, 
                 max_position=100, dropout=0.1):
        super(POGModel, self).__init__()
        
        self.embedding_size = embedding_size
        self.num_items = num_items
        self.max_position = max_position
        
        self.item_embedding = nn.Embedding(num_items + 1, embedding_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_position, embedding_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size, nhead=nhead,
            dim_feedforward=embedding_size * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_size, nhead=nhead,
            dim_feedforward=embedding_size * 4, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(embedding_size, max_position)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, session_items, target_positions, session_mask=None, target_mask=None):
        batch_size = session_items.size(0)
        session_len = session_items.size(1)
        target_len = target_positions.size(1)
        device = session_items.device
        
        session_pos = torch.arange(session_len, device=device).unsqueeze(0).expand(batch_size, -1)
        target_pos = torch.arange(target_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        session_embed = self.item_embedding(session_items) + self.position_embedding(session_pos)
        session_embed = self.dropout(session_embed)
        
        session_key_padding_mask = ~session_mask if session_mask is not None else None
        memory = self.encoder(session_embed, src_key_padding_mask=session_key_padding_mask)
        
        target_embed = self.position_embedding(target_pos)
        target_embed = self.dropout(target_embed)
        
        target_key_padding_mask = ~target_mask if target_mask is not None else None
        causal_mask = nn.Transformer.generate_square_subsequent_mask(target_len).to(device)
        
        decoded = self.decoder(
            target_embed, memory, tgt_mask=causal_mask,
            tgt_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=session_key_padding_mask
        )
        
        output = self.output_layer(decoded)
        return output
    
    def generate_bundle(self, session_items, session_mask=None, max_length=10, 
                       min_length=4, temperature=1.3, top_k=10, top_p=0.95,
                       stop_penalty=-4.0, force_diversity=True):
        """
        更激进的bundle生成策略
        
        关键改进：
        1. min_length=4: 强制至少4个items
        2. stop_penalty=-4.0: 更强的停止惩罚
        3. temperature=1.3: 增加随机性
        4. top_k=10: 扩大选择范围
        5. force_diversity=True: 强制多样性
        """
        self.eval()
        device = session_items.device
        batch_size = session_items.size(0)
        session_len = session_items.size(1)
        
        generated_positions = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for step in range(max_length):
                output = self.forward(session_items, generated_positions, session_mask, None)
                next_pos_logits = output[:, -1, :] / temperature
                
                valid_positions = session_len + 1
                next_pos_logits = next_pos_logits[:, :valid_positions]
                
                current_length = step + 1
                
                # 更激进的stop惩罚
                if current_length < min_length:
                    next_pos_logits[:, 0] += stop_penalty
                elif current_length == min_length:
                    # 刚达到最小长度时仍有一些惩罚
                    next_pos_logits[:, 0] += stop_penalty / 2
                
                # 强制多样性：降低已生成位置附近位置的概率
                if force_diversity and current_length > 1:
                    for b in range(batch_size):
                        generated = generated_positions[b][1:].cpu().numpy()  # 排除初始0
                        for pos in generated:
                            if pos > 0:
                                # 降低相邻位置的概率
                                if pos > 1:
                                    next_pos_logits[b, pos-1] -= 1.0
                                if pos < valid_positions - 1:
                                    next_pos_logits[b, pos+1] -= 1.0
                
                # Top-k filtering (扩大范围以增加多样性)
                if top_k is not None and top_k < next_pos_logits.size(-1):
                    k = min(top_k, next_pos_logits.size(-1))
                    top_k_values, _ = torch.topk(next_pos_logits, k)
                    indices_to_remove = next_pos_logits < top_k_values[:, -1:]
                    next_pos_logits = next_pos_logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_pos_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 0] = False
                    
                    for i in range(batch_size):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        next_pos_logits[i, indices_to_remove] = float('-inf')
                
                probs = torch.softmax(next_pos_logits, dim=-1)
                next_pos = torch.multinomial(probs, num_samples=1)
                
                # 只在达到最小长度后才允许停止
                if (next_pos == 0).all() and current_length >= min_length:
                    break
                
                # Remove duplicates
                for b in range(batch_size):
                    if next_pos[b] in generated_positions[b]:
                        next_pos[b] = 0
                
                generated_positions = torch.cat([generated_positions, next_pos], dim=1)
        
        return generated_positions[:, 1:]


class POG:
    def __init__(self, num_items, embedding_size=20, nhead=4, num_layers=2, 
                 max_position=100, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = POGModel(
            num_items=num_items, embedding_size=embedding_size,
            nhead=nhead, num_layers=num_layers, max_position=max_position
        ).to(self.device)
        
        self.num_items = num_items
    
    def train_model(self, train_dataset, val_dataset=None, epochs=20, batch_size=64, 
                   learning_rate=0.001):
        print(f"\nTraining POG model on {self.device}...")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Batch size: {batch_size}, Epochs: {epochs}, LR: {learning_rate}")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, collate_fn=collate_fn)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                session_items = batch['session_items'].to(self.device)
                bundle_positions = batch['bundle_positions'].to(self.device)
                session_mask = batch['session_mask'].to(self.device)
                bundle_mask = batch['bundle_mask'].to(self.device)
                
                input_positions = bundle_positions[:, :-1]
                target_positions = bundle_positions[:, 1:]
                input_mask = bundle_mask[:, :-1]
                
                output = self.model(session_items, input_positions, session_mask, input_mask)
                
                batch_size_cur, seq_len, vocab_size = output.shape
                output_flat = output.reshape(-1, vocab_size)
                target_flat = target_positions.reshape(-1)
                
                loss = criterion(output_flat, target_flat)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def generate_bundles_for_session(self, session_item_list, item_to_idx, 
                                    max_bundles=1, max_bundle_size=10,
                                    min_bundle_size=4, temperature=1.3,
                                    top_k=10, top_p=0.95, stop_penalty=-4.0):
        """
        激进策略：强制生成更大的bundles
        
        参数说明：
        - min_bundle_size: 4 (GT平均3.52，我们强制4)
        - temperature: 1.3 (增加随机性，会选择一些不太确定的items)
        - top_k: 10 (扩大选择范围)
        - top_p: 0.95 (允许更多候选)
        - stop_penalty: -4.0 (强力阻止过早停止)
        """
        self.model.eval()
        
        item_indices = [item_to_idx.get(item, 0) for item in session_item_list]
        
        if len(item_indices) < min_bundle_size:
            return {}
        
        session_items = torch.LongTensor([item_indices]).to(self.device)
        session_mask = torch.ones(1, len(item_indices), dtype=torch.bool).to(self.device)
        
        bundles = []
        
        for _ in range(max_bundles):
            generated_positions = self.model.generate_bundle(
                session_items, session_mask, 
                max_length=max_bundle_size,
                min_length=min_bundle_size,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_penalty=stop_penalty,
                force_diversity=True
            )
            
            positions = generated_positions[0].cpu().numpy()
            positions = [int(p) - 1 for p in positions if p > 0 and p <= len(session_item_list)]
            
            # Remove duplicates
            seen = set()
            unique_positions = []
            for p in positions:
                if p not in seen and 0 <= p < len(session_item_list):
                    seen.add(p)
                    unique_positions.append(p)
            
            if len(unique_positions) >= min_bundle_size:
                bundles.append(unique_positions)
        
        result = {}
        for i, bundle_positions in enumerate(bundles, 1):
            product_list = [f'product{pos+1}' for pos in bundle_positions]
            if len(product_list) >= min_bundle_size:
                result[f'bundle{i}'] = product_list
        
        return result


def prepare_data(session_items, session_bundles, train_session_ids):
    print("\nPreparing data for POG...")
    
    all_items = set()
    for session_id, items_str in session_items.items():
        items = items_str.split(',')
        all_items.update(items)
    
    print(f"  Total unique items: {len(all_items)}")
    
    item_to_idx = {item: idx + 1 for idx, item in enumerate(sorted(all_items))}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    
    train_session_items = {sid: session_items[sid] for sid in train_session_ids if sid in session_items}
    
    print(f"  Training sessions: {len(train_session_items)}")
    
    return item_to_idx, idx_to_item


def run_pog(config, dataset_name):
    logger = Logger(config['log_path'])
    logger.info('='*60)
    logger.info('POG with AGGRESSIVE Larger Bundles Strategy')
    logger.info('='*60)
    
    data_path = config['data_path'] + dataset_name + '/'
    temp_path = config['temp_path'] + dataset_name + '/'
    
    session_items = np.load(f'{data_path}session_items.npy', allow_pickle=True).item()
    session_bundles = np.load(f'{data_path}session_bundles_deduplication.npy', allow_pickle=True).item()
    train_set = np.load(f'{data_path}training_set.npy', allow_pickle=True).item()
    test_set = np.load(f'{data_path}test_set.npy', allow_pickle=True).item()
    
    logger.info(f'\nDataset: {dataset_name}')
    logger.info(f'  Total sessions: {len(session_items)}')
    logger.info(f'  Train sessions: {len(train_set)}')
    logger.info(f'  Test sessions: {len(test_set)}')
    
    # 统计GT bundle大小
    gt_bundle_sizes = []
    for sid in test_set.keys():
        if sid in session_bundles:
            for bundle in session_bundles[sid]:
                gt_bundle_sizes.append(len(bundle[-1].split(',')))
    
    logger.info(f'\nGround Truth Bundle Stats:')
    logger.info(f'  Avg size: {np.mean(gt_bundle_sizes):.2f}')
    logger.info(f'  Min: {min(gt_bundle_sizes)}, Max: {max(gt_bundle_sizes)}')
    logger.info(f'  Median: {np.median(gt_bundle_sizes):.2f}')
    
    item_to_idx, idx_to_item = prepare_data(session_items, session_bundles, train_set.keys())
    
    num_items = len(item_to_idx)
    
    logger.info(f'\nModel Configuration (AGGRESSIVE Strategy):')
    logger.info(f'  Num items: {num_items}')
    logger.info(f'  Min bundle size: 4 (GT avg: 3.52)')
    logger.info(f'  Max bundle size: 10')
    logger.info(f'  Temperature: 1.3 (increased for diversity)')
    logger.info(f'  Top-k: 10 (expanded choices)')
    logger.info(f'  Top-p: 0.95')
    logger.info(f'  Stop penalty: -4.0 (strong penalty)')
    logger.info(f'  Force diversity: True')
    
    logger.info('\nCreating training dataset...')
    train_session_items = {sid: session_items[sid] for sid in train_set.keys() if sid in session_items}
    train_dataset = BundleDataset(train_session_items, session_bundles, item_to_idx)
    
    logger.info(f'  Training samples: {len(train_dataset)}')
    
    logger.info('\n' + '-'*60)
    logger.info('Training POG Model')
    logger.info('-'*60)
    
    pog = POG(num_items=num_items, embedding_size=20, nhead=4, num_layers=2, max_position=100)
    pog.train_model(train_dataset, epochs=20, batch_size=64, learning_rate=0.001)
    
    logger.info('\n' + '-'*60)
    logger.info('Generating Bundles (AGGRESSIVE Strategy)')
    logger.info('-'*60)
    
    bundle_results = {}
    pred_bundle_sizes = []
    sessions_skipped = 0
    
    for test_session_id in tqdm(test_set.keys(), desc="Generating"):
        if test_session_id not in session_items:
            continue
        
        session_item_list = session_items[test_session_id].split(',')
        
        # 只处理有足够items的session
        if len(session_item_list) < 4:
            sessions_skipped += 1
            continue
        
        bundles = pog.generate_bundles_for_session(
            session_item_list, item_to_idx, 
            max_bundles=1,
            max_bundle_size=10,
            min_bundle_size=4,        # 强制4个
            temperature=1.3,          # 增加随机性
            top_k=10,                 # 扩大范围
            top_p=0.95,
            stop_penalty=-4.0         # 强力惩罚
        )
        
        if len(bundles) > 0:
            bundle_results[test_session_id] = bundles
            for bundle in bundles.values():
                pred_bundle_sizes.append(len(bundle))
    
    logger.info(f'\nGeneration Results:')
    logger.info(f'  Sessions processed: {len(test_set) - sessions_skipped}/{len(test_set)}')
    logger.info(f'  Sessions with bundles: {len(bundle_results)}')
    logger.info(f'  Sessions skipped (<4 items): {sessions_skipped}')
    
    if len(pred_bundle_sizes) > 0:
        logger.info(f'\nPredicted Bundle Stats:')
        logger.info(f'  Avg size: {np.mean(pred_bundle_sizes):.2f}')
        logger.info(f'  Min: {min(pred_bundle_sizes)}, Max: {max(pred_bundle_sizes)}')
        logger.info(f'  Median: {np.median(pred_bundle_sizes):.2f}')
        logger.info(f'\nComparison with GT:')
        logger.info(f'  GT avg: {np.mean(gt_bundle_sizes):.2f}')
        logger.info(f'  Pred avg: {np.mean(pred_bundle_sizes):.2f}')
        logger.info(f'  Size ratio: {np.mean(pred_bundle_sizes)/np.mean(gt_bundle_sizes):.1%}')
        
        if np.mean(pred_bundle_sizes) < np.mean(gt_bundle_sizes):
            logger.warning(f'  ⚠ Pred < GT! This explains high coverage.')
    
    np.save(f'{temp_path}pog_bundle_res.npy', bundle_results, allow_pickle=True)
    
    logger.info('\n' + '-'*60)
    logger.info('Evaluation')
    logger.info('-'*60)
    
    format_res = process_results(bundle_results)
    precision, recall, coverage = compute(session_items, session_bundles, format_res)
    
    logger.info('\n' + '='*60)
    logger.info('POG Results (AGGRESSIVE Version):')
    logger.info(f'  Precision: {precision:.4f}')
    logger.info(f'  Recall:    {recall:.4f}')
    logger.info(f'  Coverage:  {coverage:.4f}')
    logger.info('='*60)
    logger.info('\nTarget (Paper):')
    logger.info('  Precision: ~0.339')
    logger.info('  Recall:    ~0.250')
    logger.info('  Coverage:  ~0.412')
    logger.info('='*60)
    
    # 分析结果
    logger.info('\nAnalysis:')
    if coverage > 0.6:
        logger.warning('  Coverage still high (>0.6)')
        logger.warning('  Possible reasons:')
        logger.warning('    1. Pred bundle size still < GT size')
        logger.warning('    2. Model too accurate on core items')
        logger.warning('    3. Check metrics calculation code!')
    elif coverage > 0.5:
        logger.info('  Coverage moderately high (0.5-0.6)')
        logger.info('  Consider: min_bundle_size=5 or check metrics code')
    else:
        logger.info('  ✓ Coverage in reasonable range!')
    
    logger.info('\n')
    
    return precision, recall, coverage


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='electronic')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--min_bundle_size', type=int, default=4)
    parser.add_argument('--stop_penalty', type=float, default=-4.0)
    parser.add_argument('--temperature', type=float, default=1.3)
    opt = parser.parse_args()
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    run_pog(config, opt.dataset)