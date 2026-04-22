"""
T5 Bundle Generation - 集成版本
训练、生成和评估一体化脚本
使用原始代码中的compute函数进行评估
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import json
import re
import os
import argparse
from typing import List, Dict, Tuple


# ============================================
# 评估函数 (从原始代码中提取)
# ============================================

def process_results(bundle_res):
    """
    处理生成的bundles，移除只包含1个产品的bundles
    
    Args:
        bundle_res: {session_id: {'bundle1': ['product1', 'product2'], ...}}
    
    Returns:
        format_res: {session_id: {bundle_id: [item_indices]}}
    """
    format_res = {}
    
    for session_id, bundles in bundle_res.items():
        if not isinstance(bundles, dict):
            continue
        
        session_bundles = {}
        for bundle_id, items in bundles.items():
            # 确保items是列表且长度>=2
            if isinstance(items, list) and len(items) >= 2:
                # 从'product1'格式提取索引
                try:
                    indices = []
                    for item in items:
                        if isinstance(item, str) and item.startswith('product'):
                            idx = int(item.replace('product', ''))
                            indices.append(idx)
                        elif isinstance(item, int):
                            indices.append(item)
                    
                    if len(indices) >= 2:
                        session_bundles[bundle_id] = indices
                except (ValueError, AttributeError) as e:
                    continue
        
        if len(session_bundles) > 0:
            format_res[session_id] = session_bundles
    
    return format_res


def compute(session_items, session_bundles, predictions):
    """
    计算Precision, Recall和Coverage指标
    这是从原始代码中提取的compute函数
    
    Args:
        session_items: {session_id: 'item1,item2,item3'}
        session_bundles: {session_id: [(freq, intent, 'item1,item2')]}
        predictions: {session_id: {bundle_id: [item_indices]}}
    
    Returns:
        (session_precision, session_recall, coverage)
    """
    session_precision = 0
    session_recall = 0
    coverage = 0
    
    for session_id, pred_bundles in predictions.items():
        if session_id not in session_bundles:
            continue
        
        # 获取该session的所有items
        items_session = session_items[session_id].split(',')
        # 获取ground truth bundles
        ground_truth_bundles = session_bundles[session_id]
        
        # 转换预测bundles为item ID集合
        pred_bundle_sets = []
        for bundle_id, item_indices in pred_bundles.items():
            try:
                # item_indices是产品编号列表（1-based）
                # 转换为0-based索引并获取实际item ID
                item_ids = set()
                for idx in item_indices:
                    if isinstance(idx, int) and 1 <= idx <= len(items_session):
                        item_ids.add(items_session[idx - 1])
                
                if len(item_ids) >= 2:
                    pred_bundle_sets.append(item_ids)
            except (ValueError, IndexError, TypeError) as e:
                continue
        
        if len(pred_bundle_sets) == 0:
            continue
        
        # 转换ground truth bundles为item ID集合
        gt_bundle_sets = []
        for bundle_data in ground_truth_bundles:
            # bundle_data[-1]是逗号分隔的item ID字符串
            item_ids = set(bundle_data[-1].split(','))
            gt_bundle_sets.append(item_ids)
        
        # 计算session-level的precision和recall
        matched_pred = 0
        matched_gt = 0
        coverage_scores = []
        
        # 检查预测的bundles有多少是正确的
        for pred_set in pred_bundle_sets:
            for gt_set in gt_bundle_sets:
                if pred_set <= gt_set:  # pred是gt的子集
                    matched_pred += 1
                    # 计算coverage
                    cov = len(pred_set) / len(gt_set)
                    coverage_scores.append(cov)
                    break
        
        # 检查ground truth有多少被预测到
        for gt_set in gt_bundle_sets:
            for pred_set in pred_bundle_sets:
                if pred_set <= gt_set:
                    matched_gt += 1
                    break
        
        # 累加session-level指标
        if len(pred_bundle_sets) > 0:
            session_precision += matched_pred / len(pred_bundle_sets)
        
        if len(gt_bundle_sets) > 0:
            session_recall += matched_gt / len(gt_bundle_sets)
        
        if len(coverage_scores) > 0:
            coverage += np.mean(coverage_scores)
    
    # 计算平均值
    num_sessions = len(predictions)
    if num_sessions == 0:
        return 0.0, 0.0, 0.0
    
    session_precision /= num_sessions
    session_recall /= num_sessions
    coverage /= num_sessions
    
    return session_precision, session_recall, coverage


# ============================================
# T5模型相关类
# ============================================

class BundleDataset(Dataset):
    """Bundle Generation数据集"""
    
    def __init__(self, sessions, tokenizer, max_source_length=512, max_target_length=128):
        self.sessions = sessions
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.session_ids = list(sessions.keys())
    
    def __len__(self):
        return len(self.session_ids)
    
    def __getitem__(self, idx):
        session_id = self.session_ids[idx]
        session_data = self.sessions[session_id]
        
        # 构建输入文本
        items = session_data['items']
        source_text = self._format_items_as_input(items)
        
        # 构建目标文本
        bundles = session_data['bundles']
        target_text = self._format_bundles_as_output(bundles)
        
        # Tokenize
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': labels,
            'session_id': session_id
        }
    
    def _format_items_as_input(self, items: List[str]) -> str:
        formatted_items = []
        for idx, item in enumerate(items, 1):
            formatted_items.append(f"product{idx}: {item}")
        input_text = "Generate bundles from products: " + ", ".join(formatted_items)
        return input_text
    
    def _format_bundles_as_output(self, bundles: List[List[int]]) -> str:
        bundle_dict = {}
        for idx, bundle in enumerate(bundles, 1):
            products = [f"product{item_idx+1}" for item_idx in bundle]
            bundle_dict[f"bundle{idx}"] = products
        output_text = str(bundle_dict).replace("'", '"')
        return output_text


class T5BundleGenerator:
    """T5 Bundle Generation模型"""
    
    def __init__(self, model_name='t5-base', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # 如果是t5-base，使用本地路径
        if model_name == 't5-base':
            local_model_path = '/root/.cache/modelscope/hub/models/zjx200715/T5-base'
            print(f"Loading model from local path: {local_model_path}")
            self.tokenizer = T5Tokenizer.from_pretrained(local_model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(local_model_path)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        self.model.to(self.device)
        
        print(f"✓ Model loaded: {model_name}")
        print(f"✓ Total parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        print(f"✓ Device: {self.device}")
    
    def train(self, train_dataset, val_dataset=None, 
              batch_size=4, learning_rate=0.00005, epochs=10,
              save_path='t5_bundle_model.pt'):
        """训练模型"""
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4
        )
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
                
                pbar.set_postfix({'loss': train_loss / train_steps})
            
            avg_train_loss = train_loss / train_steps
            print(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_dataset is not None:
                val_loss = self.evaluate(val_dataset, batch_size)
                print(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(save_path)
                    print(f"✓ Model saved to {save_path}")
    
    def evaluate(self, dataset, batch_size=4):
        """评估模型"""
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                total_steps += 1
        
        return total_loss / total_steps
    
    def generate_bundles(self, items: List[str], num_beams=5, max_length=128) -> Dict:
        """为给定商品列表生成bundles"""
        self.model.eval()
        
        # 格式化输入
        formatted_items = []
        for idx, item in enumerate(items, 1):
            formatted_items.append(f"product{idx}: {item}")
        
        input_text = "Generate bundles from products: " + ", ".join(formatted_items)
        
        # Tokenize
        input_encoding = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_encoding['input_ids'],
                attention_mask=input_encoding['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse bundles
        try:
            bundles = self._parse_bundles(generated_text)
        except Exception as e:
            bundles = {}
        
        return bundles
    
    def _parse_bundles(self, text: str) -> Dict:
        """解析生成的bundle文本"""
        try:
            bundles = eval(text)
            if isinstance(bundles, dict):
                return bundles
        except:
            pass
        
        # 使用正则表达式提取
        bundle_dict = {}
        pattern = r"['\"]bundle(\d+)['\"]:\s*\[(.*?)\]"
        matches = re.findall(pattern, text)
        
        for bundle_num, items_str in matches:
            items = re.findall(r"['\"]?(product\d+)['\"]?", items_str)
            if items:
                bundle_dict[f"bundle{bundle_num}"] = items
        
        return bundle_dict
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"✓ Model loaded from {path}")


# ============================================
# 数据准备函数
# ============================================

def prepare_data_from_npy(data_path: str) -> Dict:
    """从npy文件准备数据"""
    print(f"\n{'='*60}")
    print("Loading data...")
    print(f"{'='*60}")
    
    train_set = np.load(f'{data_path}training_set.npy', allow_pickle=True).item()
    test_set = np.load(f'{data_path}test_set.npy', allow_pickle=True).item()
    session_items = np.load(f'{data_path}session_items.npy', allow_pickle=True).item()
    session_bundles = np.load(f'{data_path}session_bundles_deduplication.npy', allow_pickle=True).item()
    
    print(f"✓ Train sessions: {len(train_set)}")
    print(f"✓ Test sessions: {len(test_set)}")
    print(f"✓ Total sessions: {len(session_items)}")
    print(f"✓ Sessions with bundles: {len(session_bundles)}")
    
    def process_sessions(session_dict, session_items, session_bundles):
        """处理会话数据"""
        processed = {}
        for session_id, items_str in session_dict.items():
            items = items_str.split('|')
            
            bundles = []
            if session_id in session_bundles:
                for bundle in session_bundles[session_id]:
                    item_ids = bundle[-1].split(',')
                    bundle_indices = []
                    session_item_ids = session_items.get(session_id, '').split(',')
                    for item_id in item_ids:
                        if item_id in session_item_ids:
                            bundle_indices.append(session_item_ids.index(item_id))
                    if bundle_indices:
                        bundles.append(bundle_indices)
            
            processed[session_id] = {
                'items': items,
                'bundles': bundles
            }
        
        return processed
    
    train_data = process_sessions(train_set, session_items, session_bundles)
    test_data = process_sessions(test_set, session_items, session_bundles)
    
    return {
        'train': train_data,
        'test': test_data,
        'session_items': session_items,
        'session_bundles': session_bundles
    }


def generate_predictions(model, test_data, session_items):
    """
    使用训练好的模型生成预测
    
    Returns:
        bundle_res: {session_id: {'bundle1': ['product1', 'product2'], ...}}
    """
    print(f"\n{'='*60}")
    print("Generating predictions...")
    print(f"{'='*60}")
    
    bundle_res = {}
    
    for session_id, session_data in tqdm(test_data.items(), desc="Generating"):
        try:
            bundles = model.generate_bundles(session_data['items'])
            if bundles:
                bundle_res[session_id] = bundles
        except Exception as e:
            print(f"Error in session {session_id}: {e}")
            continue
    
    print(f"✓ Generated bundles for {len(bundle_res)}/{len(test_data)} sessions")
    
    return bundle_res


# ============================================
# 主函数
# ============================================

def main():
    parser = argparse.ArgumentParser(description='T5 Bundle Generation with Integrated Evaluation')
    parser.add_argument('--dataset', type=str, default='electronic',
                        help='数据集名称 (例如: electronic, sports, toys)')
    parser.add_argument('--model_name', type=str, default='t5-base',
                        help='T5模型名称')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小（论文设置为4）')
    parser.add_argument('--learning_rate', type=float, default=0.00005,
                        help='学习率（论文最优设置为0.00005）')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'test', 'all'],
                        help='运行模式: train=仅训练, test=仅测试, all=训练+测试+评估')
    parser.add_argument('--result_dir', type=str, default='result',
                        help='结果保存目录')
    
    args = parser.parse_args()
    
    # 从dataset自动派生data_path和model_path
    args.data_path = f'data/{args.dataset}/'
    args.model_path = f't5_bundle_model_{args.dataset}.pt'
    
    # 创建结果目录
    result_dir = f'{args.result_dir}/{args.dataset}'
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, 't5_results.txt')
    
    print("\n" + "="*60)
    print("T5 Bundle Generation - Integrated Version")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Data path: {args.data_path}")
    print(f"Model path: {args.model_path}")
    print(f"Model: {args.model_name}")
    print(f"Mode: {args.mode}")
    print("="*60)
    
    # 准备数据
    data = prepare_data_from_npy(args.data_path)
    
    # 初始化tokenizer (使用本地路径)
    if args.model_name == 't5-base':
        local_model_path = '/root/.cache/modelscope/hub/models/zjx200715/T5-base'
        tokenizer = T5Tokenizer.from_pretrained(local_model_path)
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    
    # 训练模式
    if args.mode in ['train', 'all']:
        print(f"\n{'='*60}")
        print("TRAINING PHASE")
        print(f"{'='*60}")
        
        # 创建数据集
        train_dataset = BundleDataset(data['train'], tokenizer)
        test_dataset = BundleDataset(data['test'], tokenizer)
        
        # 初始化模型
        model = T5BundleGenerator(model_name=args.model_name)
        
        # 训练
        model.train(
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            save_path=args.model_path
        )
        
        print(f"\n✓ Training completed!")
    
    # 测试和评估模式
    if args.mode in ['test', 'all']:
        print(f"\n{'='*60}")
        print("TESTING & EVALUATION PHASE")
        print(f"{'='*60}")
        
        # 加载模型
        if args.mode == 'test':
            model = T5BundleGenerator(model_name=args.model_name)
            model.load_model(args.model_path)
        
        # 生成预测
        bundle_res = generate_predictions(model, data['test'], data['session_items'])
        
        # 处理结果（移除单个产品的bundles）
        print(f"\n{'='*60}")
        print("Processing results...")
        print(f"{'='*60}")
        format_res = process_results(bundle_res)
        print(f"✓ Valid predictions: {len(format_res)}/{len(bundle_res)} sessions")
        
        # 评估
        print(f"\n{'='*60}")
        print("EVALUATION")
        print(f"{'='*60}")
        
        if len(format_res) > 0:
            session_precision, session_recall, coverage = compute(
                data['session_items'], 
                data['session_bundles'], 
                format_res
            )
            
            # 打印结果
            print(f"\n{'='*60}")
            print("FINAL RESULTS")
            print(f"{'='*60}")
            print(f"Dataset: {args.dataset}")
            print(f"Precision: {session_precision:.4f}")
            print(f"Recall:    {session_recall:.4f}")
            print(f"Coverage:  {coverage:.4f}")
            print(f"{'='*60}")
            
            # 保存结果
            with open(result_file, 'w') as f:
                f.write(f"Dataset: {args.dataset}\n")
                f.write(f"Model: {args.model_name}\n")
                f.write(f"Precision: {session_precision:.4f}\n")
                f.write(f"Recall: {session_recall:.4f}\n")
                f.write(f"Coverage: {coverage:.4f}\n")
                f.write(f"Valid sessions: {len(format_res)}/{len(data['test'])}\n")
            
            print(f"\n✓ Results saved to {result_file}")
        else:
            print("✗ No valid predictions generated!")
            print("Please check the model and data.")


if __name__ == "__main__":
    main()