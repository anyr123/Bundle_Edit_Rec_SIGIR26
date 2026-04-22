# Bundle Add Task - 模型运行指南

## 概述
本目录包含了用于bundle add任务的5个深度学习模型的实现，以及统一的运行脚本。

## 模型列表
1. **TSF** - Transformer-based Sequential Framework
2. **Mean-VAE** - Variational Autoencoder with Mean Pooling
3. **ItemKNN** - Item-based K-Nearest Neighbors
4. **Concat-VAE** - Variational Autoencoder with Concatenation
5. **BPRMF** - Bayesian Personalized Ranking Matrix Factorization

## 快速使用

### 运行单个模型
```bash
# 运行TSF模型，使用clothing数据集
python run_add.py -m tsf -d clothing

# 运行Mean-VAE模型，使用food数据集  
python run_add.py -m mean_vae -d food

# 运行所有模型可用的参数
python run_add.py -m {tsf|mean_vae|itemknn|concat_vae|bprmf} -d {food|clothing|electronic}
```

### 批量运行所有模型
```bash
# 运行所有模型，使用clothing数据集（推荐用法）
python run_all_add.py -d clothing

# 运行指定的几个模型
python run_all_add.py -d clothing -m tsf mean_vae bprmf

# 跳过已有结果文件的模型（快速生成报告）
python run_all_add.py -d clothing --skip-existing
```

## 参数说明

### run_add.py 参数
- `-m, --model`: 选择要运行的模型 (必需)
  - `tsf`: TSF模型
  - `mean_vae`: Mean-VAE模型  
  - `itemknn`: ItemKNN模型
  - `concat_vae`: Concat-VAE模型
  - `bprmf`: BPRMF模型
- `-d, --dataset`: 选择数据集 (默认: clothing)
  - `food`: 食品数据集
  - `clothing`: 服装数据集
  - `electronic`: 电子产品数据集

### run_all_add.py 参数
- `-d, --dataset`: 选择数据集 (必需)
- `-m, --models`: 选择要运行的模型 (默认: 所有模型)
  - 指定具体模型: `tsf mean_vae itemknn concat_vae bprmf`
- `--skip-existing`: 跳过已有结果文件的模型

## 输出结果

### 结果文件位置
所有结果都保存在 `../result/{dataset}/` 目录下：
- `tsf_add_{dataset}_results.txt`: TSF模型结果
- `mean_vae_add_{dataset}_results.txt`: Mean-VAE模型结果  
- `itemknn_add_{dataset}_results.txt`: ItemKNN模型结果
- `concat_vae_add_{dataset}_results.txt`: Concat-VAE模型结果
- `bprmf_add_{dataset}_results.txt`: BPRMF模型结果
- `add_comparison_report.txt`: 模型对比报告

### 评估指标
每个模型的结果文件包含：
- **Hit@1**: 命中率@1
- **Hit count**: 命中次数
- **Total bundles**: 总测试bundle数量
- **Success rate**: 成功处理的样本比例
- **Time**: 运行时间（分钟）

## 示例输出

### 单个模型结果
```
TSF Add Task Results - clothing
Hit@1: 0.2350
Hit count: 47
Total bundles: 200
Success rate: 195/200
Time: 12.34 minutes
```

### 批量运行对比报告
```
[REPORT] CLOTHING 数据集 - Add任务模型对比报告
================================================================================
模型         Hit@1    命中数    总数      成功率        耗时(分钟)
--------------------------------------------------------------------------------
TSF          0.2350   47       200      195/200      12.34
MEAN_VAE     0.2200   44       200      195/200      10.56
BPRMF        0.2100   42       200      195/200      8.45
CONCAT_VAE   0.2050   41       200      195/200      11.23
ITEMKNN      0.1950   39       200      195/200      15.67
================================================================================
[BEST] 最佳模型: TSF
[BEST] 最高Hit@1: 0.2350
```

## 数据要求

### 输入数据文件
- `../../dataset/{dataset}/bundle_item.csv`: Bundle-Item关系数据
- `../../dataset/{dataset}/user_item.csv`: User-Item交互数据（ItemKNN使用）
- `../../dataset/{dataset}/user_bundle.csv`: User-Bundle交互数据
- `../../testdata/{dataset}/add_test.txt`: 测试数据

### 数据格式
- `bundle_item.csv`: `bundle_id,item_id`
- `user_item.csv`: `user_id,item_id,timestamp`  
- `add_test.txt`: `bundle_id\tbundle_items\tcandidate_items`

## 技术细节

### 模型特点
1. **TSF**: 使用Transformer编码器处理bundle序列
2. **Mean-VAE**: 通过平均池化聚合item embeddings
3. **ItemKNN**: 基于item相似度的协同过滤
4. **Concat-VAE**: 通过连接item embeddings进行编码
5. **BPRMF**: 使用BPR损失的矩阵分解

### 训练设置
- 所有模型训练10个epoch
- 使用Adam优化器
- 批量大小128（部分模型可能不同）
- 自动GPU/CPU检测

## 故障排除

### 常见问题
1. **CUDA out of memory**: 减少批量大小或使用CPU
2. **文件路径错误**: 确保数据文件存在于正确位置
3. **依赖包缺失**: 安装所需的Python包

### 依赖包
```bash
pip install torch pandas numpy scikit-learn tqdm
```

注意：新版本的批量运行脚本需要pandas来生成对比报告。

## 性能建议
- 使用GPU可显著加速训练（如果可用）
- clothing数据集通常运行最快
- 可以使用`--skip-existing`选项快速生成已有结果的对比报告
- 批量运行会自动生成模型对比报告，便于分析最佳模型
