import pandas as pd
import os
import numpy as np

# 定义文件路径
data_dir = 'd:/code/bioinformatics/Genetic-classification-of-gastric-cancer/data'
output_dir = 'd:/code/bioinformatics/Genetic-classification-of-gastric-cancer/preprocessed_data'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取临床数据
clinical_path = os.path.join(data_dir, 'clinical.csv')
clinical_data = pd.read_csv(clinical_path)
clinical_samples = set(clinical_data['sampleID'])

# 读取其他数据文件并提取样本ID
file_paths = {
    'cnv': os.path.join(data_dir, 'cnv_filtered.csv'),
    'methylation': os.path.join(data_dir, 'methylation_filtered.csv'),
    'mirna': os.path.join(data_dir, 'mirna_filtered.csv'),
    'mutation': os.path.join(data_dir, 'mutation_filtered.csv'),
    'protein': os.path.join(data_dir, 'protein_filtered.csv'),
    'rna': os.path.join(data_dir, 'rna_filtered.csv')
}

# 获取每个文件的样本ID集合
sample_sets = {}
dfs = {}

for name, path in file_paths.items():
    try:
        df = pd.read_csv(path, index_col=0)
        # 保存DataFrame以供后续使用
        dfs[name] = df
        # 提取样本ID (列名)
        samples = set(df.columns)
        sample_sets[name] = samples
        print(f"读取 {name} 文件, 样本数: {len(samples)}")
    except Exception as e:
        print(f"读取 {name} 文件时出错: {e}")
        sample_sets[name] = set()

# 找出所有文件共有的样本ID
common_samples = clinical_samples.copy()
for name, samples in sample_sets.items():
    common_samples = common_samples.intersection(samples)

print(f"所有文件共有的样本数: {len(common_samples)}")

# 如果没有共同样本，则退出
if len(common_samples) == 0:
    print("没有找到共同的样本ID，请检查数据文件。")
    exit()

# 转换为有序列表，以确保输出文件中的样本顺序一致
common_samples = sorted(list(common_samples))

# 为每个文件筛选共同样本并保存
# 保存临床数据
clinical_filtered = clinical_data[clinical_data['sampleID'].isin(common_samples)]
clinical_filtered.to_csv(os.path.join(output_dir, 'clinical_filtered.csv'), index=False)
print(f"保存临床数据，形状: {clinical_filtered.shape}")

# 保存其他数据文件
for name, df in dfs.items():
    try:
        # 选择共同样本的列
        common_cols = [col for col in df.columns if col in common_samples]
        filtered_df = df[common_cols]
        output_path = os.path.join(output_dir, f'{name}_filtered.csv')
        filtered_df.to_csv(output_path)
        print(f"保存 {name} 文件，形状: {filtered_df.shape}")
    except Exception as e:
        print(f"处理 {name} 文件时出错: {e}")

print("所有数据处理完成！")
