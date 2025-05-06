import pandas as pd
import os
import sys

# 添加UTF-8编码设置，避免中文乱码
sys.stdout.reconfigure(encoding='utf-8')

def load_and_process_data(filepath, file_type):
    """
    加载数据文件并进行初步处理
    
    Args:
        filepath: 文件路径
        file_type: 文件类型（用于区分处理逻辑）
    
    Returns:
        处理后的DataFrame
    """
    print(f"正在加载 {os.path.basename(filepath)}...")
    
    # 读取数据，使用第一列作为索引
    df = pd.read_csv(filepath, index_col=0)
    
    # 根据不同文件类型添加前缀到基因/特征名
    prefix = file_type + "_"
    df.index = [prefix + str(gene) for gene in df.index]
    
    return df

# 预处理数据目录
preprocessed_dir = 'd:/code/bioinformatics/Genetic-classification-of-gastric-cancer/preprocessed_data'

# 待合并的文件及其类型
files_to_merge = {
    'cnv_filtered.csv': 'cnv',
    'methylation_filtered.csv': 'meth',
    'mirna_filtered.csv': 'mirna',
    'mutation_filtered.csv': 'mut',
    'protein_filtered.csv': 'prot',
    'rna_filtered.csv': 'rna'
}

# 存储各文件的DataFrame
dfs = []

# 加载所有文件
for filename, file_type in files_to_merge.items():
    filepath = os.path.join(preprocessed_dir, filename)
    if os.path.exists(filepath):
        df = load_and_process_data(filepath, file_type)
        dfs.append(df)
        print(f"  - 形状: {df.shape}")
    else:
        print(f"警告: 找不到文件 {filename}")

if not dfs:
    print("没有找到任何有效文件，无法合并")
    sys.exit(1)

# 合并所有数据框
print("正在合并数据...")
merged_df = pd.concat(dfs, axis=0)

# 检查缺失值情况
missing_values = merged_df.isna().sum().sum()
print(f"合并后的数据维度: {merged_df.shape}")
print(f"合并后的数据中有 {missing_values} 个缺失值")

# 保存合并后的数据
output_path = 'd:/code/bioinformatics/Genetic-classification-of-gastric-cancer/merged_multi_omics_data.csv'
print(f"正在保存合并数据到 {output_path}...")
merged_df.to_csv(output_path)

print("数据合并完成！")

# 输出每个组学数据的特征数量
print("\n各组学数据的特征数量:")
for prefix in [f"{file_type}_" for file_type in files_to_merge.values()]:
    feature_count = sum(merged_df.index.str.startswith(prefix))
    print(f"  - {prefix[:-1]}: {feature_count} 特征")

# 输出样本数量
print(f"\n样本总数: {merged_df.shape[1]}")