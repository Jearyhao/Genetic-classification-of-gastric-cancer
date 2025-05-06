import pandas as pd
import os

# 定义预处理数据目录
preprocessed_dir = 'd:/code/bioinformatics/Genetic-classification-of-gastric-cancer/preprocessed_data'

# 获取目录中的所有CSV文件
csv_files = [f for f in os.listdir(preprocessed_dir) if f.endswith('.csv')]

print("预处理数据文件的维度信息：")
print("-" * 50)
print("{:<25} {:<15} {:<15}".format("文件名", "行数", "列数"))
print("-" * 50)

# 读取每个文件并输出其维度
for file_name in sorted(csv_files):
    file_path = os.path.join(preprocessed_dir, file_name)
    try:
        # 临床数据有标题行，其他数据第一列是特征名
        if file_name == 'clinical_filtered.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path, index_col=0)
        
        rows, cols = df.shape
        print("{:<25} {:<15} {:<15}".format(file_name, rows, cols))
    except Exception as e:
        print("{:<25} 读取出错: {}".format(file_name, str(e)))

print("-" * 50)