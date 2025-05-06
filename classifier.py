import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import sys
import os
from datetime import datetime

# 避免中文乱码
sys.stdout.reconfigure(encoding='utf-8')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

print("开始运行胃癌亚型分类程序...")
start_time = datetime.now()

# 数据加载
print("正在加载数据...")
data_path = 'd:/code/bioinformatics/Genetic-classification-of-gastric-cancer/merged_multi_omics_data.csv'
clinical_path = 'd:/code/bioinformatics/Genetic-classification-of-gastric-cancer/preprocessed_data/clinical_filtered.csv'

try:
    # 加载表达数据
    expression_data = pd.read_csv(data_path, index_col=0)
    # 加载临床数据
    clinical_data = pd.read_csv(clinical_path)
    
    print(f"表达数据维度: {expression_data.shape}")
    print(f"临床数据维度: {clinical_data.shape}")
except Exception as e:
    print(f"数据加载失败: {str(e)}")
    sys.exit(1)

# 数据预处理
print("正在进行数据预处理...")

# 将表达数据转置，使每行代表一个样本，每列代表一个基因
X = expression_data.transpose()

# 从临床数据中提取样本ID和亚型
y = clinical_data.set_index('sampleID')['CDE_ID_3226963']

# 确保X和y的样本一致
common_samples = list(set(X.index) & set(y.index))
print(f"共有 {len(common_samples)} 个样本同时存在于表达数据和临床数据中")

X = X.loc[common_samples]
y = y.loc[common_samples]

print(f"亚型分布:\n{y.value_counts()}")

# 特征选择
print("正在进行特征选择...")
# 使用ANOVA F-value选择前1000个最相关的特征
selector = SelectKBest(f_classif, k=1000)
X_new = selector.fit_transform(X, y)

# 提取选中的特征名称
selected_features = X.columns[selector.get_support()]
print(f"选择了 {len(selected_features)} 个特征")

# 数据分割（80%训练，20%测试）
print("正在分割训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X_new, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建和训练SVM模型
print("正在训练SVM模型...")
# 使用网格搜索找到最佳参数
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(
    SVC(probability=True),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

# 获取最佳模型
best_model = grid_search.best_estimator_
print(f"最佳SVM参数: {grid_search.best_params_}")
print(f"交叉验证最佳得分: {grid_search.best_score_:.4f}")

# 模型评估
print("正在评估模型...")
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")

# 打印分类报告
class_report = classification_report(y_test, y_pred)
print("分类报告:")
print(class_report)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 创建结果保存目录
results_dir = 'd:/code/bioinformatics/Genetic-classification-of-gastric-cancer/classification_results'
os.makedirs(results_dir, exist_ok=True)

# 可视化结果
print("正在生成可视化结果...")

# 1. 混淆矩阵热图
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=best_model.classes_,
            yticklabels=best_model.classes_)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('胃癌亚型分类混淆矩阵')
plt.tight_layout()
plt.savefig(f"{results_dir}/confusion_matrix.png", dpi=300)

# 2. 使用PCA进行降维可视化
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

plt.figure(figsize=(12, 10))
# 绘制训练集
for i, label in enumerate(np.unique(y_train)):
    plt.scatter(
        X_train_pca[y_train == label, 0],
        X_train_pca[y_train == label, 1],
        alpha=0.5,
        label=f'训练 {label}'
    )
# 绘制测试集
for i, label in enumerate(np.unique(y_test)):
    plt.scatter(
        X_test_pca[y_test == label, 0],
        X_test_pca[y_test == label, 1],
        marker='x',
        alpha=0.7,
        s=100,
        label=f'测试 {label}'
    )

plt.xlabel('主成分 1')
plt.ylabel('主成分 2')
plt.title('胃癌样本基因表达PCA降维可视化')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{results_dir}/pca_visualization.png", dpi=300)

# 3. 亚型分布柱状图
plt.figure(figsize=(10, 6))
y.value_counts().plot(kind='bar', color='skyblue')
plt.xlabel('胃癌亚型')
plt.ylabel('样本数量')
plt.title('胃癌亚型分布')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{results_dir}/subtype_distribution.png", dpi=300)

# 4. 准确率条形图
plt.figure(figsize=(10, 6))
class_accuracy = {}
for label in np.unique(y_test):
    mask = y_test == label
    class_accuracy[label] = accuracy_score(y_test[mask], y_pred[mask])

plt.bar(class_accuracy.keys(), class_accuracy.values(), color='lightgreen')
plt.axhline(y=accuracy, color='r', linestyle='-', label=f'总准确率: {accuracy:.4f}')
plt.xlabel('胃癌亚型')
plt.ylabel('准确率')
plt.title('不同亚型的分类准确率')
plt.ylim(0, 1.1)
for i, (label, acc) in enumerate(class_accuracy.items()):
    plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center')
plt.legend()
plt.tight_layout()
plt.savefig(f"{results_dir}/accuracy_by_subtype.png", dpi=300)

# 5. 保存分类结果到CSV
results_df = pd.DataFrame({
    'Sample_ID': X_test.index if hasattr(X_test, 'index') else y_test.index,
    'True_Subtype': y_test,
    'Predicted_Subtype': y_pred
})
results_df.to_csv(f"{results_dir}/classification_results.csv", index=False)

# 保存重要特征
selected_indices = selector.get_support()
selected_features_list = X.columns[selected_indices].tolist()
feature_scores = selector.scores_[selected_indices]

feature_importance_df = pd.DataFrame({
    'Feature': selected_features_list,
    'Score': feature_scores
}).sort_values('Score', ascending=False)
feature_importance_df.to_csv(f"{results_dir}/important_features.csv", index=False)

# 保存模型
from joblib import dump
dump(best_model, f"{results_dir}/svm_model.joblib")
dump(scaler, f"{results_dir}/scaler.joblib")
dump(selector, f"{results_dir}/feature_selector.joblib")

# 计算运行时间
end_time = datetime.now()
execution_time = end_time - start_time
print(f"程序执行完成! 总耗时: {execution_time}")
print(f"结果已保存至: {results_dir}")

plt.close('all')  # 关闭所有图形

# 进行预测概率分析和可视化
probabilities = best_model.predict_proba(X_test_scaled)

# 可视化预测概率
plt.figure(figsize=(12, 8))
for i, label in enumerate(best_model.classes_):
    plt.subplot(1, len(best_model.classes_), i+1)
    plt.hist(probabilities[:, i], bins=20, alpha=0.7)
    plt.title(f'亚型 {label} 的预测概率分布')
    plt.xlabel('预测概率')
    plt.ylabel('样本数量')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{results_dir}/prediction_probabilities.png", dpi=300)
plt.close()

# 输出总结报告
with open(f"{results_dir}/summary_report.txt", 'w', encoding='utf-8') as f:
    f.write("胃癌亚型SVM分类模型总结报告\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"数据集信息:\n")
    f.write(f"  - 总样本数: {len(common_samples)}\n")
    f.write(f"  - 特征数量: {X.shape[1]}\n")
    f.write(f"  - 选取特征: {len(selected_features)}\n\n")
    
    f.write(f"亚型分布:\n")
    for subtype, count in y.value_counts().items():
        f.write(f"  - {subtype}: {count} 样本 ({count/len(y)*100:.1f}%)\n")
    
    f.write("\n模型信息:\n")
    f.write(f"  - 模型类型: SVM\n")
    f.write(f"  - 最佳参数: {grid_search.best_params_}\n")
    f.write(f"  - 交叉验证准确率: {grid_search.best_score_:.4f}\n\n")
    
    f.write("性能评估:\n")
    f.write(f"  - 测试集准确率: {accuracy:.4f}\n\n")
    
    f.write("分类报告:\n")
    f.write(class_report)
    
    f.write("\n各亚型准确率:\n")
    for label, acc in class_accuracy.items():
        f.write(f"  - {label}: {acc:.4f}\n")
    
    f.write("\n注: 详细结果和可视化图表可在classification_results目录中找到。\n")

print("完整分析和可视化已完成!")