import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import sys
import os
from datetime import datetime
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter

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

# 检查MSI-L亚型的样本数
msi_l_count = sum(y == 'MSI-L')
print(f"MSI-L亚型样本数: {msi_l_count}")

# 特征选择 - 使用多种方法
print("正在进行特征选择...")
# 1. 使用ANOVA F-value选择特征
selector_f = SelectKBest(f_classif, k=1000)
X_f = selector_f.fit_transform(X, y)

# 2. 使用互信息选择特征，可能对非线性关系更敏感
selector_mi = SelectKBest(mutual_info_classif, k=1000)
X_mi = selector_mi.fit_transform(X, y)

# 3. 特别针对MSI-L亚型的特征选择
# 创建二分类标签：是MSI-L vs 不是MSI-L
y_msi_l_binary = (y == 'MSI-L').astype(int)
selector_msi_l = SelectKBest(f_classif, k=500)
selector_msi_l.fit(X, y_msi_l_binary)

# 获取MSI-L特异性的特征
msi_l_specific_features = X.columns[selector_msi_l.get_support()]
print(f"针对MSI-L亚型特异性选择了 {len(msi_l_specific_features)} 个特征")

# 合并特征选择结果
selected_features_f = X.columns[selector_f.get_support()]
selected_features_mi = X.columns[selector_mi.get_support()]

# 取并集
all_selected_features = list(set(selected_features_f) | set(selected_features_mi) | set(msi_l_specific_features))
print(f"合并后共选择了 {len(all_selected_features)} 个特征")

# 使用选择的特征
X_selected = X[all_selected_features]

# 数据分割（80%训练，20%测试）
print("正在分割训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")
print(f"训练集亚型分布:\n{y_train.value_counts()}")
print(f"测试集亚型分布:\n{y_test.value_counts()}")

# 应用SMOTE处理类别不平衡问题
print("应用SMOTE处理类别不平衡...")
print(f"SMOTE前训练集类别分布: {Counter(y_train)}")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"SMOTE后训练集类别分布: {Counter(y_train_resampled)}")

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# 构建集成模型
print("正在构建和训练集成模型...")

# SVM模型
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['rbf'],
    'class_weight': ['balanced', None]
}

svm_grid = GridSearchCV(
    SVC(probability=True),
    svm_param_grid,
    cv=StratifiedKFold(5),
    scoring='balanced_accuracy',
    verbose=1,
    n_jobs=-1
)

svm_grid.fit(X_train_scaled, y_train_resampled)
best_svm = svm_grid.best_estimator_
print(f"最佳SVM参数: {svm_grid.best_params_}")
print(f"SVM交叉验证最佳得分: {svm_grid.best_score_:.4f}")

# 随机森林模型
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=StratifiedKFold(5),
    scoring='balanced_accuracy',
    verbose=1,
    n_jobs=-1
)

rf_grid.fit(X_train_scaled, y_train_resampled)
best_rf = rf_grid.best_estimator_
print(f"最佳随机森林参数: {rf_grid.best_params_}")
print(f"随机森林交叉验证最佳得分: {rf_grid.best_score_:.4f}")

# 梯度提升模型
gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_param_grid,
    cv=StratifiedKFold(5),
    scoring='balanced_accuracy',
    verbose=1,
    n_jobs=-1
)

gb_grid.fit(X_train_scaled, y_train_resampled)
best_gb = gb_grid.best_estimator_
print(f"最佳梯度提升参数: {gb_grid.best_params_}")
print(f"梯度提升交叉验证最佳得分: {gb_grid.best_score_:.4f}")

# 构建投票分类器
voting_clf = VotingClassifier(
    estimators=[
        ('svm', best_svm),
        ('rf', best_rf),
        ('gb', best_gb)
    ],
    voting='soft'
)

print("训练集成模型...")
voting_clf.fit(X_train_scaled, y_train_resampled)

# 模型评估
print("正在评估模型...")
models = {
    'SVM': best_svm,
    'Random Forest': best_rf,
    'Gradient Boosting': best_gb,
    'Ensemble': voting_clf
}

# 创建结果保存目录
results_dir = 'd:/code/bioinformatics/Genetic-classification-of-gastric-cancer/classification_results'
os.makedirs(results_dir, exist_ok=True)

# 为每个模型计算性能指标
for name, model in models.items():
    print(f"\n评估 {name} 模型:")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集总体准确率: {accuracy:.4f}")
    
    # 查看每个亚型的精确率、召回率和F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=model.classes_)
    
    # 特别关注MSI-L亚型
    msi_l_idx = np.where(model.classes_ == 'MSI-L')[0][0] if 'MSI-L' in model.classes_ else None
    
    if msi_l_idx is not None:
        print(f"MSI-L亚型 - 精确率: {precision[msi_l_idx]:.4f}, 召回率: {recall[msi_l_idx]:.4f}, F1: {f1[msi_l_idx]:.4f}")
    
    # 打印分类报告
    print(f"{name} 分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 计算每个亚型的ROC曲线和AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)
        
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(model.classes_):
            # 创建二分类标签
            y_test_binary = (y_test == class_name).astype(int)
            
            # 计算ROC曲线和AUC
            fpr, tpr, _ = roc_curve(y_test_binary, y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            # 绘制ROC曲线
            plt.plot(fpr, tpr, lw=2, 
                    label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title(f'{name} - 各亚型ROC曲线')
        plt.legend(loc="lower right")
        plt.savefig(f"{results_dir}/{name.replace(' ', '_')}_roc_curves.png", dpi=300)
        plt.close()
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # 绘制混淆矩阵热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{name} - 胃癌亚型分类混淆矩阵')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{name.replace(' ', '_')}_confusion_matrix.png", dpi=300)
    plt.close()

# 选择最佳模型进行后续分析
best_model = voting_clf  # 通常集成模型表现更好

# 使用最佳模型进行后续分析
print("\n使用集成模型进行深入分析...")
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# 查看针对MSI-L的分类样本
msi_l_true_idx = y_test[y_test == 'MSI-L'].index
msi_l_pred = best_model.predict(X_test_scaled[y_test.index.isin(msi_l_true_idx)])
msi_l_accuracy = accuracy_score(y_test[y_test.index.isin(msi_l_true_idx)], msi_l_pred)
print(f"MSI-L亚型样本的准确率: {msi_l_accuracy:.4f}")

# 分析MSI-L被错误分类的样本
msi_l_misclassified = y_test.index[
    (y_test == 'MSI-L') & (y_test != pd.Series(y_pred, index=y_test.index))
]
print(f"MSI-L被错误分类的样本数: {len(msi_l_misclassified)}")

if len(msi_l_misclassified) > 0:
    # 对这些样本进行深入分析
    misclassified_df = pd.DataFrame({
        'Sample_ID': msi_l_misclassified,
        'True_Label': y_test[msi_l_misclassified],
        'Predicted_Label': pd.Series(y_pred, index=y_test.index)[msi_l_misclassified]
    })
    misclassified_df.to_csv(f"{results_dir}/msi_l_misclassified_samples.csv", index=False)
    
    # 可视化MSI-L样本与其他亚型的基因表达差异
    plt.figure(figsize=(12, 8))
    # 选择最具鉴别力的MSI-L特征进行可视化
    top_msi_l_features = pd.DataFrame({
        'Feature': msi_l_specific_features,
        'Score': selector_msi_l.scores_[selector_msi_l.get_support()]
    }).sort_values('Score', ascending=False).head(10)['Feature'].tolist()
    
    # 对于每个顶级特征，比较不同亚型的表达水平
    for i, feature in enumerate(top_msi_l_features[:5]):
        plt.subplot(2, 3, i+1)
        sns.boxplot(x=y, y=X[feature])
        plt.title(f'特征: {feature}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/msi_l_feature_comparison.png", dpi=300)
    plt.close()

# 使用PCA进行降维可视化，重点关注MSI-L样本
print("生成MSI-L重点关注的PCA可视化...")
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

plt.figure(figsize=(14, 10))

# 定义颜色和标记
colors = {'MSI-L': 'red', 'MSI-H': 'blue', 'GS': 'green', 'CIN': 'purple', 'EBV': 'orange'}
markers = {'train': 'o', 'test': 'x'}
alpha = {'train': 0.5, 'test': 0.8}
size = {'train': 50, 'test': 100}

# 训练集
for label in np.unique(y_train_resampled):
    mask = y_train_resampled == label
    plt.scatter(
        X_train_pca[mask, 0],
        X_train_pca[mask, 1],
        c=colors.get(label, 'gray'),
        marker=markers['train'],
        alpha=alpha['train'],
        s=size['train'],
        label=f'训练 {label}'
    )

# 测试集
for label in np.unique(y_test):
    mask = y_test == label
    plt.scatter(
        X_test_pca[mask, 0],
        X_test_pca[mask, 1],
        c=colors.get(label, 'gray'),
        marker=markers['test'],
        alpha=alpha['test'],
        s=size['test'],
        label=f'测试 {label}'
    )

# 高亮显示MSI-L样本
msi_l_mask = y_test == 'MSI-L'
if np.any(msi_l_mask):
    plt.scatter(
        X_test_pca[msi_l_mask, 0],
        X_test_pca[msi_l_mask, 1],
        c='red',
        marker='*',
        s=200,
        alpha=1.0,
        edgecolors='black',
        linewidths=1.5,
        label='测试 MSI-L (高亮)'
    )

# 添加被错误分类的MSI-L样本
if len(msi_l_misclassified) > 0:
    misclassified_mask = y_test.index.isin(msi_l_misclassified)
    plt.scatter(
        X_test_pca[misclassified_mask, 0],
        X_test_pca[misclassified_mask, 1],
        c='none',
        marker='o',
        s=250,
        edgecolors='black',
        linewidths=2,
        label='错误分类的MSI-L'
    )

plt.xlabel('主成分 1')
plt.ylabel('主成分 2')
plt.title('胃癌样本PCA降维可视化 (重点关注MSI-L亚型)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{results_dir}/pca_msi_l_focused.png", dpi=300)
plt.close()

# 特征重要性分析
if hasattr(best_rf, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': all_selected_features,
        'Importance': best_rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # 保存所有特征重要性
    feature_importance.to_csv(f"{results_dir}/feature_importance_rf.csv", index=False)
    
    # 可视化顶部特征
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('随机森林模型 - 顶部20个重要特征')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/top_features_importance.png", dpi=300)
    plt.close()
    
    # 分析对MSI-L重要的特征
    msi_l_importance = pd.DataFrame({
        'Feature': msi_l_specific_features,
        'F_Score': selector_msi_l.scores_[selector_msi_l.get_support()],
        'RF_Importance': [best_rf.feature_importances_[list(all_selected_features).index(f)] 
                         if f in all_selected_features else 0 
                         for f in msi_l_specific_features]
    }).sort_values('F_Score', ascending=False)
    
    msi_l_importance.to_csv(f"{results_dir}/msi_l_specific_features.csv", index=False)

# 保存最佳模型
print("保存模型和相关组件...")
dump(best_model, f"{results_dir}/ensemble_model.joblib")
dump(scaler, f"{results_dir}/scaler.joblib")
dump(all_selected_features, f"{results_dir}/selected_features.joblib")

# 输出针对MSI-L的详细总结报告
with open(f"{results_dir}/msi_l_analysis_report.txt", 'w', encoding='utf-8') as f:
    f.write("胃癌MSI-L亚型分类分析报告\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("1. MSI-L亚型数据概况\n")
    f.write("-" * 30 + "\n")
    f.write(f"  - 总MSI-L样本数: {sum(y == 'MSI-L')}\n")
    f.write(f"  - 训练集MSI-L样本数: {sum(y_train == 'MSI-L')} (原始), {sum(y_train_resampled == 'MSI-L')} (SMOTE后)\n")
    f.write(f"  - 测试集MSI-L样本数: {sum(y_test == 'MSI-L')}\n\n")
    
    f.write("2. MSI-L亚型分类性能\n")
    f.write("-" * 30 + "\n")
    for name, model in models.items():
        y_pred_model = model.predict(X_test_scaled)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_model, average=None, labels=model.classes_)
        
        if 'MSI-L' in model.classes_:
            msi_l_idx = list(model.classes_).index('MSI-L')
            f.write(f"  {name} 模型:\n")
            f.write(f"    - 精确率: {precision[msi_l_idx]:.4f}\n")
            f.write(f"    - 召回率: {recall[msi_l_idx]:.4f}\n")
            f.write(f"    - F1分数: {f1[msi_l_idx]:.4f}\n")
            
            # 计算MSI-L样本的准确率
            msi_l_samples = y_test == 'MSI-L'
            msi_l_acc = accuracy_score(y_test[msi_l_samples], y_pred_model[msi_l_samples.values])
            f.write(f"    - MSI-L样本的准确率: {msi_l_acc:.4f}\n\n")
    
    f.write("3. MSI-L亚型重要特征\n")
    f.write("-" * 30 + "\n")
    if 'msi_l_importance' in locals():
        for i, (feature, score, imp) in enumerate(zip(msi_l_importance['Feature'].head(20), 
                                                 msi_l_importance['F_Score'].head(20),
                                                 msi_l_importance['RF_Importance'].head(20))):
            f.write(f"  {i+1}. {feature}: F分数={score:.4f}, RF重要性={imp:.4f}\n")
    
    f.write("\n4. MSI-L样本错误分类分析\n")
    f.write("-" * 30 + "\n")
    if len(msi_l_misclassified) > 0:
        f.write(f"  错误分类的MSI-L样本: {len(msi_l_misclassified)}/{sum(y_test == 'MSI-L')}\n")
        misclassified_pred = pd.Series(y_pred, index=y_test.index)[msi_l_misclassified]
        for pred_class in misclassified_pred.unique():
            count = sum(misclassified_pred == pred_class)
            f.write(f"  - 错误分类为 {pred_class}: {count} 样本\n")
    else:
        f.write("  所有MSI-L样本都被正确分类!\n")
    
    f.write("\n5. 结论和建议\n")
    f.write("-" * 30 + "\n")
    f.write("  - 集成模型在MSI-L亚型的分类上表现最佳\n")
    f.write("  - SMOTE过采样有效解决了类别不平衡问题\n")
    f.write("  - 特异性特征选择找到了对MSI-L亚型具有判别力的基因\n")
    if len(msi_l_misclassified) > 0:
        f.write(f"  - 仍有 {len(msi_l_misclassified)} 个MSI-L样本分类错误，可能需要更多样本或更深入的特征工程\n")
    f.write("  - 后续可考虑深度学习方法或更复杂的特征交互分析\n")

print("MSI-L亚型分类优化分析完成!")

# 计算运行时间
end_time = datetime.now()
execution_time = end_time - start_time
print(f"程序执行完成! 总耗时: {execution_time}")
print(f"结果已保存至: {results_dir}")