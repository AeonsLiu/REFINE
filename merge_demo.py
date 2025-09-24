import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from openai import OpenAI

# 递归函数：将决策树转换为 if-else 规则
def tree_to_code(tree, feature_names, node_index=0, depth=0):
    indent = "    " * depth
    if tree.children_left[node_index] == -1 and tree.children_right[node_index] == -1:
        return f"{indent}return {tree.value[node_index].argmax()} ({dic[tree.value[node_index].argmax()]})"

    feature = feature_names[tree.feature[node_index]]
    threshold = tree.threshold[node_index]
    left_subtree = tree_to_code(tree, feature_names, tree.children_left[node_index], depth + 1)
    right_subtree = tree_to_code(tree, feature_names, tree.children_right[node_index], depth + 1)

    return f"""{indent}if {feature} <= {threshold:.2f}:
{left_subtree}
{indent}else:
{right_subtree}"""

df0 = pd.read_csv("./data/sampled_30.csv")

df_temp = pd.concat([df0], ignore_index=True)
df_test = pd.concat([df0], ignore_index=True)

label_columns = ["Gender", "Ethnicity", "EducationLevel", "Smoking", "FamilyHistoryAlzheimers", "CardiovascularDisease", 
                 "Diabetes", "Depression", "HeadInjury", "Hypertension", "MemoryComplaints", "BehavioralProblems", "Confusion", 
                 "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness", "Diagnosis"]


if 'Diagnosis' in df_temp.columns:
    cols = ['Diagnosis'] + [col for col in df_temp.columns if col != 'Diagnosis']
    df_temp = df_temp[cols]
    df_test = df_test[cols]

label_encoders = {}
for col in label_columns:
    # 转换为字符串
    df_temp[col] = df_temp[col].astype(str)
    df_test[col] = df_test[col].astype(str)

    # 训练 LabelEncoder
    le = LabelEncoder()
    df0[col] = le.fit_transform(df0[col])  # 训练数据 fit_transform
    label_encoders[col] = le  # 存储编码器

    # 获取类别映射表
    mapping = {cls: idx for idx, cls in enumerate(le.classes_)}

    # 用 `replace()` 高效转换
    df_temp[col] = df_temp[col].map(mapping).fillna(-1).astype(int)
    df_test[col] = df_test[col].map(mapping).fillna(-1).astype(int)

# 目标变量
target = "Diagnosis"
X_train = df_temp.drop(columns=[target])
y_train = df_temp[target]

X_test = df_test.drop(columns=[target])
y_test = df_test[target]

# 设置参数搜索范围
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 2, 3, 4, 5],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None, 0.5],
    'bootstrap': [True, False],
    'max_samples': [0.3, 0.5, 0.7, 0.9]  # 只在 bootstrap=True 时有效
}

# 执行网格搜索
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=200,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)

# 使用最优模型
rf = random_search.best_estimator_
print("\n整个随机森林模型的最佳参数:")
print(random_search.best_params_)

print(f"\n最佳模型在交叉验证中的平均得分（accuracy）: {random_search.best_score_:.4f}")

# 计算每一棵树的投票正确率（准确率）
estimator_accuracies = []
for idx, estimator in enumerate(rf.estimators_):
    y_pred_estimator = estimator.predict(X_test.to_numpy())
    score = f1_score(y_test, y_pred_estimator, average='macro')
    estimator_accuracies.append((idx, score))

# 根据准确率对 estimator 降序排序
estimator_accuracies_sorted = sorted(
    estimator_accuracies, key=lambda x: x[1], reverse=True)

# 打印所有树的准确率
print("\n所有树在验证集上的准确率:")
for idx, acc in estimator_accuracies:
    print(f"Estimator {idx}: f1 macro = {acc:.4f}")

# 定义 k 值，选择前 k 个最佳的树
k = 3  # 可根据需要调整
top_k_estimators = estimator_accuracies_sorted[:k]

# 返回选出的最佳树的索引列表
top_k_estimators_indices = [idx for idx, acc in top_k_estimators]

# 输出
print("最佳的k个estimator的索引、F1分数及其参数:")
for idx, acc in top_k_estimators:
    estimator = rf.estimators_[idx]
    params = estimator.get_params()
    print(f"\nEstimator {idx}: f1_macro = {acc:.4f}")
    print("参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")

# 打印特征重要性
importances = rf.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

print("\n随机森林特征重要性:")
print(feature_importance_df)

# 定义标签对应关系
dic = {
    0: "Diagnosis = 0",
    1: "Diagnosis = 1",
}


rules = []
# 打印 top-k 棵树的 if-else 结构
for idx in top_k_estimators_indices:
    estimator = rf.estimators_[idx]
    print(f"\nEstimator {idx} 的 if-else 结构:")
    # tree_rules = f"def decision_tree_{idx}_rule({', '.join(X.columns)}):\n" + tree_to_code(estimator.tree_, X.columns)
    tree_rule = tree_to_code(estimator.tree_, X_train.columns)
    print(tree_rule)
    rules.append(tree_rule)
    connector_template = "If-else Structure Sample {}"

    # 在每个字符串前加上对应的连接词
    result = "\n\n".join([f"{connector_template.format(i+1)}:\n{rules[i]}" for i in range(len(rules))])

    print("------")

print(result)


prompt = """
You are an intelligent assistant responsible for transforming nested if-else classification logic into generalized, \
structured rules for data generation. These rules involve a target label called Diagnosis (values 0 or 1) determined by thresholds \
across multiple input features such as MMSE, FunctionalAssessment, ADL, SystolicBP, SleepQuality, etc.

Your task consists of three main steps:

### Step 1: Parse If-Else Logic into Flat Rule Format

Given one or more nested if-else decision structures (represented by placeholders), flatten them into a set of human-readable rule statements. Each statement must describe the conditions that lead to a specific Diagnosis assignment.

#### Example:
Input:
if MMSE <= 18.77:
    if PhysicalActivity <= 3.41:
        if Smoking <= 0.50:
            return 1 (Diagnosis = 1)
        else:
            return 0 (Diagnosis = 0)
    else:
        if ADL <= 3.18:
            return 1 (Diagnosis = 1)
        else:
            return 0 (Diagnosis = 0)
else:
    return 0 (Diagnosis = 0)

Output:
- If Diagnosis = 0, then MMSE > 18.77 and Smoking = 1 and ADL > 3.18
- If Diagnosis = 1, then MMSE <= 18.77 and Smoking = 0 and ADL <= 3.18

Repeat this process for all provided if-else blocks. The output of this step should be a set of conditions grouped by Diagnosis.

#### Input:
{if_else_logic}

### Step 2: Merge Similar Diagnosis Rules Across Samples

Now that each if-else structure has been flattened into rules, your goal is to merge rules belonging to the same Diagnosis across multiple samples.

#### Merging Logic:
- Identify all rules corresponding to a given Diagnosis.
- For each feature (e.g., MMSE, PhysicalActivity), extract the relevant conditions across all rules.
- Instead of strictly choosing the second smallest or second largest value, focus on:
  1. **Maintaining differentiation** between adjacent Diagnosis, ensuring that boundaries between different Diagnosis are clearly defined and preserved.
  2. **Maximizing generalization**, meaning that overlapping condition ranges should be merged in such a way that it still respects the distinctions while \
covering the broadest possible range.

### Step 3: Output Final Rule Set

Produce a **final set of generalized rules**, but **only one rule per Diagnosis** (0 or 1). The rules should strictly incorporate only mention in the **Key Features with Importance**, **strictly** following this format:

- If Diagnosis = 0, then [CONDITION1] and [CONDITION2] ...
- If Diagnosis = 1, then [CONDITION1] and [CONDITION2] ...

Where each condition is a feature threshold based on the merging strategy above, using only the **Key Features with Importance**. Each rule should focus on these features, and you should only have one rule for each Diagnosis.

Make sure your output:
- Uses readable formatting
- Maintains clear zones between different Diagnosises, - Maintains clear **while allowing overlap where necessary**
- Uses inclusive and exclusive operators appropriately (i.e., >, >=, <, <=)
- Properly incorporates feature importance to prioritize significant features
- Avoids overly specific or restrictive rules

"""
prompt = prompt.format(if_else_logic=result)

print(prompt)

client = OpenAI(
    api_key = "your_api_key",
    base_url = "your_base_url"
)

response = client.chat.completions.create(
    model="gpt-4o-0806",messages=[
    {"role": "system", "content": "You are a tabular synthetic data generation model."},
    {"role": "user", "content": prompt}
]
)

# print(response)
# 获取生成的文本数据
generated_data = response.choices[0].message.content

print(generated_data)