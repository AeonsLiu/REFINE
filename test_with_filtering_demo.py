import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filtering import grid_search_filter

df1 = pd.read_csv('./data/sampled_30.csv')

df2 = pd.read_csv('./data/synthetic_data.csv')


df_test = pd.read_csv('./data/test.csv')

df = pd.concat([df2], ignore_index=True)

df = df.drop("DoctorInCharge", axis=1)
df_test = df_test.drop("DoctorInCharge", axis=1)

print("Original length: ", len(df))

filter = True        

print("filter:", filter)


label_columns = ["Gender", "Ethnicity", "EducationLevel", "Smoking", "FamilyHistoryAlzheimers", "CardiovascularDisease", 
            "Diabetes", "Depression", "HeadInjury", "Hypertension", "MemoryComplaints", "BehavioralProblems", "Confusion", 
            "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness"]

if 'Diagnosis' in df_test.columns:
    cols = ['Diagnosis'] + [col for col in df_test.columns if col != 'Diagnosis']
    df = df[cols]
    df1 = df1[cols]
    df_test = df_test[cols]

categorical_features = label_columns
all_features = df_test.drop('Diagnosis',axis=1).columns.tolist()

numerical_features = [feat for feat in all_features if feat not in categorical_features]

label_encoders = {}
for col in label_columns:
    df[col] = df[col].astype(str)
    df1[col] = df1[col].astype(str)
    df_test[col] = df_test[col].astype(str)

    le = LabelEncoder()
    df_test[col] = le.fit_transform(df_test[col])
    label_encoders[col] = le
    df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    df1[col] = df1[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)


X_train_orig = df1.drop('Diagnosis', axis=1)

y_train_orig = df1['Diagnosis']
X_train = df.drop('Diagnosis', axis=1)
y_train = df['Diagnosis']  

X_test = df_test.drop('Diagnosis', axis=1)
y_test = df_test['Diagnosis']

result = grid_search_filter(
    X_small=X_train_orig, y_small=y_train_orig,
    X_large=X_train, y_large=y_train,
    X_test=X_train_orig, y_test=y_train_orig,
    numerical_features = numerical_features,
    categorical_features = categorical_features,
    block_sizes=[20, 25, 30, 35, 40, 45, 50, 55, 60], 
    n_iterations=10,
    curate = True
)

selected = result["idxs"]

print(len(selected))
df = df.iloc[selected]
df.to_csv('./filtered.csv', index=False)

X_train = df.drop('Diagnosis', axis=1)
y_train = df['Diagnosis']  

print("train length:", len(X_train))


xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=2)


param_grid = {
    'max_depth': [2, 3, 4, 5],
    'learning_rate': [0.1, 0.01],
    'n_estimators': [50, 100, 150, 200]
}


grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='f1_macro')

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

f1_macro = f1_score(y_test, y_pred, average='macro')
print(f'F1 on test set: {f1_macro * 100:.2f}% -- Accuracy on test set: {accuracy * 100:.2f}%')