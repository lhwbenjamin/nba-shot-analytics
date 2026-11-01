import xgboost as xgb
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from interpret.glassbox import ExplainableBoostingClassifier

def get_split_data(data, feature_list, train_ratio=0.80):
    split_idx = int(len(data)*train_ratio)
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()

    target_col = 'SHOT_MADE_FLAG'
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    
    X_train = train_df[feature_list].copy()
    X_test = test_df[feature_list].copy()
    
    if 'ACTION_TYPE' in X_train.columns:
        action_means = train_df.groupby('ACTION_TYPE')[target_col].mean()
        global_mean = train_df[target_col].mean()
        
        X_train['ACTION_TYPE_TARGET_ENC'] = train_df['ACTION_TYPE'].map(action_means).fillna(global_mean).astype(float)
        X_test['ACTION_TYPE_TARGET_ENC'] = test_df['ACTION_TYPE'].map(action_means).fillna(global_mean).astype(float)

    for col in X_train.select_dtypes(include=['object']).columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    split_dataset = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
    return split_dataset

def run_xgb_experiment(split_dataset, exp_name="Exp", reg_alpha=0.0, reg_lambda=1.0):
    print(f"\n" + "="*60 + f"\n This is xgb experiment: {exp_name} \n" + "="*60)
    
    X_train = split_dataset["X_train"].copy()
    X_test = split_dataset["X_test"].copy()
    y_train = split_dataset["y_train"]
    y_test = split_dataset["y_test"]

    if 'ACTION_TYPE' in X_train.columns and 'ACTION_TYPE_TARGET_ENC' in X_train.columns:
        X_train = X_train.drop(columns=['ACTION_TYPE'])
        X_test = X_test.drop(columns=['ACTION_TYPE'])

    model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        random_state=42, 
        enable_categorical=True, 
        n_jobs=-1, 
        eval_metric='logloss',
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc_score = accuracy_score(y_test, preds)
    auc_score = roc_auc_score(y_test, probs)
    logloss_score = log_loss(y_test, probs)
    print(f"Result | ACC: {acc_score:.4f} | AUC: {auc_score:.4f} | LogLoss: {logloss_score:.4f}")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    shap.plots.beeswarm(shap_values, max_display=30)
    shap.plots.bar(shap_values, max_display=35)
    
    return {"Experiment": exp_name, "Model": "XGB", "ACC": round(acc_score,4), "AUC": round(auc_score, 4), "LogLoss": round(logloss_score, 4)}

def run_ebm_experiment(split_dataset, exp_name="EBM_Exp"):
    print(f"\n" + "="*60 + f"\n This is ebm experiment: {exp_name} \n" + "="*60)
    
    X_train = split_dataset["X_train"].copy()
    X_test = split_dataset["X_test"].copy()
    y_train = split_dataset["y_train"]
    y_test = split_dataset["y_test"]

    if 'ACTION_TYPE' in X_train.columns and 'ACTION_TYPE_TARGET_ENC' in X_train.columns:
        X_train = X_train.drop(columns=['ACTION_TYPE_TARGET_ENC'])
        X_test = X_test.drop(columns=['ACTION_TYPE_TARGET_ENC'])
    
    model = ExplainableBoostingClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc_score = accuracy_score(y_test, preds)
    auc_score = roc_auc_score(y_test, probs)
    logloss_score = log_loss(y_test, probs)
    print(f"Result | ACC: {acc_score:.4f} | AUC: {auc_score:.4f} | LogLoss: {logloss_score:.4f}")

    
    ebm_global = model.explain_global(name=f'{exp_name} Global Explanation')
    
    data_dict = ebm_global.data()
    names = data_dict['names']
    scores = data_dict['scores']
    
    sorted_idx = np.argsort(scores)[::-1][:35]
    plt.figure(figsize=(10, 8))
    plt.barh([names[i] for i in sorted_idx][::-1], [scores[i] for i in sorted_idx][::-1], color='teal')
    plt.xlabel('Mean Absolute Score (Importance)')
    plt.title(f'{exp_name} - EBM Feature Importance')
    plt.tight_layout()
    plt.show()
    
    return {"Experiment": exp_name, "Model": "EBM", "ACC": round(acc_score,4), "AUC": round(auc_score, 4), "LogLoss": round(logloss_score, 4)}

if __name__ == "__main__":
    from data_processing import run_process
    data = run_process('2014-15')

    test_features = ['SHOT_DIST', 'SHOT_CLOCK']
    dataset_package = get_split_data(data.head(1000), test_features)

    flag = True
    if flag:
        test_res = run_xgb_experiment(dataset_package, exp_name="XGB_Test")
        print("Result：", pd.DataFrame(test_res))
    else:
        test_res = run_ebm_experiment(dataset_package, exp_name="EBM_Test")
        print("Result：", pd.DataFrame(test_res))