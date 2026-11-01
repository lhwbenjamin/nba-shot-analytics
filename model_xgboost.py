import xgboost as xgb
import shap
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


def run_xgb_experiment(data, feature_list, use_target_enc=False, exp_name="Exp", reg_alpha=0.0, reg_lambda=1.0):
    print(f"\n" + "="*60 + f"\n This is experiment: {exp_name} \n" + "="*60)
    
    split_idx = int(len(data) * 0.80)
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()
    
    target_col = 'SHOT_MADE_FLAG'
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    
    X_train = train_df[feature_list].copy()
    X_test = test_df[feature_list].copy()
    
    if use_target_enc:
        action_type_means = train_df.groupby('ACTION_TYPE')[target_col].mean()
        global_mean = train_df[target_col].mean()
        
        X_train['ACTION_TYPE_TARGET_ENC'] = train_df['ACTION_TYPE'].map(action_type_means).fillna(global_mean)
        X_test['ACTION_TYPE_TARGET_ENC'] = test_df['ACTION_TYPE'].map(action_type_means).fillna(global_mean)

        if 'ACTION_TYPE' in X_train.columns:
            X_train = X_train.drop(columns=['ACTION_TYPE'])
            X_test = X_test.drop(columns=['ACTION_TYPE'])
            
    for col in X_train.select_dtypes(include=['object']).columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')


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
    
    return {"Experiment": exp_name, "ACC": round(acc_score,4), "AUC": round(auc_score, 4), "LogLoss": round(logloss_score, 4)}


if __name__ == "__main__":
    from data_processing import run_process
    data = run_process('2014-15')
    
    test_features = ['SHOT_DIST', 'SHOT_CLOCK']
    test_res = run_xgb_experiment(
        data=data.head(1000), 
        feature_list=test_features,
        exp_name="Test"
    )
    print("Result：", pd.DataFrame(test_res))