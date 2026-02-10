import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from config import FEATURE_COLS, TARGET_COLS_REG, TARGET_COL_CLS, VAR_MAP

def train_models(df):
    """
    Trains regression and classification models.
    Returns:
        reg_models: Dict of trained regression models
        clf_model: Trained classification model
        data_split: Tuple of (X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test)
        model_results: List of dicts with regression metrics
        cls_results: List of dicts with classification metrics
        conf_mx: Confusion matrix for the best classifier
        best_cls_name: Name of the best classifier
    """
    X = df[FEATURE_COLS]
    y_reg = df[TARGET_COLS_REG]
    y_cls = df[TARGET_COL_CLS]

    # Split Data
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42
    )

    print(f"Training Data: {X_train.shape}")
    print(f"Test Data:     {X_test.shape}")
    print("-" * 60)

    # --- Regression Models ---
    print("\n[PART 2.1] Model Selection & Training (K, L, M)")
    reg_candidates = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42)
    }

    best_models = {}
    model_results = []

    for target in TARGET_COLS_REG:
        print(f"\n  Evaluating models for Target {VAR_MAP[target]}...")
        best_rmse = float('inf')
        best_model_name = None
        best_model_obj = None

        for name, model in reg_candidates.items():
            model.fit(X_train, y_reg_train[target])
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_reg_test[target], y_pred)
            rmse = np.sqrt(mean_squared_error(y_reg_test[target], y_pred))
            mae = mean_absolute_error(y_reg_test[target], y_pred)
            
            model_results.append({
                'Target': target, 'Model': name, 'R2': r2, 'RMSE': rmse, 'MAE': mae
            })
            print(f"    > {name:18s}: R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = name
                best_model_obj = model
        
        print(f"  [WINNER] {best_model_name} selected for {target}.")
        best_models[target] = best_model_obj
        
        if hasattr(best_model_obj, 'feature_importances_'):
            importances = best_model_obj.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_feature = FEATURE_COLS[indices[0]]
            print(f"    Key Driver: {VAR_MAP[top_feature]} ({importances[indices[0]]:.1%})")

    # --- Classification Models ---
    print("-" * 60)
    print("\n[PART 2.2] Predicting Status N (Classification)")
    clf_candidates = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    best_cls_score = -1
    best_cls_name = None
    clf_model = None
    cls_results = []

    for name, model in clf_candidates.items():
        model.fit(X_train, y_cls_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_cls_test, y_pred)
        f1 = f1_score(y_cls_test, y_pred)
        cls_results.append({
            'Model': name, 'Accuracy': acc, 'F1 Score': f1,
            'Precision': precision_score(y_cls_test, y_pred),
            'Recall': recall_score(y_cls_test, y_pred)
        })
        print(f"  > {name:18s}: Acc={acc:.1%}, F1={f1:.3f}")
        
        if f1 > best_cls_score:
            best_cls_score = f1
            best_cls_name = name
            clf_model = model

    print(f"  [WINNER] {best_cls_name} selected for Status N.")
    y_cls_pred = clf_model.predict(X_test)
    conf_mx = confusion_matrix(y_cls_test, y_cls_pred)
    print(f"  > Confusion Matrix:\n{conf_mx}")

    data_split = (X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test)
    return best_models, clf_model, data_split, model_results, cls_results, conf_mx, best_cls_name
