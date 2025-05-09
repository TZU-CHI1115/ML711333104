# HW3_common_utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import  GridSearchCV, StratifiedShuffleSplit
from datetime import datetime

def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values  # 所有特徵欄位
    y = df.iloc[:, -1].astype(int).values  # 最後一欄是標籤
    return X, y

def split_and_scale(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

def apply_pca(X_train, X_test, variance_ratio=0.8):
    pca = PCA(n_components=variance_ratio)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

def run_logistic_regression_cv(
    X_train, X_test, y_train, y_test,
    solver='lbfgs',
    Cs=np.logspace(-5, 5, 20),
    cv=5,
    tol=1e-6,
    max_iter=int(1e6),
    verbose=0,
    print_report=True
):
    """
    執行 Logistic Regression with Cross Validation，找出最佳 C 並評估表現。

    Parameters:
    - solver: 解法，例如 'lbfgs', 'liblinear', 'saga' 等
    - Cs: 候選 C 值（懲罰強度的倒數）
    - cv: 交叉驗證摺數
    - tol: 收斂容忍度
    - max_iter: 最大迭代次數
    - verbose: 是否顯示訓練過程細節
    - print_report: 是否印出報表（分類結果）

    Returns:
    dict 包含模型、最佳 C、訓練測試準確率、預測值
    """

    clf = LogisticRegressionCV(
        solver=solver,
        Cs=Cs,
        cv=cv,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    train_score = accuracy_score(y_train, clf.predict(X_train))
    test_score = accuracy_score(y_test, y_pred)

    if print_report:
        print(f"Logistic Regression with CV (solver = {solver})")
        print(f"Best C = {clf.C_}")
        print(f"Training Accuracy: {train_score:.2%}")
        print(f"Testing Accuracy : {test_score:.2%}")
        #print(classification_report(y_test, y_pred))

    return {
        'model': clf,
        'best_C': clf.C_,
        'train_acc': train_score,
        'test_acc': test_score,
        'y_pred': y_pred
    }

def run_logistic_gridcv(
    X_train, y_train,
    param_grid=None,
    tol=1e-6,
    max_iter=int(1e6),
    n_splits=5,
    test_size=0.3,
    random_state=0,
    scoring='accuracy'
):
    """
    執行 Logistic Regression 的 GridSearchCV，只輸出三項：最佳參數、分數、最佳模型。

    Returns:
    - best_params_: 最佳參數組合
    - best_score_: 交叉驗證下的最佳準確率
    - best_estimator_: 最佳模型物件
    """
    if param_grid is None:
        param_grid = {
            'solver': ['lbfgs'],
            'C': [1.0]
        }

    opts = dict(tol=tol, max_iter=max_iter)

    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    grid = GridSearchCV(
        estimator=LogisticRegression(**opts),
        param_grid=param_grid,
        cv=cv,
        scoring=scoring
    )

    grid.fit(X_train, y_train)

    print(grid.best_params_)       # e.g. {'C': 0.1, 'solver': 'lbfgs'}
    print(grid.best_score_)        # e.g. 0.9842
    print(grid.best_estimator_)    # e.g. LogisticRegression(C=0.1, ...)
    
    return grid.best_params_, grid.best_score_, grid.best_estimator_