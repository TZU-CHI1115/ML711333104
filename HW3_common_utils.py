# HW3_common_utils.py

# 基本套件
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-learn：資料處理
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scikit-learn：模型
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier

# Scikit-learn：評估工具
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


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

def apply_pca(X_train_scaled, X_test_scaled, variance_ratio=0.8):
    pca = PCA(n_components=variance_ratio)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    return X_train_pca, X_test_pca, pca

def run_logistic_regression_cv(
    X_train_scaled, X_test_scaled, y_train, y_test,
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
    
    回傳內容包含訓練時間與測試結果。
    """

    start_time = time.time()

    clf = LogisticRegressionCV(
        solver=solver,
        Cs=Cs,
        cv=cv,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose
    )

    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    end_time = time.time()
    elapsed_time = end_time - start_time

    train_score = accuracy_score(y_train, clf.predict(X_train_scaled))
    test_score = accuracy_score(y_test, y_pred)

    if print_report:
        print(f"Logistic Regression with CV (solver = {solver})")
        print(f"Best C = {clf.C_}")
        print(f"Training Accuracy: {train_score:.2%}")
        print(f"Testing Accuracy : {test_score:.2%}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        #print(classification_report(y_test, y_pred))

    return {
        'model': clf,
        'best_C': clf.C_,
        'train_acc': train_score,
        'test_acc': test_score,
        'y_pred': y_pred,
        'elapsed_time': elapsed_time
    }

def run_logistic_gridcv(
    X_train_scaled, y_train,
    param_grid=None,
    tol=1e-6,
    max_iter=int(1e6),
    cv=5,
    test_size=0.3,
    random_state=0,
    scoring='accuracy',
    print_report=True
):
    """
    執行 Logistic Regression 的 GridSearchCV，輸出最佳參數、CV分數、最佳模型與耗時。

    Returns:
    dict: 包含 best_params, best_score, best_estimator, elapsed_time
    """

    if param_grid is None:
        param_grid = {
            'solver': ['lbfgs'],
            'C': [1.0]
        }

    opts = dict(tol=tol, max_iter=max_iter)

    cv_split = StratifiedShuffleSplit(
        n_splits=cv, test_size=test_size, random_state=random_state
    )

    start_time = time.time()

    grid = GridSearchCV(
        estimator=LogisticRegression(**opts),
        param_grid=param_grid,
        cv=cv_split,
        scoring=scoring
    )

    grid.fit(X_train_scaled, y_train)

    elapsed_time = time.time() - start_time

    if print_report:
        print(grid.best_params_)        
        print(grid.best_score_)         
        print(grid.best_estimator_)     
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

    return {
        'best_params': grid.best_params_,
        'best_score': grid.best_score_,
        'best_estimator': grid.best_estimator_,
        'elapsed_time': elapsed_time
    }


def run_svm_classification(
    X_train_scaled, X_test_scaled, y_train, y_test,
    kernel='linear',
    C=1,
    gamma='scale',
    degree=3,
    tol=1e-6,                    # ← 新增 tol 參數
    max_iter=int(1e6),          # ← 也建議把 max_iter 拉出來
    use_linear_svc=False,
    print_report=True
):
    """
    執行 SVM 分類並統一回傳 dict 格式，支援多 kernel 與 LinearSVC。
    """

    start_time = time.time()

    opts = dict(C=C, tol=tol, max_iter=max_iter)

    if use_linear_svc:
        clf = LinearSVC(**opts)
        kernel_used = 'LinearSVC'
    else:
        clf = SVC(kernel=kernel, gamma=gamma, degree=degree, **opts)
        kernel_used = f'SVC-{kernel}'

    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)

    elapsed_time = time.time() - start_time

    if print_report:
        print(f"SVM Classification ({kernel_used})")
        print(f"Test Accuracy: {test_acc:.2%}")
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")

    return {
        'model': clf,
        'test_acc': test_acc,
        'y_pred': y_pred,
        'elapsed_time': elapsed_time
    }



def run_mlp_classifier(
    X_train_scaled, X_test_scaled, y_train, y_test,
    hidden_layers=(100,),
    activation='relu',
    solver='adam',
    max_iter=int(1e6),
    tol=1e-6,
    print_report=True,
    random_state=None
):
    """
    執行 MLP 分類器，訓練與測試，支援輸出報告。

    Returns:
    dict: 包含模型、準確率、預測結果、訓練時間
    """

    opts = dict(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
        random_state=random_state
    )

    start_time = time.time()

    clf = MLPClassifier(**opts)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)

    end_time = time.time()
    elapsed_time = end_time - start_time

    if print_report:
        print("MLP Classifier Report")
        print(f"Test Accuracy: {test_acc:.2%}")
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")
        #print(classification_report(y_test, y_pred))

    return {
        'model': clf,
        'test_acc': test_acc,
        'y_pred': y_pred,
        'elapsed_time': elapsed_time
    }

def plot_mlp_loss_curve(model):
    """
    繪製 MLP 的訓練損失曲線。
    """
    if hasattr(model, 'loss_curve_'):
        plt.figure(figsize=(5, 3))
        plt.plot(model.loss_curve_)
        plt.title('Training Loss Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

def plot_mlp_confusion_matrix(model, X_test, y_test):
    """
    繪製 MLP 的混淆矩陣。
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))  # 小一點的圖
    score = 100 * model.score(X_test, y_test)
    title = 'Testing score = {:.2f}%'.format(score)
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        xticks_rotation=45,
        cmap=plt.cm.Blues,
        normalize='true',
        ax=ax
    )
    disp.ax_.set_title(title)
    plt.tight_layout()
    plt.show()