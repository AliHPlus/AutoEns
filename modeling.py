import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
import pickle
import time

from sklearn.ensemble import (
    VotingClassifier,
    StackingClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def load_analysis_report(dataset_name):
    """Load the EDA analysis report produced by /analyze, if it exists."""
    path = os.path.join('analyses', dataset_name, 'analysis_report.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def load_selected_features(dataset_name):
    """Load the feature-selection result produced by /features, if it exists."""
    path = os.path.join('features', dataset_name, 'selected_features.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def evaluate_model(y_true, y_pred, y_prob=None):
    """Return a dict of all classification metrics for one ensemble strategy."""
    metrics = {
        'Accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'Precision': round(float(precision_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
        'Recall': round(float(recall_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
        'F1': round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
        'MSE': round(float(mean_squared_error(y_true, y_pred)), 4),
        'RMSE': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        'AUC_ROC': None,
    }
    if y_prob is not None:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                # Binary classification: use positive-class probability column
                metrics['AUC_ROC'] = round(float(roc_auc_score(y_true, y_prob[:, 1])), 4)
            else:
                # Multiclass: one-vs-rest AUC
                metrics['AUC_ROC'] = round(
                    float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')), 4
                )
        except Exception:
            metrics['AUC_ROC'] = None
    return metrics


def save_metrics_csv(metrics_dict, output_directory):
    """Persist the full metrics table for all ensemble strategies."""
    df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    df.index.name = 'Ensemble_Technique'
    df.to_csv(os.path.join(output_directory, 'models.csv'))


def save_metrics_comparison_plot(metrics_dict, output_directory):
    """Bar chart comparing all strategies across every metric (Reviewer 2 point 4)."""
    df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    plot_metrics = [m for m in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC_ROC']
                    if m in df.columns and df[m].notna().any()]
    if not plot_metrics:
        return

    x = np.arange(len(df))
    width = 0.8 / len(plot_metrics)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, metric in enumerate(plot_metrics):
        offset = (i - len(plot_metrics) / 2 + 0.5) * width
        bars = ax.bar(x + offset, df[metric].fillna(0), width, label=metric)

    ax.set_xlabel('Ensemble Strategy')
    ax.set_ylabel('Score')
    ax.set_title('Ensemble Strategy Comparison — All Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=20, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'metrics_comparison.png'))
    plt.close()


def save_training_time_plot(metrics_dict, output_directory):
    """Horizontal bar chart of training time per strategy (Reviewer 2 point 5)."""
    df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    if 'Training_Time_s' not in df.columns:
        return
    df_sorted = df['Training_Time_s'].sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df_sorted.index, df_sorted.values, color='steelblue')
    ax.set_xlabel('Training time (seconds)')
    ax.set_title('Training Time per Ensemble Strategy')
    for i, (name, val) in enumerate(df_sorted.items()):
        ax.text(val + 0.002, i, f'{val:.3f}s', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'training_time.png'))
    plt.close()


def save_confusion_matrix(y_true, y_pred, output_directory):
    """Save a confusion matrix plot for the best model."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, colorbar=False)
    plt.title('Confusion Matrix — Best Ensemble Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'confusion_matrix.png'))
    plt.close()


def save_roc_curve(y_true, y_prob, output_directory):
    """Save an ROC curve plot for the best model (binary classification only)."""
    try:
        proba = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve — Best Ensemble Model')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, 'roc_curve.png'))
        plt.close()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
#  Main function                                                               #
# --------------------------------------------------------------------------- #

def perform_modeling(dataset_name, label):
    datasets_path = 'datasets/'
    models_path = 'models/'

    dataset = pd.read_csv(os.path.join(datasets_path, dataset_name + '.csv'))

    output_directory = os.path.join(models_path, dataset_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load EDA and feature-selection insights (graceful fallback if absent)
    analysis_report = load_analysis_report(dataset_name)
    selected_features_data = load_selected_features(dataset_name)

    task_type = analysis_report.get('task_type', 'binary_classification')
    recommended_imputer = analysis_report.get('recommended_imputer', 'median')
    imbalance_ratio = analysis_report.get('imbalance_ratio', 1.0)
    drop_candidates = analysis_report.get('drop_candidates', [])

    # ------------------------------------------------------------------ #
    #  1. Encode categorical columns                                       #
    # ------------------------------------------------------------------ #
    categorical_columns = dataset.select_dtypes(include=['object', 'str']).columns.tolist()
    encodings = {}
    for col in categorical_columns:
        codes, uniques = pd.factorize(dataset[col])
        dataset[col] = codes
        # Store {integer_code: original_string} for decoding at prediction time
        encodings[col] = dict(enumerate(uniques.tolist()))

    with open(os.path.join(output_directory, 'encodings.pkl'), 'wb') as f:
        pickle.dump(encodings, f)

    # ------------------------------------------------------------------ #
    #  2. Drop high-correlation features flagged by EDA                   #
    # ------------------------------------------------------------------ #
    columns_to_drop = [c for c in drop_candidates if c in dataset.columns and c != label]
    dataset = dataset.drop(columns=columns_to_drop)

    X = dataset.drop(label, axis=1)
    y = dataset[label]

    # ------------------------------------------------------------------ #
    #  3. Apply feature selection from /features (if available)           #
    # ------------------------------------------------------------------ #
    selected_features = []
    if selected_features_data:
        selected_features = [
            f for f in selected_features_data.get('selected_features', [])
            if f in X.columns
        ]
        if selected_features:
            X = X[selected_features]

    # ------------------------------------------------------------------ #
    #  4. Impute missing values (strategy from EDA report)                #
    # ------------------------------------------------------------------ #
    imputer = SimpleImputer(strategy=recommended_imputer)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # ------------------------------------------------------------------ #
    #  5. Scale features (StandardScaler benefits KNN and LogReg)         #
    # ------------------------------------------------------------------ #
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    # Persist preprocessing artifacts for use by /predict
    with open(os.path.join(output_directory, 'imputer.pkl'), 'wb') as f:
        pickle.dump(imputer, f)
    with open(os.path.join(output_directory, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    preprocessing_config = {
        'task_type': task_type,
        'dropped_columns': columns_to_drop,
        'selected_features': selected_features if selected_features else X_scaled.columns.tolist(),
        'feature_order': X_scaled.columns.tolist(),
        'recommended_imputer': recommended_imputer,
    }
    with open(os.path.join(output_directory, 'preprocessing_config.json'), 'w') as f:
        json.dump(preprocessing_config, f, indent=2)

    # ------------------------------------------------------------------ #
    #  6. Stratified train / test split                                    #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------------------------ #
    #  7. Class-imbalance handling                                        #
    # ------------------------------------------------------------------ #
    # Pass class_weight='balanced' to classifiers that support it when the
    # minority class is less than one-third of the majority class.
    class_weight = 'balanced' if imbalance_ratio > 3.0 else None

    # ------------------------------------------------------------------ #
    #  8. Data-driven weights for Weighted Averaging via 5-fold CV        #
    # ------------------------------------------------------------------ #
    # Train each base learner independently; their relative CV F1 scores
    # determine how much influence each has in the weighted ensemble.
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    base_classifiers = [
        DecisionTreeClassifier(random_state=42, class_weight=class_weight),
        KNeighborsClassifier(),
        RandomForestClassifier(random_state=42, class_weight=class_weight),
    ]

    cv_f1_scores = []
    for clf in base_classifiers:
        scores = cross_val_score(clf, X_train, y_train, cv=cv_splitter, scoring='f1_weighted')
        cv_f1_scores.append(float(np.mean(scores)))

    # Normalise so that proportionally better models carry more weight
    raw = np.array(cv_f1_scores)
    raw = np.clip(raw, 1e-6, None)          # guard against negatives / zero
    data_driven_weights = (raw / raw.sum() * len(raw)).round(2).tolist()

    # ------------------------------------------------------------------ #
    #  9. Train all seven ensemble strategies                              #
    # ------------------------------------------------------------------ #
    clf_dt = DecisionTreeClassifier(random_state=42, class_weight=class_weight)
    clf_knn = KNeighborsClassifier()
    clf_rf = RandomForestClassifier(random_state=42, class_weight=class_weight)
    clf_lr = LogisticRegression(random_state=42, max_iter=1000, class_weight=class_weight)

    clf_dt.fit(X_train, y_train)
    clf_knn.fit(X_train, y_train)
    clf_rf.fit(X_train, y_train)

    estimators = [('dt', clf_dt), ('knn', clf_knn), ('rf', clf_rf)]

    ensemble_strategies = {
        'Max Voting': VotingClassifier(estimators=estimators, voting='hard'),
        'Averaging': VotingClassifier(estimators=estimators, voting='soft'),
        'Weighted Averaging': VotingClassifier(
            estimators=estimators, voting='soft', weights=data_driven_weights
        ),
        'Stacking': StackingClassifier(estimators=estimators, final_estimator=clf_lr),
        'Bagging': BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            n_estimators=10, random_state=42
        ),
        'AdaBoost': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=50, random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
    }

    # ------------------------------------------------------------------ #
    #  10. Evaluate every strategy with the full metric suite             #
    # ------------------------------------------------------------------ #
    metrics_by_strategy = {}
    trained_models = {}

    for name, clf in ensemble_strategies.items():
        t_start = time.time()
        clf.fit(X_train, y_train)
        training_time = round(time.time() - t_start, 4)

        y_pred = clf.predict(X_test)

        y_prob = None
        if hasattr(clf, 'predict_proba'):
            try:
                y_prob = clf.predict_proba(X_test)
            except Exception:
                pass

        metrics = evaluate_model(y_test, y_pred, y_prob)
        # Reviewer 2 point 5: record training time so efficiency claims can be validated
        metrics['Training_Time_s'] = training_time
        metrics_by_strategy[name] = metrics
        trained_models[name] = clf

    # Persist full metrics table and comparison plots
    save_metrics_csv(metrics_by_strategy, output_directory)
    save_metrics_comparison_plot(metrics_by_strategy, output_directory)
    save_training_time_plot(metrics_by_strategy, output_directory)

    # ------------------------------------------------------------------ #
    #  11. Select the best model by F1 (most robust for classification)   #
    # ------------------------------------------------------------------ #
    best_name = max(metrics_by_strategy, key=lambda k: metrics_by_strategy[k]['F1'])
    best_metrics = metrics_by_strategy[best_name]
    best_model = trained_models[best_name]

    with open(os.path.join(output_directory, 'model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)

    # ------------------------------------------------------------------ #
    #  12. Diagnostic plots for the best model                            #
    # ------------------------------------------------------------------ #
    y_pred_best = best_model.predict(X_test)
    save_confusion_matrix(y_test, y_pred_best, output_directory)

    if hasattr(best_model, 'predict_proba'):
        try:
            y_prob_best = best_model.predict_proba(X_test)
            if y_prob_best.shape[1] == 2:
                save_roc_curve(y_test, y_prob_best, output_directory)
        except Exception:
            pass

    # Summarise which EDA insights were applied (Reviewer 1 point 1)
    eda_applied = []
    if columns_to_drop:
        eda_applied.append(f'dropped {len(columns_to_drop)} high-corr feature(s): {columns_to_drop}')
    if selected_features:
        eda_applied.append(f'selected {len(selected_features)} features from /features stage')
    if recommended_imputer != 'median':
        eda_applied.append(f'imputer={recommended_imputer} (EDA-derived)')
    if class_weight == 'balanced':
        eda_applied.append(f'class_weight=balanced (imbalance ratio {imbalance_ratio}x)')
    eda_summary = '; '.join(eda_applied) if eda_applied else 'defaults (run /analyze first for adaptive preprocessing)'

    return (
        f"{output_directory}/model.pkl "
        f"(Best ensemble: {best_name} | "
        f"F1: {best_metrics['F1']:.4f} | "
        f"AUC-ROC: {best_metrics['AUC_ROC']} | "
        f"Accuracy: {best_metrics['Accuracy']:.4f} | "
        f"RMSE: {best_metrics['RMSE']:.4f} | "
        f"EDA applied: {eda_summary})"
    )
