import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def perform_features(dataset_name, label):
    datasets_path = 'datasets/'
    features_path = 'features/'
    analyses_path = 'analyses/'

    dataset = pd.read_csv(os.path.join(datasets_path, dataset_name + '.csv'))

    output_directory = os.path.join(features_path, dataset_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load EDA analysis report if available
    analysis_report_path = os.path.join(analyses_path, dataset_name, 'analysis_report.json')
    analysis_report = {}
    if os.path.exists(analysis_report_path):
        with open(analysis_report_path, 'r') as f:
            analysis_report = json.load(f)

    task_type = analysis_report.get('task_type', 'binary_classification')
    categorical_columns = analysis_report.get('categorical_columns', [])

    # Encode categorical columns using the same factorize approach as modeling
    for col in dataset.select_dtypes(include=['object', 'str']).columns:
        dataset[col] = pd.factorize(dataset[col])[0]

    # Drop rows with missing values
    dataset = dataset.dropna()

    X = dataset.drop(label, axis=1)
    y = dataset[label]
    feature_names = X.columns.tolist()

    # Feature importance via RandomForest (100 trees for stable estimates)
    if task_type == 'regression':
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

    rf.fit(X, y)
    importances = rf.feature_importances_

    # Sort features by importance descending
    order = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in order]
    sorted_importances = [float(importances[i]) for i in order]
    cumulative_importance = list(np.cumsum(sorted_importances))

    # --- Feature importance CSV ---
    importance_df = pd.DataFrame({
        'Feature': sorted_features,
        'Importance': sorted_importances,
        'Cumulative_Importance': [round(c, 4) for c in cumulative_importance],
    })
    importance_df.to_csv(os.path.join(output_directory, 'feature_importance.csv'), index=False)

    # --- Feature importance horizontal bar chart ---
    n = len(sorted_features)
    plt.figure(figsize=(12, max(5, n * 0.45)))
    bars = plt.barh(sorted_features[::-1], sorted_importances[::-1], color='steelblue')
    plt.xlabel('Importance Score')
    plt.title(f'Feature Importance — {dataset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'feature_importance.png'))
    plt.close()

    # --- Feature selection: retain features covering >= 95 % cumulative importance ---
    n_selected = int(np.searchsorted(cumulative_importance, 0.95)) + 1
    n_selected = min(n_selected, n)   # never exceed total feature count
    n_selected = max(n_selected, 1)   # always keep at least one feature
    selected_features = sorted_features[:n_selected]

    # --- Feature profile: per-feature summary table ---
    profile_rows = []
    for feat in feature_names:
        rank = sorted_features.index(feat) + 1
        profile_rows.append({
            'Feature': feat,
            'Type': 'categorical' if feat in categorical_columns else 'numeric',
            'Missing_Pct': analysis_report.get('missing_pct', {}).get(feat, 0.0),
            'Skewness': analysis_report.get('skewness', {}).get(feat, 0.0),
            'Importance_Rank': rank,
            'Importance_Score': round(float(importances[feature_names.index(feat)]), 6),
            'Cumulative_Importance': round(cumulative_importance[rank - 1], 4),
            'Selected': feat in selected_features,
        })

    profile_df = pd.DataFrame(profile_rows).sort_values('Importance_Rank').reset_index(drop=True)
    profile_df.to_csv(os.path.join(output_directory, 'feature_profile.csv'), index=False)

    # --- Cumulative importance curve ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n + 1), cumulative_importance, marker='o', color='steelblue')
    plt.axhline(y=0.95, color='red', linestyle='--', label='95 % threshold')
    plt.axvline(x=n_selected, color='orange', linestyle='--', label=f'Selected: {n_selected} features')
    plt.xlabel('Number of features (ranked by importance)')
    plt.ylabel('Cumulative importance')
    plt.title(f'Cumulative Feature Importance — {dataset_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'cumulative_importance.png'))
    plt.close()

    # --- Selected features JSON (consumed by /modeling) ---
    selected_features_data = {
        'dataset': dataset_name,
        'label': label,
        'selected_features': selected_features,
        'n_total': n,
        'n_selected': n_selected,
        'cumulative_importance_retained': round(cumulative_importance[n_selected - 1], 4),
    }
    with open(os.path.join(output_directory, 'selected_features.json'), 'w') as f:
        json.dump(selected_features_data, f, indent=2)

    retained = cumulative_importance[n_selected - 1]
    return (
        f"Saved at {output_directory} "
        f"({n_selected}/{n} features selected, {retained:.1%} importance retained)"
    )
