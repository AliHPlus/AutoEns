import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json


def perform_analyze(dataset_name, label):
    datasets_path = 'datasets/'
    analyses_path = 'analyses/'

    dataset = pd.read_csv(os.path.join(datasets_path, dataset_name + '.csv'))

    output_directory = os.path.join(analyses_path, dataset_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    n_rows = len(dataset)

    # ------------------------------------------------------------------ #
    #  Fig 1 — Missing values per column                                  #
    #  Shows data completeness explicitly.  A flat-zero bar is itself     #
    #  an insight: the dataset contains no missing values.                #
    # ------------------------------------------------------------------ #
    missing_counts = dataset.isnull().sum()
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(dataset.columns, missing_counts, color='steelblue')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Missing value count')
    ax.set_title(f'Missing Values per Column  (dataset: {n_rows} rows)')
    ax.set_xticks(range(len(dataset.columns)))
    ax.set_xticklabels(dataset.columns, rotation=45, ha='right')

    # Annotate each bar with its count and percentage
    for bar, count in zip(bars, missing_counts):
        pct = count / n_rows * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{count}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=8
        )

    if missing_counts.sum() == 0:
        ax.text(
            0.5, 0.6,
            'Dataset is complete — no missing values detected.',
            ha='center', va='center', fontsize=11, color='green',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.7)
        )

    plt.subplots_adjust(bottom=0.25, top=0.88)
    plt.savefig(os.path.join(output_directory, 'bar_plot.png'))
    plt.close()

    # ------------------------------------------------------------------ #
    #  Statistical description                                            #
    # ------------------------------------------------------------------ #
    description_df = dataset.describe()
    description_df.to_csv(os.path.join(output_directory, 'description.csv'))

    # ------------------------------------------------------------------ #
    #  Fig 2 — Missing data heatmap                                       #
    #  If the dataset is fully complete, replacing the blank heatmap with #
    #  an informative text panel avoids a visually meaningless all-one-   #
    #  colour figure (Reviewer 1, point 2).                               #
    # ------------------------------------------------------------------ #
    if missing_counts.sum() > 0:
        plt.figure(figsize=(10, 8))
        sns.heatmap(dataset.isnull(), cmap='viridis', yticklabels=False)
        plt.title('Missing Data Heatmap  (yellow = missing)')
        plt.subplots_adjust(left=0.15, bottom=0.27, right=0.9, top=0.9)
        plt.savefig(os.path.join(output_directory, 'missing_data_heatmap.png'))
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(
            0.5, 0.5,
            'No missing values detected in any column.\nThe dataset is fully complete.',
            ha='center', va='center', fontsize=14, color='green',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.8)
        )
        ax.set_axis_off()
        ax.set_title('Missing Data Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, 'missing_data_heatmap.png'))
        plt.close()

    # ------------------------------------------------------------------ #
    #  Table 1 — Missing data analysis CSV                               #
    #  Includes completeness ratio alongside missing counts so the table  #
    #  carries insight even when all values are zero.                     #
    # ------------------------------------------------------------------ #
    total_missing = dataset.isnull().sum().sort_values(ascending=False)
    pct_missing = (dataset.isnull().mean() * 100).sort_values(ascending=False).round(2)
    completeness = (100 - pct_missing).round(2)
    missing_data = pd.concat(
        [total_missing, pct_missing, completeness],
        axis=1,
        keys=['Missing_Count', 'Missing_Pct', 'Completeness_Pct']
    )
    missing_data.index.name = 'Feature'
    missing_data.to_csv(os.path.join(output_directory, 'missing_data_analysis.csv'))

    # Drop rows with missing values for visual analyses
    dataset_clean = dataset.dropna()

    # ------------------------------------------------------------------ #
    #  Fig 3 — Outlier analysis (IQR method)                             #
    #  Reviewer 2 point 3: "provide more details on how to identify       #
    #  potential outliers based on the distribution."                     #
    #  IQR rule: value < Q1 - 1.5×IQR  or  value > Q3 + 1.5×IQR        #
    # ------------------------------------------------------------------ #
    numeric_cols = dataset_clean.select_dtypes(include=[np.number]).columns.tolist()
    outlier_rows = []
    for col in numeric_cols:
        q1 = dataset_clean[col].quantile(0.25)
        q3 = dataset_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (dataset_clean[col] < lower) | (dataset_clean[col] > upper)
        n_out = int(mask.sum())
        outlier_rows.append({
            'Feature': col,
            'Q1': round(float(q1), 4),
            'Q3': round(float(q3), 4),
            'IQR': round(float(iqr), 4),
            'Lower_Fence': round(float(lower), 4),
            'Upper_Fence': round(float(upper), 4),
            'Outlier_Count': n_out,
            'Outlier_Pct': round(n_out / len(dataset_clean) * 100, 2),
        })

    outlier_df = pd.DataFrame(outlier_rows).set_index('Feature')
    outlier_df.to_csv(os.path.join(output_directory, 'outlier_analysis.csv'))

    # Outlier bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    colours = ['tomato' if v > 0 else 'steelblue' for v in outlier_df['Outlier_Count']]
    ax.bar(outlier_df.index, outlier_df['Outlier_Count'], color=colours)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Outlier count (IQR method)')
    ax.set_title('Outliers per Numeric Feature  (Q1 − 1.5×IQR  /  Q3 + 1.5×IQR  fences)')
    ax.set_xticks(range(len(outlier_df)))
    ax.set_xticklabels(outlier_df.index, rotation=45, ha='right')
    for i, (feat, row) in enumerate(outlier_df.iterrows()):
        ax.text(i, row['Outlier_Count'] + 0.2, str(row['Outlier_Count']),
                ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'outlier_analysis.png'))
    plt.close()

    # ------------------------------------------------------------------ #
    #  Fig 4 — Histograms annotated with outlier fences                  #
    # ------------------------------------------------------------------ #
    dataset_clean.hist(figsize=(14, 10), bins=20, color='steelblue', edgecolor='white')
    plt.suptitle(
        'Feature Distributions  (red dashed lines = IQR outlier fences)',
        fontsize=12, y=1.01
    )
    # Overlay outlier fences on each histogram subplot
    axes = plt.gcf().axes
    for ax, col in zip(axes, [c for c in dataset_clean.columns if c in outlier_df.index]):
        row = outlier_df.loc[col]
        ax.axvline(row['Lower_Fence'], color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(row['Upper_Fence'], color='red', linestyle='--', linewidth=1, alpha=0.7)
        n_out = row['Outlier_Count']
        if n_out > 0:
            ax.set_title(f"{col}  [{n_out} outlier{'s' if n_out != 1 else ''}]", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'histograms.png'))
    plt.close('all')

    # ------------------------------------------------------------------ #
    #  Task type detection                                                 #
    # ------------------------------------------------------------------ #
    label_series = dataset[label].dropna()
    unique_count = label_series.nunique()
    if label_series.dtype == float and unique_count > 20:
        task_type = 'regression'
    elif unique_count == 2:
        task_type = 'binary_classification'
    elif unique_count <= 20:
        task_type = 'multiclass_classification'
    else:
        task_type = 'regression'

    # ------------------------------------------------------------------ #
    #  Fig 5 — Class distribution (classification only)                  #
    #  Reviewer 2 point 2: separates "missing data" from "class          #
    #  imbalance" — two distinct phenomena shown in separate figures.     #
    # ------------------------------------------------------------------ #
    class_balance = {}
    imbalance_ratio = 1.0
    if task_type != 'regression':
        raw_counts = label_series.value_counts().sort_index()
        norm_counts = label_series.value_counts(normalize=True).sort_index()
        class_balance = {str(k): round(float(v), 4) for k, v in norm_counts.items()}
        imbalance_ratio = round(float(raw_counts.max() / raw_counts.min()), 4)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        labels_str = [str(c) for c in raw_counts.index]

        ax1.bar(labels_str, raw_counts.values, color='steelblue')
        ax1.set_xlabel(label)
        ax1.set_ylabel('Count')
        ax1.set_title(f'Class Distribution — {label}')
        for i, (lbl, cnt) in enumerate(zip(labels_str, raw_counts.values)):
            ax1.text(i, cnt + 1, str(cnt), ha='center', va='bottom', fontsize=10)

        ax2.pie(
            raw_counts.values,
            labels=[f'{l} ({norm_counts[k]:.1%})' for l, k in zip(labels_str, raw_counts.index)],
            autopct='%1.1f%%',
            colors=plt.cm.Set2.colors[:len(raw_counts)]
        )
        ax2.set_title(f'Class Proportion — {label}  (imbalance ratio: {imbalance_ratio}x)')

        plt.suptitle(
            'Class Distribution Analysis\n'
            '(Note: imbalance is distinct from missing data shown in Fig 1)',
            fontsize=11
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, 'class_distribution.png'))
        plt.close()

    # ------------------------------------------------------------------ #
    #  Fig 6 — Pairplot coloured by label                                #
    # ------------------------------------------------------------------ #
    pairplot_fig = sns.pairplot(dataset_clean, hue=label)
    pairplot_fig.fig.suptitle('Pairplot — Pairwise Feature Relationships', y=1.02)
    pairplot_fig.savefig(os.path.join(output_directory, 'pairplot.png'))
    plt.close('all')

    # ------------------------------------------------------------------ #
    #  Encode categoricals for correlation analysis                       #
    # ------------------------------------------------------------------ #
    dataset_encoded = dataset_clean.copy()
    categorical_columns = dataset_encoded.select_dtypes(include=['object', 'str']).columns.tolist()
    for col in categorical_columns:
        dataset_encoded[col] = pd.factorize(dataset_encoded[col])[0]

    # ------------------------------------------------------------------ #
    #  Fig 7 — Correlation heatmap with significant-pair annotation       #
    # ------------------------------------------------------------------ #
    corr_matrix = dataset_encoded.corr()

    # Collect significant correlations for annotation
    sig_pairs = []
    cols_list = [c for c in corr_matrix.columns if c != label]
    for i in range(len(cols_list)):
        for j in range(i + 1, len(cols_list)):
            r = corr_matrix.loc[cols_list[i], cols_list[j]]
            if abs(r) >= 0.7:
                direction = 'positive' if r > 0 else 'negative'
                sig_pairs.append(f'{cols_list[i]} ↔ {cols_list[j]}: r={r:.2f} ({direction})')

    fig, ax = plt.subplots(figsize=(13, 9))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                linewidths=0.5, annot_kws={'size': 8})
    plt.subplots_adjust(left=0.17, bottom=0.28, right=0.92, top=0.88)

    title = 'Correlation Heatmap'
    if sig_pairs:
        title += f'\nStrong correlations (|r| ≥ 0.7): ' + ' | '.join(sig_pairs[:4])
        if len(sig_pairs) > 4:
            title += f' ... +{len(sig_pairs) - 4} more'
    else:
        title += '\nNo strong correlations (|r| ≥ 0.7) found between features.'
    ax.set_title(title, fontsize=9, pad=10)
    plt.savefig(os.path.join(output_directory, 'correlation_heatmap.png'))
    plt.close()

    # ------------------------------------------------------------------ #
    #  Analysis report — machine-readable EDA summary consumed by         #
    #  /features and /modeling for data-driven preprocessing decisions.   #
    # ------------------------------------------------------------------ #

    # Missing percentage per column
    missing_pct = {
        col: round(float(dataset[col].isnull().mean()), 4)
        for col in dataset.columns
    }

    # High-correlation feature pairs (|corr| > 0.95, excluding label)
    abs_corr = corr_matrix.drop(columns=[label], errors='ignore') \
                          .drop(index=[label], errors='ignore').abs()
    high_corr_pairs = []
    drop_candidates = set()
    cols = abs_corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs_corr.iloc[i, j] > 0.95:
                high_corr_pairs.append([cols[i], cols[j], round(float(abs_corr.iloc[i, j]), 4)])
                drop_candidates.add(cols[j])

    # Skewness of numeric columns (excluding label)
    num_cols_enc = dataset_encoded.select_dtypes(include=[np.number]).columns
    skewness = {
        col: round(float(dataset_encoded[col].skew()), 4)
        for col in num_cols_enc
        if col != label
    }

    # Highly skewed features (|skewness| > 1.5) benefit from log transform
    highly_skewed = [col for col, sk in skewness.items() if abs(sk) > 1.5]

    # Imputation strategy: median for skewed distributions, mean otherwise
    missing_cols = [c for c, pct in missing_pct.items() if pct > 0 and c != label]
    if missing_cols:
        avg_skew = float(np.mean([abs(skewness.get(c, 0.0)) for c in missing_cols]))
        recommended_imputer = 'median' if avg_skew > 0.5 else 'mean'
    else:
        recommended_imputer = 'median'

    # Outlier counts for the report
    outlier_counts = {
        row['Feature']: row['Outlier_Count']
        for row in outlier_rows
    }
    n_outliers_total = int(sum(outlier_counts.values()))

    analysis_report = {
        'dataset': dataset_name,
        'label': label,
        'task_type': task_type,
        'n_samples': int(n_rows),
        'n_features': int(len(dataset.columns) - 1),
        'class_balance': class_balance,
        'imbalance_ratio': imbalance_ratio,
        'missing_pct': missing_pct,
        'high_correlation_pairs': high_corr_pairs,
        'drop_candidates': list(drop_candidates),
        'skewness': skewness,
        'highly_skewed_features': highly_skewed,
        'outlier_counts': outlier_counts,
        'n_outliers_total': n_outliers_total,
        'categorical_columns': categorical_columns,
        'recommended_imputer': recommended_imputer,
        'recommended_scaler': 'standard',
    }

    report_path = os.path.join(output_directory, 'analysis_report.json')
    with open(report_path, 'w') as f:
        json.dump(analysis_report, f, indent=2)

    return "Saved at " + output_directory
