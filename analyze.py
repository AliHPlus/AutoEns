import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

def perform_analyze(dataset_name, label):
    datasets_path = 'datasets/'
    analyses_path = 'analyses/'

    dataset = pd.read_csv(os.path.join(datasets_path, dataset_name + '.csv'))

    output_directory = os.path.join(analyses_path, dataset_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Bar plot: Columns vs. Records
    plt.figure(figsize=(12, 8))
    plt.bar(dataset.columns, dataset.count())
    plt.xlabel('Columns')
    plt.ylabel('Records')
    plt.title('Number of records per column')
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'bar_plot.png'))
    plt.close()

    # DataFrame Description
    description_df = dataset.describe()
    description_df.to_csv(os.path.join(output_directory, 'description.csv'))

    # Missing data heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(dataset.isnull(), cmap='viridis')
    plt.title('Missing data heatmap')

    # Adjust the layout to fit the plot inside the figure
    plt.subplots_adjust(left=0.15, bottom=0.27, right=0.9, top=0.9)

    plt.savefig(os.path.join(output_directory, 'missing_data_heatmap.png'))
    plt.close()

    # Missing data analysis
    total = dataset.isnull().sum().sort_values(ascending=False)
    percent = (dataset.isnull().sum() / dataset.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data.to_csv(os.path.join(output_directory, 'missing_data_analysis.csv'))

    # Remove missing data by dropping rows with missing values
    dataset = dataset.dropna()

    # Histograms for all columns
    num_columns = len(dataset.columns)

    # Create a multi-axes layout for the histograms
    fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(12, 8))

    # Histograms for all columns
    plt.figure(figsize=(12, 8))
    dataset.hist(ax=plt.gca())
    plt.suptitle('Histograms of columns')
    plt.savefig(os.path.join(output_directory, 'histograms.png'))
    plt.close()

    # Pairplot to visualize pairwise relationships
    plt.figure(figsize=(12, 8))
    sns.pairplot(dataset, hue=label)
    plt.suptitle('Pairplot')
    plt.savefig(os.path.join(output_directory, 'pairplot.png'))
    plt.close()

    # Label encoding for categorical columns
    categorical_columns = dataset.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        dataset[column] = pd.factorize(dataset[column])[0]

    # Correlation heatmap with all columns (including strings)
    corr_matrix = dataset.corr()

    # Create a larger figure with specified margins
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)

    # Adjust the layout to fit the plot inside the figure
    plt.subplots_adjust(left=0.17, bottom=0.26, right=0.9, top=0.9)

    plt.title('Correlation heatmap')
    plt.savefig(os.path.join(output_directory, 'correlation_heatmap.png'))
    plt.close()

    # Return the output directory path
    return "Saved at " + output_directory
