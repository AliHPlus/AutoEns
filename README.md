# AutoEns: An Intelligent Framework for Automated Ensemble Learning Model Development and Deployment

## Overview

AutoEns is an API-based intelligent framework designed for automated ensemble learning model development and deployment. It streamlines the process of creating, training, and using ensemble models for various machine learning tasks through a four-stage pipeline:

1. **Data Analysis** - Comprehensive EDA with missing value analysis, outlier detection, and class distribution
2. **Feature Engineering** - Automated feature importance ranking and selection
3. **Model Development** - Adaptive ensemble training with 7 techniques and full metric evaluation
4. **Deployment** - Real-time predictions with uncertainty quantification

## Key Innovations

- **EDA-Driven Preprocessing**: AutoEns reads insights from exploratory data analysis to automatically configure imputation, feature selection, and class balancing
- **Adaptive Weighted Averaging**: Ensemble weights derived from cross-validation performance rather than manual tuning
- **Uncertainty Quantification**: Shannon entropy-based confidence scores for every prediction
- **Comprehensive Evaluation**: Full metric suite including Accuracy, Precision, Recall, F1, AUC-ROC, MSE, RMSE, and Training Time

## Installation

```bash
# Clone the repository
git clone https://github.com/alih-net/AutoEns.git
cd AutoEns

# Create and activate virtual environment
python3 -m virtualenv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

```bash
python app.py
```

The Flask server starts on `http://127.0.0.1:5000`

## API Endpoints

### 1. Data Analysis (`/analyze`)

Performs comprehensive exploratory data analysis and generates insights for subsequent stages.

**Request:**
```bash
curl -X POST http://127.0.0.1:5000/analyze \
     --form 'dataset="HeartFailure"' \
     --form 'label="HeartDisease"'
```

**Outputs:**
- `bar_plot.png` - Missing values per column with annotations
- `missing_data_heatmap.png` - Missing data visualization
- `missing_data_analysis.csv` - Missing value statistics with completeness %
- `outlier_analysis.csv` / `.png` - IQR-based outlier detection
- `class_distribution.png` - Class balance visualization with imbalance ratio
- `histograms.png` - Feature distributions with outlier fences
- `pairplot.png` - Pairwise feature relationships
- `correlation_heatmap.png` - Correlation matrix with significant pairs annotated
- `analysis_report.json` - Machine-readable EDA summary

### 2. Feature Engineering (`/features`)

Performs feature importance ranking and selection based on 95% cumulative importance threshold.

**Request:**
```bash
curl -X POST http://127.0.0.1:5000/features \
     --form 'dataset="HeartFailure"' \
     --form 'label="HeartDisease"'
```

**Outputs:**
- `feature_importance.csv` / `.png` - Ranked feature importance
- `cumulative_importance.png` - Cumulative importance curve
- `feature_profile.csv` - Per-feature type, missing %, skewness, rank
- `selected_features.json` - Feature subset consumed by modeling

### 3. Model Development (`/modeling`)

Trains 7 ensemble techniques and selects the best based on F1 (weighted) score.

**Request:**
```bash
curl -X POST http://127.0.0.1:5000/modeling \
     --form 'dataset="HeartFailure"' \
     --form 'label="HeartDisease"'
```

**Ensemble Techniques:**
| Technique | Description |
|----------|-------------|
| Max Voting | Majority vote from base learners |
| Averaging | Soft voting with probability averaging |
| Weighted Averaging | CV-derived weights |
| Stacking | Meta-learner on base predictions |
| Bagging | Bootstrap aggregating |
| AdaBoost | Sequential boosting |
| Gradient Boosting | Gradient-based boosting |

**Outputs:**
- `model.pkl` - Best trained ensemble model
- `models.csv` - Full metrics table (Accuracy, Precision, Recall, F1, AUC-ROC, MSE, RMSE, Training Time)
- `confusion_matrix.png` - Classification performance visualization
- `roc_curve.png` - ROC curve (binary classification)
- `metrics_comparison.png` - All strategies compared
- `training_time.png` - Computational efficiency chart
- `preprocessing_config.json` - Feature order, dropped columns, scaler/imputer config
- `encodings.pkl`, `imputer.pkl`, `scaler.pkl` - Preprocessing artifacts

### 4. Prediction (`/predict`)

Makes predictions with uncertainty quantification.

**Request:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
     --form 'model="HeartFailure"' \
     --form 'data="46,M,ASY,120,277,0,Normal,125,Y,1,Flat"' \
     --form 'columns="Age,Sex,ChestPainType,RestingBloodPressure,Cholesterol,FastingBloodSugar,RestingElectrocardiography,MaxHeartRate,ExerciseAngina,Oldpeak,STSlope"'
```

**Response:**
```
Prediction: [1] | Probabilities: [[0.23, 0.77]] | Confidence: 77% | Uncertainty (entropy): 23%
```

## Testing

```bash
# Run all tests
python -m unittest discover -s tests -p "*_test.py"

# Run specific test
python -m unittest tests.analyze_test
python -m unittest tests.features_test
python -m unittest tests.modeling_test
python -m unittest tests.predict_test
```

## Dataset Citation

This project uses the "Heart Failure Prediction Dataset" by fedesoriano (Kaggle, September 2021).  
Available at: https://www.kaggle.com/fedesoriano/heart-failure-prediction

## License

This project is licensed under the GNU General Public License version 3 (GPLv3). See the LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request to help improve AutoEns.
