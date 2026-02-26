import pandas as pd
import numpy as np
import os
import json
import pickle


def perform_predict(model_name, data, columns):
    models_path = os.path.join('models', model_name)
    features_path = os.path.join('features', model_name)

    # ------------------------------------------------------------------ #
    #  Load model and preprocessing artifacts                             #
    # ------------------------------------------------------------------ #
    with open(os.path.join(models_path, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(models_path, 'encodings.pkl'), 'rb') as f:
        encodings = pickle.load(f)

    # Preprocessing config records exactly how the training data was shaped
    config_path = os.path.join(models_path, 'preprocessing_config.json')
    preprocessing_config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            preprocessing_config = json.load(f)

    # ------------------------------------------------------------------ #
    #  Parse raw input                                                     #
    # ------------------------------------------------------------------ #
    column_list = [c.strip() for c in columns.split(',')]
    data_values = [v.strip() for v in data.split(',')]
    new_data = pd.DataFrame([data_values], columns=column_list)

    # ------------------------------------------------------------------ #
    #  Apply label encodings for categorical columns                      #
    # ------------------------------------------------------------------ #
    # encodings[col] = {integer_code: original_string}
    # Reverse it to map original_string → integer_code for incoming data.
    for col, mapping in encodings.items():
        if col in new_data.columns:
            reverse_mapping = {str(v): int(k) for k, v in mapping.items()}
            new_data[col] = new_data[col].map(reverse_mapping)

    # Convert every column to numeric (handles any remaining string columns)
    for col in new_data.columns:
        new_data[col] = pd.to_numeric(new_data[col], errors='coerce')

    # ------------------------------------------------------------------ #
    #  Drop columns that were removed during training                     #
    # ------------------------------------------------------------------ #
    dropped = preprocessing_config.get('dropped_columns', [])
    new_data = new_data.drop(columns=[c for c in dropped if c in new_data.columns], errors='ignore')

    # ------------------------------------------------------------------ #
    #  Apply feature selection (reorder to match training feature order)  #
    # ------------------------------------------------------------------ #
    feature_order = preprocessing_config.get('feature_order', new_data.columns.tolist())
    # Keep only the features the model was trained on, in the correct order
    available = [f for f in feature_order if f in new_data.columns]
    new_data = new_data.reindex(columns=available)

    # ------------------------------------------------------------------ #
    #  Apply imputer and scaler                                           #
    # ------------------------------------------------------------------ #
    imputer_path = os.path.join(models_path, 'imputer.pkl')
    if os.path.exists(imputer_path):
        with open(imputer_path, 'rb') as f:
            imputer = pickle.load(f)
        new_data = pd.DataFrame(
            imputer.transform(new_data), columns=new_data.columns
        )

    scaler_path = os.path.join(models_path, 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        new_data = pd.DataFrame(
            scaler.transform(new_data), columns=new_data.columns
        )

    # ------------------------------------------------------------------ #
    #  Run inference                                                       #
    # ------------------------------------------------------------------ #
    predictions = model.predict(new_data)

    # ------------------------------------------------------------------ #
    #  Uncertainty estimation via Shannon entropy (Reviewer 2 point 6)   #
    #  Entropy = 0 % → model is fully certain about the prediction.       #
    #  Entropy = 100 % → model is maximally uncertain (uniform proba).    #
    # ------------------------------------------------------------------ #
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(new_data)
            proba_rounded = np.round(proba, 4)

            # Shannon entropy per prediction row
            eps = 1e-10
            entropy = -np.sum(proba * np.log2(proba + eps), axis=1)
            max_entropy = np.log2(max(proba.shape[1], 2))
            uncertainty_pct = round(float(entropy[0] / max_entropy * 100), 1)
            confidence_pct = round(100 - uncertainty_pct, 1)

            return (
                f"Prediction: {predictions.tolist()} | "
                f"Probabilities: {proba_rounded.tolist()} | "
                f"Confidence: {confidence_pct}% | "
                f"Uncertainty (entropy): {uncertainty_pct}%"
            )
        except Exception:
            pass

    return f"Prediction: {predictions.tolist()}"
