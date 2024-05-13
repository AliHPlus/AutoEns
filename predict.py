import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder

def perform_predict(model, data, columns):
    models_path = 'models/'
    models_path = os.path.join(models_path, model)

    with open(os.path.join(models_path, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)

    new_data = pd.DataFrame([data.split(',')], columns=columns.split(','))

    label_encodings_path = os.path.join(models_path, 'encodings.pkl')
    with open(label_encodings_path, 'rb') as f:
        label_encodings = pickle.load(f)

    for column in label_encodings:
        if column in new_data.columns:
            encoder = LabelEncoder()
            new_data[column] = encoder.fit_transform(new_data[column])
            new_data[column] = new_data[column].replace(dict(zip(encoder.classes_, label_encodings[column])))

    predictions = model.predict(new_data)

    return predictions