from flask import Flask, request, jsonify
from analyze import perform_analyze
from modeling import perform_modeling
from predict import perform_predict

app = Flask(__name__)

@app.route('/')
def homepage():
    return "Welcome to AutoEns!"

@app.route('/analyze', methods=['POST'])
def analyze():
    result = perform_analyze(request.form.get('dataset'), request.form.get('label'))
    return f"Result: {result}"

@app.route('/modeling', methods=['POST'])
def modeling():
    result = perform_modeling(request.form.get('dataset'), request.form.get('label'))
    return f"Result: {result}"

@app.route('/predict', methods=['POST'])
def predict():
    model = request.form.get('model')
    data = request.form.get('data')
    columns = request.form.get('columns')

    result = perform_predict(model, data, columns)

    return f"Result: {result}"

if __name__ == '__main__':
    app.run(debug=True)