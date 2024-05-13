AutoEns: An API-Based Intelligent Framework for Automated Ensemble Learning Model Development and Deployment
============================================================================================================

Description
-----------

AutoEns is an intelligent API-based framework designed for automated ensemble learning model development and deployment. It streamlines the process of creating, training, and using ensemble models for various machine learning tasks.

Table of Contents
-----------------
1. Introduction
2. Getting Started
3. API Endpoints
    - Analyze Dataset
    - Model Development
    - Prediction
4. Unit Tests
5. Dataset Citation
6. Contributing
7. License

Introduction
--------------
AutoEns is an API-based framework that automates the process of creating, training, and using ensemble learning models for various machine learning tasks. This documentation outlines the steps to set up and use the AutoEns framework.

Getting Started
----------------
To start using AutoEns, follow these steps:

### 1. Clone the Repository and Change Directory
First, clone the repository and navigate into it using the following commands:

    git clone https://github.com/alih-net/AutoEns.git # or git clone git@github.com:alih-net/AutoEns.git
    cd AutoEns

### 2. Create a Virtual Environment
A virtual environment helps isolate your project's dependencies from the system-wide Python installation.
#### On Linux/macOS
    python3 -m virtualenv venv
    source venv/bin/activate

#### On Windows
    python -m virtualenv venv
    virtualenv\Scripts\activate

### 3. Install Dependencies
Once the virtual environment is activated, use pip to install the required packages from the requirements.txt file.

    pip install -r requirements.txt

API Endpoints
---------------
### 1. Analyze Dataset
Analyze and prepare the dataset for modeling.

Request:
- Endpoint: /analyze
- Method: POST
- Parameters:
    - dataset: Name of the dataset (string)
    - label: Name of the target variable (string)

Example:
`curl --location 'http://127.0.0.1:5000/analyze' --form 'dataset="HeartFailure"' --form 'label="HeartDisease"' `

### 2. Model Development
Develop an ensemble learning model using the pre-processed dataset.

Request:
- Endpoint: /modeling
- Method: POST
- Parameters:
    - dataset: Name of the dataset (string)
    - label: Name of the target variable (string)

Example:
`curl --location 'http://127.0.0.1:5000/modeling' --form 'dataset="HeartFailure"' --form 'label="HeartDisease"' `

### 3. Prediction
Make predictions using the trained ensemble model.

Request:
- Endpoint: /predict
- Method: POST
- Parameters:
    - model: Name of the trained model (string)
    - data: Comma-separated input data values (string)
    - columns: Comma-separated column names corresponding to the input data (string)

Example:
`curl --location 'http://127.0.0.1:5000/predict' --form 'model="HeartFailure"' --form 'data="46,M,ASY,120,277,0,Normal,125,Y,1,Flat"' --form 'columns="Age,Sex,ChestPainType,RestingBloodPressure,Cholesterol,FastingBloodSugar,RestingElectrocardiography,MaxHeartRate,ExerciseAngina,Oldpeak,STSlope"' `

Unit Tests
-----------
To run the unit tests for this project, execute the following command:

`python -m unittest discover -s tests -p "*_test.py"`

The command above will discover and run all the test files that end with _test.py in the tests directory.

After running the tests, you should see an output similar to the following:

    ----------------------------------------------------------------------
    Ran [number] tests in [time]s

    OK

Dataset Citation
-------
As an example, this project uses the "Heart Failure Prediction Dataset" provided by fedesoriano in September 2021. Retrieved in July 2023 from Kaggle: https://www.kaggle.com/fedesoriano/heart-failure-prediction.

Contributing
-------
We welcome contributions from the community. If you find any issues or have suggestions to improve AutoEns, please feel free to open an issue or submit a pull request. Let's collaborate to make ensemble learning more accessible and effective for everyone.

License
-------
This project is licensed under the GNU General Public License version 3 (GPLv3) - see the LICENSE file for details.
