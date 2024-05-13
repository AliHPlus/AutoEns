import unittest
from app import app

class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    # Test case for /predict endpoint
    def test_predict_endpoint(self):
        data = "46,M,ASY,120,277,0,Normal,125,Y,1,Flat"
        columns = "Age,Sex,ChestPainType,RestingBloodPressure,Cholesterol,FastingBloodSugar,RestingElectrocardiography,MaxHeartRate,ExerciseAngina,Oldpeak,STSlope"
        response = self.app.post('/predict', data={'model': 'HeartFailure', 'data': data, 'columns': columns})
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()