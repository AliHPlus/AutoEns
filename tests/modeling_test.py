import unittest
from app import app

class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    # Test case for /modeling endpoint
    def test_modeling_endpoint(self):
        response = self.app.post('/modeling', data={'dataset': 'HeartFailure', 'label': 'HeartDisease'})
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()