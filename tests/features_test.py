import unittest
from app import app


class FeaturesTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    # Test case for /features endpoint
    def test_features_endpoint(self):
        response = self.app.post('/features', data={
            'dataset': 'HeartFailure',
            'label': 'HeartDisease',
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('features/HeartFailure', response.data.decode('utf-8'))

    def test_features_response_contains_selection_summary(self):
        response = self.app.post('/features', data={
            'dataset': 'HeartFailure',
            'label': 'HeartDisease',
        })
        body = response.data.decode('utf-8')
        # Response should report how many features were selected out of total
        self.assertIn('features selected', body)


if __name__ == '__main__':
    unittest.main()
