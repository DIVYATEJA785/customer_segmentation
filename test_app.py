import os
import unittest
from app import app

class CustomerSegmentationTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Customer Segmentation", response.data)

    def test_dashboard_page(self):
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Silhouette Score", response.data)

    def test_dataset_page(self):
        response = self.client.get('/dataset')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Customer Dataset", response.data)

    def test_predict_get(self):
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Customer Segment Prediction", response.data)

    def test_predict_post(self):
        form_data = {
            'gender': 'Female',
            'age': '38',
            'income': '2100000',
            'spending': '88',
            'frequency': '30',
            'aov': '8500',
            'total_purchases': '95',
            'recency': '7',
            'online': '54',
            'offline': '41',
            'visits': '30'
        }
        response = self.client.post('/predict', data=form_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Customer belongs to Cluster", response.data)
        self.assertIn(b"High-Value Champions", response.data)

    def test_results_page(self):
        response = self.client.get('/results')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Clustering Results", response.data)

    def test_api_cluster_data(self):
        response = self.client.get('/api/cluster_data')
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn('cluster_counts', json_data)
        self.assertIn('elbow_k', json_data)

    def test_download_file(self):
        response = self.client.get('/download/clustered_customers.csv')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Customer_ID", response.data)

if __name__ == '__main__':
    unittest.main()
