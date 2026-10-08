import unittest
import json
import joblib
import pandas as pd

from app import app


class TestPredictionApplication(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv("breast_cancer_v1_100.csv")
        cls.data = cls.data.loc[
            :, ~cls.data.columns.str.contains("^Unnamed")
        ]

        cls.model = joblib.load("breast_cancer_model.pkl")

        cls.features = list(cls.model.feature_names_in_)

        cls.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_prediction_with_real_dataset_row(self):
        row = self.data.iloc[0]

        sample = {
            feature: row[feature]
            for feature in self.features
        }

        expected_prediction = int(
            self.model.predict(
                pd.DataFrame([sample])
            )[0]
        )

        response = self.client.post(
            "/predict",
            json=sample
        )

        self.assertEqual(response.status_code, 200)

        result = response.get_json()

        self.assertEqual(
            result["prediction_code"],
            expected_prediction
        )

        self.assertEqual(
            result["prediction"],
            f"CLASS_{expected_prediction}"
        )

    def test_missing_field_validation(self):
        row = self.data.iloc[0]

        sample = {
            feature: row[feature]
            for feature in self.features
        }

        # Remove one required feature
        removed_feature = self.features[0]
        del sample[removed_feature]

        response = self.client.post(
            "/predict",
            json=sample
        )

        self.assertEqual(response.status_code, 400)

        result = response.get_json()

        self.assertIn("missing_fields", result)
        self.assertIn(
            removed_feature,
            result["missing_fields"]
        )


if __name__ == "__main__":
    unittest.main()
