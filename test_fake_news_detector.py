import unittest

from fake_news_detector import NaiveBayesFakeNewsDetector, build_default_detector


class FakeNewsDetectorTests(unittest.TestCase):
    def test_predict_fake_like_content(self):
        detector = build_default_detector()
        prediction = detector.predict("Conspiracy claims secret machine controls storms")
        self.assertEqual(prediction.label, "fake")
        self.assertGreaterEqual(prediction.confidence, 0.5)

    def test_predict_real_like_content(self):
        detector = build_default_detector()
        prediction = detector.predict("Government releases official policy and budget report")
        self.assertEqual(prediction.label, "real")
        self.assertGreaterEqual(prediction.confidence, 0.5)

    def test_fit_requires_samples(self):
        detector = NaiveBayesFakeNewsDetector()
        with self.assertRaises(ValueError):
            detector.fit([])


if __name__ == "__main__":
    unittest.main()
