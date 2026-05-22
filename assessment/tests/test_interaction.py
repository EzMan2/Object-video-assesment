import unittest

from app.interaction import InteractionDetector, bbox_distance, bbox_iou


class InteractionTests(unittest.TestCase):
    def test_bbox_iou_detects_overlap(self) -> None:
        self.assertEqual(round(bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]), 3), 0.143)

    def test_bbox_distance_uses_box_centers(self) -> None:
        self.assertEqual(bbox_distance([0, 0, 10, 10], [10, 0, 20, 10]), 10)

    def test_interaction_requires_consecutive_frames(self) -> None:
        detector = InteractionDetector(consecutive_frames=2)
        people = [{"bbox": [0, 0, 100, 100]}]

        first = detector.update(1, [80, 80, 120, 120], people, 0, 0.0)
        second = detector.update(1, [82, 82, 122, 122], people, 5, 0.2)

        self.assertIsNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(second["interacted_by_person"], 0)


if __name__ == "__main__":
    unittest.main()
