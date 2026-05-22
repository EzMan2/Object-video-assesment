import unittest

from app.tracker import SimpleTracker


class TrackerTests(unittest.TestCase):
    def test_tracker_assigns_stable_object_id_and_motion_interval(self) -> None:
        tracker = SimpleTracker()
        first = [{"class": "bottle", "confidence": 0.8, "bbox": [0, 0, 20, 20], "center": [10, 10]}]
        second = [{"class": "bottle", "confidence": 0.9, "bbox": [20, 0, 40, 20], "center": [30, 10]}]

        tracker.update(first, 0, 0.0)
        tracker.update(second, 5, 0.2)
        objects = tracker.export_tracks()

        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["object_id"], 1)
        self.assertEqual(objects[0]["motion_history"][-1], {"frame_range": [5, 5], "state": "moving"})


if __name__ == "__main__":
    unittest.main()
