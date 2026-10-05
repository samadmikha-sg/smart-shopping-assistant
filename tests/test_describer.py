import unittest

from assistant.describer import describe, position, proximity
from assistant.detector import Detection
from assistant.speech import Speaker

W, H = 600, 400


def box(label, x1, y1, x2, y2, conf=0.9):
    return Detection(label, conf, x1, y1, x2, y2)


class DescriberTests(unittest.TestCase):
    def test_position_thirds(self):
        self.assertEqual(position(box("bottle", 0, 0, 100, 100), W), "on your left")
        self.assertEqual(position(box("bottle", 250, 0, 350, 100), W), "ahead")
        self.assertEqual(position(box("bottle", 500, 0, 600, 100), W), "on your right")

    def test_proximity_buckets(self):
        self.assertEqual(proximity(box("person", 0, 0, 400, 300), W, H), "very close")
        self.assertEqual(proximity(box("person", 0, 0, 200, 150), W, H), "nearby")
        self.assertEqual(proximity(box("person", 0, 0, 50, 50), W, H), "further away")

    def test_empty(self):
        self.assertEqual(describe([], W, H), "No objects detected.")

    def test_largest_first_and_limit(self):
        dets = [box("bottle", 0, 0, 40, 40), box("chair", 0, 0, 300, 300),
                box("person", 0, 0, 120, 120), box("cat", 0, 0, 10, 10)]
        text = describe(dets, W, H, max_items=2)
        self.assertTrue(text.startswith("chair"))
        self.assertIn("person", text)
        self.assertNotIn("bottle", text)

    def test_spoken_names(self):
        self.assertIn("table", describe([box("diningtable", 0, 0, 300, 300)], W, H))
        self.assertIn("potted plant", describe([box("pottedplant", 0, 0, 300, 300)], W, H))


class SpeakerTests(unittest.TestCase):
    def test_cooldown_suppresses_repeats(self):
        now = [0.0]
        spoken = []
        s = Speaker(cooldown=4.0, backend=spoken.append, clock=lambda: now[0])
        self.assertTrue(s.say("bottle ahead"))
        now[0] = 2.0
        self.assertFalse(s.say("bottle ahead"))
        self.assertTrue(s.say("chair on your left"))
        now[0] = 5.0
        self.assertTrue(s.say("bottle ahead"))
        self.assertEqual(spoken, ["bottle ahead", "chair on your left", "bottle ahead"])


if __name__ == "__main__":
    unittest.main()
