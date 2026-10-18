import unittest
import numpy as np
from finite_data import appearance, feature, object_state, scene


class FiniteInputTests(unittest.TestCase):
    def test_observed_texture_and_terminal_removal(self):
        vector = feature(appearance(np.random.default_rng(7)))
        self.assertEqual(len(vector), 128)
        self.assertAlmostEqual(np.linalg.norm(vector), 1)
        self.assertEqual(object_state(31)[1], "removed")
        self.assertEqual(object_state(99)[1], "removed")
        record = scene(7, "one", "test")
        self.assertEqual(len(record["frames"]), 32)
        for frame in record["frames"]:
            self.assertEqual(len(frame["cues"]), 8)
            for observation in frame["observations"]:
                self.assertNotIn("state", observation)
                self.assertNotIn("exists", observation)
        self.assertFalse(record["frames"][-1]["observations"])
