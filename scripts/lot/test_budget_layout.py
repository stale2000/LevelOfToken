"""Contract tests for equal-work random placement controls."""
import collections
import unittest

from budget_layout import permute_root_layout
from layout import TokenRect, layout_from_rects


def extent_histogram(layout):
    result = collections.Counter()
    for rect in layout.rects:
        result[rect.extent] += rect.et * rect.eh * rect.ew
    return result


class PlacementControlTests(unittest.TestCase):
    def test_root_shuffle_preserves_tokens_extent_histogram_and_valid_cover(self):
        rects = [TokenRect(0, 0, 0, 1, 4, 4)]
        rects += [TokenRect(0, u, v, 1, 2, 2)
                  for u in (0, 2) for v in (4, 6)]
        rects += [TokenRect(0, u, v, 1, 1, 1)
                  for u in range(4, 8) for v in range(4)]
        rects += [TokenRect(0, u, v, 1, 2, 2)
                  for u in (4, 6) for v in (4, 6)]
        layout = layout_from_rects(1, 8, 8, rects)
        shuffled = permute_root_layout(layout, seed=42)
        again = permute_root_layout(layout, seed=42)
        self.assertEqual(layout.count, shuffled.count)
        self.assertEqual(layout.dense_count, shuffled.dense_count)
        self.assertEqual(extent_histogram(layout), extent_histogram(shuffled))
        self.assertNotEqual(set(layout.rects), set(shuffled.rects))
        self.assertEqual(shuffled, again)

    def test_nonmultiple_boundary_roots_remain_valid(self):
        rects = [TokenRect(0, 0, 0, 1, 4, 4),
                 TokenRect(0, 0, 4, 1, 4, 2),
                 TokenRect(0, 4, 0, 1, 2, 4),
                 TokenRect(0, 4, 4, 1, 2, 2)]
        layout = layout_from_rects(1, 6, 6, rects)
        shuffled = permute_root_layout(layout, seed=7)
        self.assertEqual(extent_histogram(layout), extent_histogram(shuffled))
        self.assertEqual(set(layout.rects), set(shuffled.rects))

    def test_root_shuffle_uses_same_spatial_permutation_for_every_frame(self):
        frame_rects = [TokenRect(0, 0, 0, 1, 4, 4)]
        frame_rects += [TokenRect(0, u, v, 1, 2, 2)
                        for u in (0, 2) for v in (4, 6)]
        frame_rects += [TokenRect(0, u, v, 1, 1, 1)
                        for u in range(4, 8) for v in range(4)]
        frame_rects += [TokenRect(0, u, v, 1, 2, 2)
                        for u in (4, 6) for v in (4, 6)]
        rects = [TokenRect(t, r.u, r.v, r.et, r.eh, r.ew)
                 for t in range(2) for r in frame_rects]
        layout = layout_from_rects(2, 8, 8, rects)
        shuffled = permute_root_layout(layout, seed=42)
        for t in range(2):
            self.assertEqual(
                {(r.u, r.v, r.et, r.eh, r.ew) for r in shuffled.rects if r.t == t},
                {(r.u, r.v, r.et, r.eh, r.ew) for r in shuffled.rects if r.t == 0},
            )
        self.assertEqual(layout.count, shuffled.count)
        self.assertEqual(extent_histogram(layout), extent_histogram(shuffled))


if __name__ == '__main__':
    unittest.main()
