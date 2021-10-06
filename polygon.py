import numpy as np
import logging
from shapely.geometry import Point
from shapely.geometry import Polygon as SPolygon
from copy import deepcopy

logging.basicConfig(level=logging.DEBUG)

ELE_THRESHOLD = 2.5


class Seg:
    def __init__(self, start: np.array, end: np.array, next_seg=None):
        self.start = start
        self.end = end
        if self.start[0] == self.end[0] and self.start[1] == self.end[1]:
            raise BaseException("Repeat point")

        self.next = next_seg
        self.ele = (start[2] + end[2]) / 2
        self.length_2d = np.linalg.norm(self.end[0:2] - self.start[0:2])
        self.dir_2d = (self.end[0:2] - self.start[0:2]) / self.length_2d
        self.polygon = None

    def _is_parallel(self, seg):
        return (self.dir_2d[0] == seg.dir_2d[0] and self.dir_2d[1] == seg.dir_2d[1]) or \
               (self.dir_2d[0] == -seg.dir_2d[0] and self.dir_2d[1] == -seg.dir_2d[1])

    def get_t(self, pt_2d):
        delta = pt_2d - self.start[0:2]
        if abs(np.linalg.norm(delta) - abs(np.dot(delta, self.dir_2d))) > 1e-6:
            raise IOError(f"Point {pt_2d} not in this seg")

        if delta[0] == 0 and delta[1] == 0:
            return 0

        if delta[0] == 0:
            return delta[1] / (self.end[1] - self.start[1])

        elif delta[1] == 0:
            return delta[0] / (self.end[0] - self.start[0])

        else:
            return (delta[0] / (self.end[0] - self.start[0]) + delta[1] / (self.end[1] - self.start[1])) / 2

    def get_pt(self, t):
        if t < 0 or t > 1:
            raise IOError("t should be in [0, 1]")

        if t == 0:
            return self.start.copy()

        if t == 1:
            return self.end.copy()

        pt = np.zeros((3, ))
        pt[0:2] = self.start[0:2] + self.length_2d * t * self.dir_2d
        pt[2] = self.start[2] + t * (self.end[2] - self.start[2])

        return pt

    def cross_pt(self, seg):
        if abs(self.ele - seg.ele) > ELE_THRESHOLD:
            return None

        if self._is_parallel(seg):
            return None

        pt = np.zeros((3,), dtype=float)

        A = np.array(
            [
                [self.dir_2d[1], -self.dir_2d[0]],
                [seg.dir_2d[1], -seg.dir_2d[0]]
            ], dtype=float
        )

        b = np.array(
            [self.dir_2d[1] * self.start[0] - self.dir_2d[0] * self.start[1],
             seg.dir_2d[1] * seg.start[0] - seg.dir_2d[0] * seg.start[1]], dtype=float
        )

        pt[0:2] = np.linalg.solve(A, b)
        try:
            t1 = self.get_t(pt[0:2])
            t2 = seg.get_t(pt[0:2])
        except IOError:
            raise IOError(f"cur seg is {self.start}, {self.end}, {self.dir_2d}, "
                          f"other seg is {seg.start}, {seg.end}, {seg.dir_2d}")

        if t1 < 0 or t1 > 1 or t2 < 0 or t2 > 1:
            return None

        pt[2] = self.start[2] + t1 * (self.end[2] - self.start[2])

        return pt

    def __eq__(self, other):
        return (self.start == other.start).all()\
               and (self.end == other.end).all() \
               and self.polygon == other.polygon


class Polygon:
    def __init__(self, pt_list: [np.array], id: str):
        self.id = id
        segs = self.create_segs(0, pt_list)
        self.segs = self.check_segs(segs)
        self.pt_list = pt_list[:]

    def create_segs(self, i, pt_list):
        if i >= len(pt_list) - 1:
            return None

        seg = Seg(pt_list[i], pt_list[i + 1])
        seg.polygon = self.id
        seg.next = self.create_segs(i + 1, pt_list)

        return seg

    def check_segs(self, segs: Seg):
        # check if is reasonable, and create a circle
        dummy = Seg(np.zeros((3,)), np.ones((3,)), segs)
        cur, nex = segs, segs.next
        while nex:
            assert cur.end[0] == nex.start[0] and cur.end[1] == nex.start[1], "Not continue"

            cur = cur.next
            nex = nex.next

        if cur.end[0] == dummy.next.start[0] and cur.end[1] == dummy.next.start[1]:
            # has circled
            cur.next = dummy.next

        else:
            new_seg = Seg(cur.end.copy(), dummy.next.start.copy(), dummy.next)
            new_seg.polygon = self.id
            cur.next = new_seg

        return dummy.next

    @staticmethod
    def merge_two_segs(seg1: Seg,
                       seg2: Seg,
                       pt_list: list,
                       start: Seg,
                       has_look: float):
        cur = seg2
        min_t1 = 1e6
        min_t2 = 1e6
        min_seg = None

        while cur:
            cross_pt = seg1.cross_pt(cur)
            if cross_pt is not None:
                t1 = seg1.get_t(cross_pt[0:2])
                t2 = cur.get_t(cross_pt[0:2])

                if t1 <= has_look:
                    if cur.next == seg2:
                        break

                    cur = cur.next
                    continue

                if t2 == 0 and np.cross(cur.dir_2d, seg1.dir_2d) <= 0:
                    if cur.next == seg2:
                        break

                    cur = cur.next
                    continue

                if t2 == 1 and np.cross(cur.dir_2d, seg1.dir_2d) >= 0:
                    if cur.next == seg2:
                        break

                    cur = cur.next
                    continue

                if t1 < min_t1:
                    min_t1 = t1
                    min_t2 = t2
                    min_seg = deepcopy(cur)

            if cur.next == seg2:
                break

            cur = cur.next

        if min_seg is not None:
            cross = seg1.get_pt(min_t1)
            pt_list.append(cross.copy())
            # logging.debug("recurrent in first...")
            Polygon.merge_two_segs(min_seg, seg1, pt_list, start, has_look=min_t2)

        else:
            pt_list.append(seg1.end.copy())
            if seg1.next == start:
                return pt_list

            else:
                # logging.debug("recurrent in second...")
                Polygon.merge_two_segs(seg1.next, seg2, pt_list, start, -1)

    def merge_two_polygons(self, poly):
        start_1, seg_1 = self.segs, self.segs
        start_2, seg_2 = poly.segs, poly.segs

        pt_list1 = deepcopy(self.pt_list)
        pt_list2 = deepcopy(poly.pt_list)

        if pt_list1[-1][0] != pt_list1[0][0] or pt_list1[-1][1] != pt_list1[0][1]:
            pt_list1.append(pt_list1[0])

        if pt_list2[-1][0] != pt_list2[0][0] or pt_list2[-1][1] != pt_list2[0][1]:
            pt_list2.append(pt_list2[0])

        pt1 = [deepcopy(pt[0:2].tolist()) for pt in pt_list1]
        pt2 = [deepcopy(pt[0:2].tolist()) for pt in pt_list2]

        poly1 = SPolygon(pt1)
        poly2 = SPolygon(pt2)

        result = []

        while seg_1:
            pt = Point(seg_1.start[0:2].tolist())
            if not pt.within(poly2) and not pt.within(poly2.exterior):
                # logging.debug("has in this function")
                self.merge_two_segs(seg_1, seg_2, result, seg_1, 0)
                return result

            if seg_1.next == start_1:
                break

            seg_1 = seg_1.next

        return result


if __name__ == '__main__':
    tri_pts = [
        np.array([0, 0, 0]), np.array([0.5, 0, 0]), np.array([0.5, 1, 0]), np.array([-1, 1, 0])
    ]

    bbox_pts = [
        np.array([-0.5, 0, 0]), np.array([0.5, 0, 0]),
        np.array([0.5, 1, 0]), np.array([-0.5, 0.5, 0])
    ]

    tri = Polygon(tri_pts, "0")
    box = Polygon(bbox_pts, "1")
    seg_tri = tri.segs
    seg_box = box.segs

    # for i in range(6):
    #     print(seg_tri.polygon)
    #     seg_tri = seg_tri.next
    #
    # for i in range(8):
    #     print(seg_box.polygon)
    #     seg_box = seg_box.next

    print(tri.merge_two_polygons(box))
