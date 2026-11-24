"""Small, read-only soft-state measurements at every physical integration step.

Rows: time, minimum tetrahedron volume ratio, bag/table penetration,
bear/table penetration, maximum cloth-edge stretch ratio, all-finite flag.
These measurements never repair, constrain, or move simulated vertices.
"""
from __future__ import annotations

import numpy as np


def tetra_volumes(points, tets):
    a, b, c, d = np.moveaxis(np.asarray(points)[tets], 1, 0)
    return np.einsum("ij,ij->i", b-a, np.cross(c-a, d-a))/6.


class PhysicalMonitor:
    columns = ["time_s", "minimum_tetra_volume_ratio",
               "bag_table_penetration_m", "bear_table_penetration_m",
               "maximum_bag_edge_stretch_ratio", "all_soft_states_finite"]

    def __init__(self, rest, config):
        self.tets = np.asarray(rest["bear_tets"], int)
        faces = np.asarray(rest["bag_faces"], int)
        self.edges = np.unique(np.sort(np.concatenate(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1), axis=0)
        self.rest_volumes = tetra_volumes(rest["bear"], self.tets)
        self.rest_lengths = np.linalg.norm(
            rest["bag"][self.edges[:, 1]]-rest["bag"][self.edges[:, 0]], axis=1)
        if np.any(np.abs(self.rest_volumes) < 1e-14) or np.any(self.rest_lengths < 1e-10):
            raise ValueError("Degenerate reference mesh cannot be monitored")
        self.table_height = float(config["table"]["height"])
        self.table_half = np.asarray(config["table"]["size"][:2])/2

    def measure(self, time, bag, bear):
        bag, bear = np.asarray(bag), np.asarray(bear)
        if not (np.isfinite(bag).all() and np.isfinite(bear).all()):
            return np.array([time, np.nan, np.nan, np.nan, np.nan, 0.])
        ratio = tetra_volumes(bear, self.tets)/self.rest_volumes
        stretch = np.linalg.norm(
            bag[self.edges[:, 1]]-bag[self.edges[:, 0]], axis=1)/self.rest_lengths
        penetration = []
        for points in (bag, bear):
            covered = np.all(np.abs(points[:, :2]) <= self.table_half, axis=1)
            penetration.append(max(0., self.table_height-float(points[covered, 2].min()))
                               if covered.any() else 0.)
        return np.array([time, ratio.min(), *penetration, stretch.max(), 1.])
