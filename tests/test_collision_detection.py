from typing import List
from src.sphere_collision import find_collisions
import pyvista as pv
import numpy as np


def test_sphere_collision():
    # GIVEN
    centers = np.array(
        [
            [0.7, 0.6, 0.0],
            [0.2, 0.1, 0.1],
            [0.7, 0.6, 0.1],
            [0.3, 0.2, 0.6],
            [0.2, 0.3, 0.9],
            [0.2, 0.0, 0.3],
            [0.9, 0.1, 0.7],
            [0.2, 0.7, 0.4],
            [0.0, 0.6, 0.4],
            [0.9, 0.6, 0.6],
        ]
    )

    _RADIUS = 0.1
    spheres = [pv.Sphere(radius=_RADIUS, center=center) for center in centers]
    sphere_coords_list = [sphere.center for sphere in spheres]

    # WHEN
    query_ids = find_collisions(np.array(sphere_coords_list))

    # THEN
    query_ids_set = set([id for collision_pair in query_ids for id in collision_pair])
    assert sorted(query_ids_set) == sorted({0, 2})
