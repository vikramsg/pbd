import pyvista as pv
from typing import List
from scipy.spatial.distance import cdist
import numpy as np


def matching_points(mesh1: pv.PolyData, mesh2: pv.PolyData) -> List[List[int]]:
    edges1 = mesh1.extract_feature_edges()
    edges2 = mesh2.extract_feature_edges()

    # Find distance matrix where the 2 edges match in y and z
    edges_dist_mat = cdist(edges1.points[:, 1:3], edges2.points[:, 1:3])
    edges_match_points = np.nonzero(edges_dist_mat < 1e-4)

    # where is edges1.points[edges_match_points[0]] in mesh1
    # where is edges2.points[edges_match_points[1]] in mesh2
    dist_1_mat = cdist(edges1.points[edges_match_points[0]], mesh1.points)
    edge_mesh_1_match_points = np.nonzero(dist_1_mat < 1e-4)

    dist_2_mat = cdist(edges2.points[edges_match_points[1]], mesh2.points)
    edge_mesh_2_match_points = np.nonzero(dist_2_mat < 1e-4)

    return list(zip(edge_mesh_1_match_points[1], edge_mesh_2_match_points[1]))


if __name__ == "__main__":
    center = [0, 0, 0]
    cloth_1 = pv.Plane(
        center=(center + np.array([0.6, 0, 0])),
        direction=(1, 0, 0),
        i_size=2.5,
        j_size=2.5,
        i_resolution=25,
        j_resolution=25,
    )
    cloth_1_triangles = cloth_1.triangulate()

    cloth_2 = pv.Plane(
        center=(center - np.array([0.6, 0, 0])),
        direction=(1, 0, 0),
        i_size=2.5,
        j_size=2.5,
        i_resolution=25,
        j_resolution=25,
    )
    cloth_2_triangles = cloth_2.triangulate()

    matching_points(cloth_1_triangles, cloth_2_triangles)
