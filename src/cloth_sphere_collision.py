from typing import List
from pyparsing import col
import pyvista as pv
import numpy as np

from src.pbd import PBDMesh, find_collisions


def pre_solve(cloth: PBDMesh, dt: float) -> PBDMesh:
    """
    This is the velocity predictor step with g=0
    Eventually we will have just one cloth mesh that
    should move and the collision object
    """
    cloth.velocity = cloth.velocity + 0.0

    cloth.position_0 = cloth.position_1.copy()

    cloth.position_1 = cloth.position_0 + dt * cloth.velocity

    return cloth


def solve(cloth: PBDMesh, dt: float) -> PBDMesh:
    """
    Shift positions to satisfy constraints
    """
    cloth = rest_length_constraint(cloth, dt)
    return cloth


def post_solve(cloth: PBDMesh, dt: float) -> PBDMesh:
    """
    Calculate velocity from positions
    """

    cloth.velocity = (cloth.position_1 - cloth.position_0) / dt

    return cloth


def rest_mesh_constraint(mesh: PBDMesh) -> PBDMesh:
    for edge in mesh.edges:
        point_0 = mesh.position_1[edge[0]]
        point_1 = mesh.position_1[edge[1]]

        orig_point_0 = mesh.mesh.points[edge[0]]
        orig_point_1 = mesh.mesh.points[edge[1]]

        w1 = mesh.weights[edge[0]]
        w2 = mesh.weights[edge[1]]

        dist = np.linalg.norm(point_1 - point_0)
        dist_0 = np.linalg.norm(orig_point_1 - orig_point_0)

        if dist > 0:
            delta_x1 = (w1 / (w1 + w2)) * (dist - dist_0) * (point_1 - point_0) / dist
            delta_x2 = -(w2 / (w1 + w2)) * (dist - dist_0) * (point_1 - point_0) / dist

            mesh.position_1[edge[0]] += delta_x1
            mesh.position_1[edge[1]] += delta_x2

    return mesh


def rest_length_constraint(cloth: PBDMesh, dt: float) -> PBDMesh:
    """
    We want the edge lengths to try to get back to original length
    """
    cloth = rest_mesh_constraint(cloth)

    return cloth


def solve_collisions(cloth: PBDMesh, collision_ids: List[int]):
    """
    1. To start with, we will just reset position_1 to position_0
    2. This could get janky but we don't care right now.
    3. Ideally we should only move it to just above the obstacle
    """
    for collision in collision_ids:
        cloth.position_1[collision[0]] = cloth.position_0[collision[0]].copy()

    return cloth


def simulate(cloth: PBDMesh, sphere: pv.PolyData, dt: float) -> None:
    cloth = pre_solve(cloth, dt)
    cloth = solve(cloth, dt)

    collision_ids_list = find_collisions(
        hash_coords=np.concatenate((cloth_PBD.position_1, sphere.points)),
        query_coords=cloth_PBD.position_1,
    )
    if collision_ids_list:
        cloth = solve_collisions(cloth, collision_ids_list)

    cloth = post_solve(cloth, dt)

    return cloth


if __name__ == "__main__":
    center = np.array([0, 0, 0])

    cloth = pv.Plane(
        center=(center + np.array([0, 0, 1])),
        i_size=2.5,
        j_size=2.5,
        i_resolution=25,
        j_resolution=25,
    )
    cloth_triangles = cloth.triangulate()

    cloth_PBD = PBDMesh(cloth_triangles, velocity=[0, 0, -0.5])
    sphere = pv.Sphere(radius=0.5, center=center)

    dt = 0.1
    for _ in range(25):
        cloth_PBD = simulate(cloth_PBD, sphere, dt)

    plotter = pv.Plotter()
    plotter.add_mesh(
        pv.PolyData(cloth_PBD.position_1, cloth_PBD.mesh.faces),
        color="green",
        show_edges=True,
    )
    plotter.add_mesh(sphere, color="yellow", show_edges=True)
    plotter.show()

    print(cloth)
