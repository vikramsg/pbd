from typing import List, Tuple
import pyvista as pv
import numpy as np
from src.mesh_utils import matching_points

from src.pbd import _MAX_DIST, PBDMesh, post_solve, solve_collisions, pre_solve, solve


class Scene:
    def __init__(
        self,
        entities: List[PBDMesh],
        obstacle: pv.PolyData,
        stitching_points: List[Tuple[int]],
    ) -> None:
        self.entities = entities

        self.obstacle = obstacle

        self.stitching_points = stitching_points


def stitch_constraint(
    rect1_mesh: PBDMesh,
    rect2_mesh: PBDMesh,
    stitch_index_list: List[Tuple[int]],
    dt: float,
):
    """
    We'll stitch point [1, 0, 0] and [1, 1, 0] with [2, 0, 0] and [2, 1, 0]

    The way PBD works is that we have to update the position of each point
    Then at the end, after satisfying all constraints, we get the velocity
    by subtracting new position with old position. Then we step using this velocity
    """

    compliance_stiffness = 0.01 / dt / dt

    for stitch_index in stitch_index_list:
        w1 = rect1_mesh.weights[stitch_index[0]]
        w2 = rect2_mesh.weights[stitch_index[1]]

        x1 = rect1_mesh.position_1[stitch_index[0]]
        x2 = rect2_mesh.position_1[stitch_index[1]]

        dist = np.linalg.norm(x2 - x1)
        dist_0 = 0

        # FIXME: Doing some tuning here
        if dist < 0.1:
            compliance_stiffness = 0

        # PBD Distance constraint with 0 distance constraint
        # Don't do anything if they are already stitched
        if dist > 0:
            delta_x1 = (
                (w1 / (w1 + w2 + compliance_stiffness))
                * (dist - dist_0)
                * (x2 - x1)
                / dist
            )
            delta_x2 = (
                -(w2 / (w1 + w2 + compliance_stiffness))
                * (dist - dist_0)
                * (x2 - x1)
                / dist
            )

            rect1_mesh.position_1[stitch_index[0]] += delta_x1
            rect2_mesh.position_1[stitch_index[1]] += delta_x2

    return rect1_mesh, rect2_mesh


def simulate(scene: Scene, dt: float) -> Scene:
    scene = Scene(
        entities=[pre_solve(entity, dt) for entity in scene.entities],
        obstacle=scene.obstacle,
        stitching_points=scene.stitching_points,
    )
    scene = Scene(
        entities=[solve(entity, dt) for entity in scene.entities],
        obstacle=scene.obstacle,
        stitching_points=scene.stitching_points,
    )
    # ToDo: this should go in solve somehow
    scene = Scene(
        entities=[
            solve_collisions(entity, scene.obstacle, dt) for entity in scene.entities
        ],
        obstacle=scene.obstacle,
        stitching_points=scene.stitching_points,
    )
    # ToDo: this should go in solve somehow
    scene = Scene(
        entities=[
            mesh
            for mesh in stitch_constraint(
                scene.entities[0], scene.entities[1], scene.stitching_points, dt
            )
        ],
        obstacle=scene.obstacle,
        stitching_points=scene.stitching_points,
    )
    scene = Scene(
        entities=[
            post_solve(entity, dt, max_dist=_MAX_DIST) for entity in scene.entities
        ],
        obstacle=scene.obstacle,
        stitching_points=scene.stitching_points,
    )

    return scene


if __name__ == "__main__":
    center = np.array([0, 0, 0])

    cloth_1 = pv.Plane(
        center=(center + np.array([0.6, 0, 0])),
        direction=(1, 0, 0),
        i_size=2.0,
        j_size=2.0,
        i_resolution=10,
        j_resolution=10,
    )
    cloth_1_triangles = cloth_1.triangulate()

    cloth_2 = pv.Plane(
        center=(center - np.array([0.6, 0, 0])),
        direction=(1, 0, 0),
        i_size=2.0,
        j_size=2.0,
        i_resolution=10,
        j_resolution=10,
    )
    cloth_2_triangles = cloth_2.triangulate()

    stitching_points_full = matching_points(cloth_1_triangles, cloth_2_triangles)
    stitching_points = list(zip(range(11), range(11))) + list(
        zip(range(110, 121), range(110, 121))
    )
    # stitching_points = stitching_points_full

    cloth_1_PBD = PBDMesh(cloth_1_triangles, velocity=[-0.0, 0, 0])
    cloth_2_PBD = PBDMesh(cloth_2_triangles, velocity=[0.0, 0, 0])

    cylinder = pv.Tube(
        pointa=(0, 0, -0.5), pointb=(0, 0, 0.5), radius=0.5, resolution=30, n_sides=45
    )
    cylinder_triangles = cylinder.triangulate()

    dt = 0.075
    scene = Scene(
        entities=[cloth_1_PBD, cloth_2_PBD],
        obstacle=cylinder_triangles,
        stitching_points=stitching_points,
    )

    for _ in range(20):
        scene = simulate(scene, dt)
        print(scene.entities[0].position_1[0], scene.entities[1].position_1[0])
        print("velocity", scene.entities[0].velocity[0], scene.entities[1].velocity[0])

    plotter = pv.Plotter()
    for entity in scene.entities:
        plotter.add_mesh(
            pv.PolyData(entity.position_1, entity.mesh.faces),
            color="green",
            show_edges=True,
        )
    plotter.add_mesh(
        scene.obstacle,
        color="yellow",
        show_edges=True,
    )

    plotter.show()

    print(scene)
