from typing import List, Tuple
import pyvista as pv
import numpy as np
from src.mesh_utils import matching_points
from pathlib import Path

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

        self.compliance_stiffness = 1


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

    compliance_stiffness = 0.1 / dt / dt

    for stitch_index in stitch_index_list:
        w1 = rect1_mesh.weights[stitch_index[0]]
        w2 = rect2_mesh.weights[stitch_index[1]]

        x1 = rect1_mesh.position_1[stitch_index[0]]
        x2 = rect2_mesh.position_1[stitch_index[1]]

        dist = np.linalg.norm(x2 - x1)
        dist_0 = 0

        # FIXME: Doing some tuning here
        if dist < 0.05:
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

    # ToDo: this should go in solve somehow
    scene = Scene(
        entities=[
            solve_collisions(entity, scene.obstacle, dt) for entity in scene.entities
        ],
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
            mesh
            for mesh in stitch_constraint(
                scene.entities[0],
                scene.entities[1],
                scene.stitching_points,
                dt=dt,
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
    tshirt_front = pv.read(Path(".").resolve() / "data" / "avatar" / "tshirt_front.obj")
    tshirt_back = pv.read(Path(".").resolve() / "data" / "avatar" / "tshirt_back.obj")
    tshirt_front = tshirt_front.scale([1.1, 1.1, 1.1])
    tshirt_back = tshirt_back.scale([1.1, 1.1, 1.1])

    tshirt_front_PBD = PBDMesh(tshirt_front, velocity=[-0.0, 0, 0])
    tshirt_back_PBD = PBDMesh(tshirt_back, velocity=[0.0, 0, 0])

    avatar_mesh_path = Path(".").resolve() / "data" / "avatar" / "spread_arms.obj"
    avatar_mesh = pv.read(avatar_mesh_path)
    avatar_mesh = avatar_mesh.scale([0.01, 0.01, 0.01])

    avatar_mesh_triangles = avatar_mesh.triangulate()

    stitching_points = [
        (1, 1),
        (13, 13),
        (14, 14),
        (2, 2),
        (3, 3),
        (15, 15),
        (17, 17),
        (18, 18),
        (5, 5),
        (6, 6),
        (22, 22),
        (21, 21),
        (24, 24),
        (8, 8),
        (9, 9),
        (25, 25),
        (26, 26),
        (10, 10),
    ]

    dt = 0.075
    scene = Scene(
        entities=[tshirt_front_PBD, tshirt_back_PBD],
        obstacle=avatar_mesh_triangles,
        stitching_points=stitching_points,
    )

    for _ in range(10):
        scene = simulate(scene, dt)
        print(
            "bottom", scene.entities[0].position_1[0], scene.entities[1].position_1[0]
        )
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
