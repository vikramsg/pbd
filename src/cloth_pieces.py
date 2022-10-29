from typing import List
import pyvista as pv
import numpy as np

from src.pbd import PBDMesh, solve_collisions, pre_solve, solve


class Scene:
    def __init__(self, entities: List[PBDMesh], obstacle: pv.PolyData) -> None:
        self.entities = entities

        self.obstacle = obstacle


def simulate(scene: Scene, dt: float) -> Scene:
    scene = Scene(
        entities=[pre_solve(entity, dt) for entity in scene.entities],
        obstacle=scene.obstacle,
    )
    scene = Scene(
        entities=[solve(entity, dt) for entity in scene.entities],
        obstacle=scene.obstacle,
    )
    scene = Scene(
        entities=[
            solve_collisions(entity, scene.obstacle, dt) for entity in scene.entities
        ],
        obstacle=scene.obstacle,
    )
    return scene


if __name__ == "__main__":
    center = np.array([0, 0, 0])

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

    cloth_1_PBD = PBDMesh(cloth_1_triangles, velocity=[-0.25, 0, 0])
    cloth_2_PBD = PBDMesh(cloth_2_triangles, velocity=[0.25, 0, 0])

    cylinder = pv.Tube(
        pointa=(0, 0, -0.5), pointb=(0, 0, 0.5), radius=0.5, resolution=30, n_sides=45
    )
    cylinder_triangles = cylinder.triangulate()

    dt = 0.075
    scene = Scene(entities=[cloth_1_PBD, cloth_2_PBD], obstacle=cylinder_triangles)

    for _ in range(20):
        scene = simulate(scene, dt)
        print(scene.entities[0].position_1[0], scene.entities[1].position_1[0])

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
