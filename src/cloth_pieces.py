from typing import List
from src.pbd import PBDMesh
import pyvista as pv
import numpy as np

from src.pbd import pre_solve


class Scene:
    def __init__(self, entities: List[PBDMesh]) -> None:
        self.entities = entities


def simulate(scene: Scene, dt: float) -> Scene:
    return Scene([pre_solve(entity, dt) for entity in scene.entities])


if __name__ == "__main__":
    center = np.array([0, 0, 0])

    cloth_1 = pv.Plane(
        center=(center + np.array([1, 0, 0])),
        direction=(1, 0, 0),
        i_size=2.5,
        j_size=2.5,
        i_resolution=25,
        j_resolution=25,
    )
    cloth_1_triangles = cloth_1.triangulate()

    cloth_2 = pv.Plane(
        center=(center - np.array([1, 0, 0])),
        direction=(1, 0, 0),
        i_size=2.5,
        j_size=2.5,
        i_resolution=25,
        j_resolution=25,
    )
    cloth_2_triangles = cloth_2.triangulate()

    cloth_1_PBD = PBDMesh(cloth_1_triangles, velocity=[-0.5, 0, 0])
    cloth_2_PBD = PBDMesh(cloth_2_triangles, velocity=[0.5, 0, 0])

    dt = 0.1
    scene = Scene([cloth_1_PBD, cloth_2_PBD])

    for _ in range(10):
        scene = simulate(scene, dt)
        print(scene.entities[0].position_1[0], scene.entities[1].position_1[0])

    plotter = pv.Plotter()
    for entity in scene.entities:
        plotter.add_mesh(
            pv.PolyData(entity.position_1, entity.mesh.faces),
            color="green",
            show_edges=True,
        )
    plotter.show()

    print(scene)
