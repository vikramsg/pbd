import pyvista as pv
import numpy as np


class PBDMesh:
    def __init__(self, mesh: pv.PolyData):
        self.mesh = mesh

        # Each point must have a weight
        # If the point is static, its inverse must be 0
        # Maybe it makes more sense to have inverse weight
        self.weights = np.ones(self.mesh.n_points)

        self.velocity = np.zeros_like(self.mesh.points)

        self.position_0 = self.mesh.points.copy()
        self.position_1 = self.mesh.points.copy()

        self.edges = self.extract_edges(self.mesh)

    def extract_edges(self, mesh: pv.PolyData):
        edges = mesh.extract_all_edges()

        return edges.lines.reshape(-1, 3)[:, 1:]


def pre_solve(cloth: PBDMesh, dt: float):
    """
    This is the velocity predictor step with g=0
    Eventually we will have just one cloth mesh that
    should move and the collision object
    """
    cloth.velocity = cloth.velocity + 0.0

    cloth.position_0 = cloth.position_1.copy()

    cloth.position_1 = cloth.position_0 + dt * cloth.velocity

    return cloth


def solve(cloth: PBDMesh, dt: float):
    """
    Shift positions to satisfy constraints
    """
    cloth = rest_length_constraint(cloth, dt)
    return cloth


def post_solve(cloth: PBDMesh, dt: float):
    """
    Calculate velocity from positions
    """

    cloth.velocity = (cloth.position_1 - cloth.position_0) / dt

    return cloth


def rest_mesh_constraint(mesh: PBDMesh):
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


def rest_length_constraint(cloth: PBDMesh, dt: float):
    """
    We want the edge lengths to try to get back to original length
    """
    cloth = rest_mesh_constraint(cloth)

    return cloth


def simulate(cloth: PBDMesh, dt: float):
    cloth = pre_solve(cloth, dt)
    cloth = solve(cloth, dt)
    cloth = post_solve(cloth, dt)

    return cloth


if __name__ == "__main__":
    center = np.array([0, 0, 0])

    cloth = pv.Plane(center=(center + np.array([0, 0, 1])))
    cloth_triangles = cloth.triangulate()

    cloth_PBD = PBDMesh(cloth_triangles)

    sphere = pv.Sphere(radius=0.5, center=center)

    plotter = pv.Plotter()
    plotter.add_mesh(cloth_triangles, color="green", show_edges=True)
    plotter.add_mesh(sphere, color="yellow", show_edges=True)
    plotter.show()

    print(cloth)
