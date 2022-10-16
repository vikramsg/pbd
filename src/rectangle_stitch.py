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

        self.position_1 = self.mesh.points.copy()


def create_rectangles():
    rect1 = pv.Rectangle(points=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]))
    rect2 = pv.Rectangle(points=np.array([[2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0]]))

    # We create points and faces
    # For eg. rect1_mesh.points and rect1_mesh.faces
    rect1_mesh = PBDMesh(mesh=rect1.triangulate())
    rect2_mesh = PBDMesh(mesh=rect2.triangulate())

    return rect1_mesh, rect2_mesh


def pre_solve(rect1_mesh: PBDMesh, rect2_mesh: PBDMesh, dt: float):
    """
    This is the velocity predictor step with g=0
    Eventually we will have just one cloth mesh that
    should move and the collision object
    """
    rect1_mesh.velocity = rect1_mesh.velocity + 0.0
    rect2_mesh.velocity = rect2_mesh.velocity + 0.0

    rect1_mesh.mesh.points = rect1_mesh.position_1.copy()
    rect2_mesh.mesh.points = rect2_mesh.position_1.copy()

    rect1_mesh.position_1 = rect1_mesh.mesh.points + dt * rect1_mesh.velocity
    rect2_mesh.position_1 = rect2_mesh.mesh.points + dt * rect2_mesh.velocity

    return rect1_mesh, rect2_mesh


def solve(rect1_mesh: PBDMesh, rect2_mesh: PBDMesh, dt: float):
    """
    Shift positions to satisfy constraints
    """
    stitch_constraint(rect1_mesh, rect2_mesh, dt)


def post_solve(rect1_mesh: PBDMesh, rect2_mesh: PBDMesh, dt: float):
    """
    Calculate velocity from positions
    """

    rect1_mesh.velocity = (rect1_mesh.position_1 - rect1_mesh.mesh.points) / dt
    rect2_mesh.velocity = (rect2_mesh.position_1 - rect2_mesh.mesh.points) / dt

    return rect1_mesh, rect2_mesh


def stitch_constraint(rect1_mesh: PBDMesh, rect2_mesh: PBDMesh, dt: float):
    """
    We'll stitch point [1, 0, 0] and [1, 1, 0] with [2, 0, 0] and [2, 1, 0]

    The way PBD works is that we have to update the position of each point
    Then at the end, after satisfying all constraints, we get the velocity
    by subtracting new position with old position. Then we step using this velocity
    """

    # Harcoded stitching array
    # We want each point in the first index stitched to the same index
    # in the second array
    stitch_index_mesh1 = [1, 2]
    stitch_index_mesh2 = [0, 3]

    for it, stitch_index in enumerate(stitch_index_mesh1):
        w1 = rect1_mesh.weights[stitch_index]
        w2 = rect2_mesh.weights[stitch_index_mesh2[it]]

        x1 = rect1_mesh.mesh.points[stitch_index]
        x2 = rect2_mesh.mesh.points[stitch_index_mesh2[it]]

        dist = np.linalg.norm(x2 - x1)
        dist_0 = 0

        # PBD Distance constraint with 0 distance constraint
        delta_x1 = (w1 / (w1 + w2)) * (dist - dist_0) * (x2 - x1) / dist
        delta_x2 = -(w2 / (w1 + w2)) * (dist - dist_0) * (x2 - x1) / dist

        rect1_mesh.position_1[stitch_index] += delta_x1
        rect2_mesh.position_1[stitch_index_mesh2[it]] += delta_x2

    return rect1_mesh, rect2_mesh


def simulate(rect1_mesh: PBDMesh, rect2_mesh: PBDMesh, dt: float):
    rect1_mesh, rect2_mesh = pre_solve(rect1_mesh, rect2_mesh, dt)
    rect1_mesh, rect2_mesh = solve(rect1_mesh, rect2_mesh, dt)
    rect1_mesh, rect2_mesh = post_solve(rect1_mesh, rect2_mesh, dt)


if __name__ == "__main__":
    rect1_mesh, rect2_mesh = create_rectangles()

    dt = 1 / 10.0
    simulate(rect1_mesh, rect2_mesh, dt)

    plotter = pv.Plotter()
    plotter.add_mesh(rect1_mesh, show_edges=True)
    plotter.add_mesh(rect2_mesh, show_edges=True)
    plotter.show()

    print(plotter)
