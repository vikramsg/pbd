from typing import List, Tuple
from pyparsing import col
import pyvista as pv
import numpy as np

# Radius of spheres
_RADIUS = 0.05
# Max number of objects, should be tuned
_MAX_NUM_OBJECTS = 10000

_TABLE_SIZE = 2 * _MAX_NUM_OBJECTS
_SPACING = _RADIUS
_MAX_DIST = 2 * _RADIUS


class PBDMesh:
    def __init__(
        self, mesh: pv.PolyData, velocity: List[float] = [0.0, 0.0, 0.0]
    ) -> None:
        self.mesh = mesh

        # Each point must have a weight
        # If the point is static, its inverse must be 0
        # Maybe it makes more sense to have inverse weight
        self.weights = np.ones(self.mesh.n_points)

        self.velocity = np.tile(velocity, self.mesh.n_points).reshape(-1, 3)

        self.position_0 = self.mesh.points.copy()
        self.position_1 = self.mesh.points.copy()

        self.edges = self.extract_edges(self.mesh)

    def extract_edges(self, mesh: pv.PolyData) -> np.ndarray:
        edges = mesh.extract_all_edges()

        return edges.lines.reshape(-1, 3)[:, 1:]


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


def solve(cloth: PBDMesh, dt: float) -> PBDMesh:
    """
    Shift positions to satisfy constraints
    """
    cloth = rest_length_constraint(cloth, dt)
    return cloth


def _int_coord(coord_1d: float) -> int:
    return int(np.floor(coord_1d / _SPACING))


def _hash_coords(x: int, y: int, z: int) -> int:
    # The caret is bitwise XOR!
    # This is a tuned hash function
    hash = (x * 92837111) ^ (y * 689287499) ^ (z * 283923481)
    return np.abs(hash) % _TABLE_SIZE


def create_hash(coords: np.array) -> Tuple[np.array, np.array]:
    # This seems a strange data structure which should be possible
    # to do with a sparse column stored vector
    # It does exactly what is mentioned in the notes.
    # https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/11-hashing.pdf
    num_objects = len(coords)

    cell_start = np.zeros(_TABLE_SIZE + 1, dtype=int)
    cell_entries = np.zeros(_MAX_NUM_OBJECTS, dtype=int)

    # determine cell sizes
    for ind in range(num_objects):
        hash = _hash_coords(
            _int_coord(coords[ind][0]),
            _int_coord(coords[ind][1]),
            _int_coord(coords[ind][2]),
        )
        cell_start[hash] += 1

    # determine cell starts
    start = 0
    for ind in range(_TABLE_SIZE):
        start += cell_start[ind]
        cell_start[ind] = start

    cell_start[_TABLE_SIZE] = start  # guard

    # fill in object ids
    for ind in range(num_objects):
        hash = _hash_coords(
            _int_coord(coords[ind][0]),
            _int_coord(coords[ind][1]),
            _int_coord(coords[ind][2]),
        )
        cell_start[hash] -= 1
        cell_entries[cell_start[hash]] = ind

    return cell_start, cell_entries


def query_collision(
    query_coord: np.array,
    max_dist: float,
    cell_start: np.array,
    cell_entries: np.array,
) -> List[int]:
    """For a given co-ordinate, we return all co-ordinate ids that must be checked for collision
       FIXME:
       The data structure does not make sense since cell_start and cell_entries only make sense
       together with the co-ordinates

    Args:
        query_coord (np.array): coordinate to check against
        max_dist (float): max distance for checking
    """
    x0 = _int_coord(query_coord[0] - max_dist)
    y0 = _int_coord(query_coord[1] - max_dist)
    z0 = _int_coord(query_coord[2] - max_dist)

    x1 = _int_coord(query_coord[0] + max_dist)
    y1 = _int_coord(query_coord[1] + max_dist)
    z1 = _int_coord(query_coord[2] + max_dist)

    query_ids = []

    for xi in range(x0, x1):
        for yi in range(y0, y1):
            for zi in range(z0, z1):
                hash = _hash_coords(xi, yi, zi)

                start = cell_start[hash]
                end = cell_start[hash + 1]

                for index in range(start, end):
                    query_ids.append(cell_entries[index])

    return list(set(query_ids))


def find_collisions(hash_coords: np.array, query_coords: np.array) -> List[List[int]]:
    cell_start, cell_entries = create_hash(hash_coords)

    query_ids = [
        query_collision(coord, _MAX_DIST, cell_start, cell_entries)
        for coord in query_coords
    ]

    collisions = []
    for index, coord in enumerate(query_coords):
        # Remove self
        query_ids_set = set(query_ids[index]) - {index}

        coord_query_ids = list(query_ids_set)
        if coord_query_ids:
            for query_id in coord_query_ids:
                if np.linalg.norm(coord - hash_coords[query_id]) < 2 * _RADIUS:
                    collisions.append([index, query_id])

    return collisions


def solve_collisions_for_ids(cloth: PBDMesh, collision_ids: List[int]) -> PBDMesh:
    """
    1. To start with, we will just reset position_1 to position_0
    2. This could get janky but we don't care right now.
    3. Ideally we should only move it to just above the obstacle
    """
    for collision in collision_ids:
        cloth.position_1[collision[0]] = cloth.position_0[collision[0]].copy()

    return cloth


def solve_collisions(cloth: PBDMesh, obstacle: pv.PolyData, dt: float) -> PBDMesh:
    collision_ids_list = find_collisions(
        hash_coords=np.concatenate((cloth.position_1, obstacle.points)),
        query_coords=cloth.position_1,
    )
    if collision_ids_list:
        cloth = solve_collisions_for_ids(cloth, collision_ids_list)

    return cloth
