import numpy as np
import pyvista as pv
from typing import List, Tuple

# Radius of spheres
_RADIUS = 0.1
# Max number of objects, should be tuned
_MAX_NUM_OBJECTS = 500

_TABLE_SIZE = 2 * _MAX_NUM_OBJECTS
_SPACING = _RADIUS
_MAX_DIST = 2 * _RADIUS

# FIXME: Only for debugging purposes
_SELF_COORDS = None


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


def create_random_spheres(n: int) -> List[pv.PolyData]:
    # Create n 3D random sphere within [0, 1]**3
    random_centers = np.random.rand(n, 3)

    spheres = [pv.Sphere(radius=_RADIUS, center=center) for center in random_centers]

    return spheres


def find_collisions(coords: np.array) -> List[List[np.array]]:
    cell_start, cell_entries = create_hash(coords)

    query_ids = [
        query_collision(coord, _MAX_DIST, cell_start, cell_entries) for coord in coords
    ]

    collisions = []
    for index, coord in enumerate(coords):
        # Remove self
        query_ids_set = set(query_ids[index]) - {index}

        coord_query_ids = list(query_ids_set)
        if coord_query_ids:
            for query_id in coord_query_ids:
                if np.linalg.norm(coord - coords[query_id]) < 2 * _RADIUS:
                    collisions.append([index, query_id])

    return collisions


if __name__ == "__main__":
    n = 10
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
    _SELF_COORDS = sphere_coords_list

    query_ids = find_collisions(np.array(sphere_coords_list))

    query_ids_set = set([id for collision_pair in query_ids for id in collision_pair])
    all_ids_set = {i for i in range(len(sphere_coords_list))}

    green_sphere_ids = all_ids_set - query_ids_set
    yellow_sphere_ids = query_ids_set.copy()

    plotter = pv.Plotter()
    _ = [
        plotter.add_mesh(spheres[sphere_id], color="green")
        for sphere_id in green_sphere_ids
    ]
    _ = [
        plotter.add_mesh(spheres[sphere_id], color="yellow")
        for sphere_id in yellow_sphere_ids
    ]
    cube = pv.Cube(center=(0.5, 0.5, 0.5))
    plotter.add_mesh(cube, opacity=0.25)
    plotter.show()

    print("Finished")
