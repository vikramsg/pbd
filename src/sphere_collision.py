import numpy as np
import pyvista as pv
from typing import List, Tuple

# Radius of spheres
_RADIUS = 0.1
# Max number of objects, should be tuned
_MAX_NUM_OBJECTS = 10

_TABLE_SIZE = 2 * _MAX_NUM_OBJECTS
_SPACING = 0.01


def _int_coord(coord_1d: float) -> int:
    return int(np.floor(coord_1d / _SPACING))


def _hash_coords(x: int, y: int, z: int) -> int:
    # The caret is bitwise XOR!
    # This is a tuned hash function
    hash = (x * 92837111) ^ (y * 689287499) ^ (z * 283923481)
    return np.abs(hash) % _TABLE_SIZE


def create_hash(coords: np.array) -> Tuple[np.array, np.array]:
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

    # fill in objects ids
    for ind in range(num_objects):
        hash = _hash_coords(
            _int_coord(coords[ind][0]),
            _int_coord(coords[ind][1]),
            _int_coord(coords[ind][2]),
        )
        cell_start[hash] -= 1
        cell_entries[cell_start[hash]] = ind

    return cell_start, cell_entries


def create_random_spheres(n: int) -> List[pv.PolyData]:
    # Create n 3D random sphere within [0, 1]**3
    random_centers = np.random.rand(n, 3)

    spheres = [pv.Sphere(radius=_RADIUS, center=center) for center in random_centers]

    return spheres


if __name__ == "__main__":
    n = 5
    spheres = create_random_spheres(n=n)

    sphere_coords_list = [sphere.center for sphere in spheres]
    create_hash(np.array(sphere_coords_list))

    plotter = pv.Plotter()
    _ = [plotter.add_mesh(sphere, color="green") for sphere in spheres]
    cube = pv.Cube(center=(0.5, 0.5, 0.5))
    plotter.add_mesh(cube, opacity=0.25)
    plotter.show()

    print("Finished")
