import numpy as np

from config import ENCIRCLE_RADIUS, ROBOT_COLLISION_DISTANCE


def encircle_error(positions, target):
    distances = np.linalg.norm(positions - target, axis=1)
    errors = np.abs(distances - ENCIRCLE_RADIUS)
    return np.mean(errors)


def spatial_variance(positions):
    group_center = np.mean(positions, axis=0)
    squared_distances = np.sum((positions - group_center) ** 2, axis=1)
    return np.mean(squared_distances)


def count_robot_collisions(positions):
    count = 0

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])

            if d < ROBOT_COLLISION_DISTANCE:
                count += 1

    return count