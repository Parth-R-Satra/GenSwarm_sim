import numpy as np

from config import (
    ARENA_LIMIT,
    ENCIRCLE_RADIUS,
    FLOCK_SEPARATION_DISTANCE,
    MAX_SPEED,
    SENSE_RADIUS,
    TARGET_STANDOFF_DISTANCE
)


def clamp_vector(v, max_norm=MAX_SPEED):
    norm = np.linalg.norm(v)

    if norm > max_norm:
        return v / norm * max_norm

    return v


def target_position(t):
    return np.array([
        0.8 * np.cos(0.6 * t),
        0.6 * np.sin(0.4 * t)
    ], dtype=float)


def sense_neighbors(robot_id, positions):
    me = positions[robot_id]
    neighbors = []

    for j, other in enumerate(positions):
        if j == robot_id:
            continue

        d = np.linalg.norm(other - me)

        if d <= SENSE_RADIUS:
            neighbors.append((j, other.copy(), d))

    return neighbors


def move_to_goal(me, goal, strength=1.0):
    return strength * (goal - me)


def avoid_neighbors(robot_id, positions, strength=1.8):
    me = positions[robot_id]
    avoid = np.zeros(2)

    for j, other in enumerate(positions):
        if j == robot_id:
            continue

        direction = me - other
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            continue

        if distance < FLOCK_SEPARATION_DISTANCE:
            away = direction / distance
            avoid += strength * (FLOCK_SEPARATION_DISTANCE - distance) * away

    return avoid


def spread_from_neighbors(robot_id, positions, strength=1.2):
    me = positions[robot_id]
    spread = np.zeros(2)

    for j, other in enumerate(positions):
        if j == robot_id:
            continue

        direction = me - other
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            continue

        if distance <= SENSE_RADIUS:
            away = direction / distance
            spread += strength * away / distance

    return spread


def keep_distance_from_target(
    me,
    target,
    desired_distance=TARGET_STANDOFF_DISTANCE,
    strength=1.2
):
    if target is None:
        return np.zeros(2)

    direction = target - me
    distance = np.linalg.norm(direction)

    if distance < 1e-6:
        return np.array([strength, 0.0])

    unit_direction = direction / distance
    distance_error = distance - desired_distance

    return strength * distance_error * unit_direction


def avoid_boundary(me):
    boundary = np.zeros(2)
    margin = 0.7
    strength = 2.0

    right_limit = ARENA_LIMIT - margin
    left_limit = -ARENA_LIMIT + margin
    top_limit = ARENA_LIMIT - margin
    bottom_limit = -ARENA_LIMIT + margin

    if me[0] > right_limit:
        boundary[0] -= strength * (me[0] - right_limit)

    if me[0] < left_limit:
        boundary[0] += strength * (left_limit - me[0])

    if me[1] > top_limit:
        boundary[1] -= strength * (me[1] - top_limit)

    if me[1] < bottom_limit:
        boundary[1] += strength * (bottom_limit - me[1])

    return boundary


def assigned_encircle_point(robot_id, total_robots, target):
    angle = 2.0 * np.pi * robot_id / total_robots

    point = target + ENCIRCLE_RADIUS * np.array([
        np.cos(angle),
        np.sin(angle)
    ])

    return point