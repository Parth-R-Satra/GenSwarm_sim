import numpy as np

from config import MAX_SPEED, N_ROBOTS
from skills import (
    assigned_encircle_point,
    avoid_boundary,
    avoid_neighbors,
    clamp_vector,
    keep_distance_from_target,
    move_to_goal,
    sense_neighbors,
    spread_from_neighbors
)


def encircle_policy(robot_id, positions, target):
    me = positions[robot_id]

    goal = assigned_encircle_point(
        robot_id,
        len(positions),
        target
    )

    velocity = (
        move_to_goal(me, goal, strength=1.8)
        + avoid_neighbors(robot_id, positions, strength=0.8)
        + avoid_boundary(me)
    )

    return clamp_vector(velocity, MAX_SPEED)


def pursuit_policy(robot_id, positions, old_velocities, target):
    me = positions[robot_id]

    target_follow = keep_distance_from_target(
        me,
        target,
        desired_distance=0.7,
        strength=1.2
    )

    separation = avoid_neighbors(
        robot_id,
        positions,
        strength=1.8
    )

    damping = -0.25 * old_velocities[robot_id]
    boundary = avoid_boundary(me)

    velocity = (
        target_follow
        + separation
        + damping
        + boundary
    )

    return clamp_vector(velocity, MAX_SPEED)


def flock_policy(robot_id, positions, old_velocities):
    me = positions[robot_id]
    neighbors = sense_neighbors(robot_id, positions)

    if len(neighbors) > 0:
        neighbor_ids = [item[0] for item in neighbors]

        neighbor_positions = positions[neighbor_ids]
        neighbor_velocities = old_velocities[neighbor_ids]

        local_center = np.mean(neighbor_positions, axis=0)

        cohesion = move_to_goal(
            me,
            local_center,
            strength=0.55
        )

        average_neighbor_velocity = np.mean(neighbor_velocities, axis=0)

        alignment = (
            average_neighbor_velocity - old_velocities[robot_id]
        ) * 0.55

    else:
        cohesion = np.zeros(2)
        alignment = np.zeros(2)

    separation = avoid_neighbors(
        robot_id,
        positions,
        strength=2.4
    )

    damping = -0.35 * old_velocities[robot_id]
    boundary = avoid_boundary(me)

    velocity = (
        cohesion
        + alignment
        + separation
        + damping
        + boundary
    )

    return clamp_vector(velocity, MAX_SPEED)


def dispersion_policy(robot_id, positions, old_velocities):
    me = positions[robot_id]

    spread = spread_from_neighbors(
        robot_id,
        positions,
        strength=1.2
    )

    separation = avoid_neighbors(
        robot_id,
        positions,
        strength=1.5
    )

    damping = -0.3 * old_velocities[robot_id]
    boundary = avoid_boundary(me)

    velocity = (
        spread
        + separation
        + damping
        + boundary
    )

    return clamp_vector(velocity, MAX_SPEED)


def initial_positions_for_task(task):
    if task == "dispersion":
        return np.array([
            [-0.6,  0.4],
            [-0.3,  0.2],
            [ 0.0,  0.5],
            [ 0.3,  0.1],
            [ 0.6,  0.3],
            [-0.4, -0.2],
            [ 0.1, -0.4],
            [ 0.5, -0.3]
        ], dtype=float)

    if task == "flock":
        return np.array([
            [-3.1,  2.7],
            [-2.4,  1.8],
            [-1.6,  3.0],
            [-0.8,  1.5],
            [ 0.2,  2.6],
            [ 1.2,  1.7],
            [ 2.2,  2.9],
            [ 3.0,  1.6]
        ], dtype=float)

    angles = np.linspace(0, 2 * np.pi, N_ROBOTS, endpoint=False)

    return np.stack([
        2.7 * np.cos(angles),
        2.7 * np.sin(angles)
    ], axis=1)