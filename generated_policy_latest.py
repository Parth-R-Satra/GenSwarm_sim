
def generated_policy(robot_id, positions, old_velocities, target):
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
