def generated_policy(robot_id, positions, old_velocities, target):
    velocity = np.zeros(2)
    me = positions[robot_id]

    # Core dispersion behavior: spread from neighbors
    velocity += spread_from_neighbors(robot_id, positions, strength=1.5)

    # Avoid immediate collisions with neighbors
    velocity += avoid_neighbors(robot_id, positions, strength=1.8)

    # Stay inside the arena boundaries
    velocity += avoid_boundary(me)

    # Damping to smooth movement and prevent oscillations
    # Incorporate a fraction of the old velocity
    damping_factor = 0.5
    velocity += old_velocities[robot_id] * damping_factor

    return clamp_vector(velocity, MAX_SPEED)