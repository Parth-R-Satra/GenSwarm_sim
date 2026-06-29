def generated_policy(robot_id, positions, old_velocities, target):
    me = positions[robot_id]

    # Calculate the center of mass of all robots for cohesion
    swarm_center = np.mean(positions, axis=0)

    # Cohesion: Move towards the swarm's center
    cohesion_velocity = move_to_goal(me, swarm_center, strength=1.0)

    # Separation: Avoid colliding with neighbors
    separation_velocity = avoid_neighbors(robot_id, positions, strength=1.8)

    # Boundary Avoidance: Stay within the arena
    boundary_velocity = avoid_boundary(me)

    # Combine all desired velocities
    total_velocity = cohesion_velocity + separation_velocity + boundary_velocity

    # Clamp the final velocity to MAX_SPEED
    return clamp_vector(total_velocity, MAX_SPEED)