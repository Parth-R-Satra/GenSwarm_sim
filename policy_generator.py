def analyze_instruction(instruction):
    text = instruction.lower()

    task_spec = {
        "task": "flock",
        "target_required": False,
        "description": instruction,
        "skills": [],
        "constraints": [
            "avoid_robot_collision",
            "stay_inside_arena"
        ]
    }

    encircle_words = [
        "encircle",
        "surround",
        "circle",
        "around",
        "ring",
        "orbit"
    ]

    flock_words = [
        "flock",
        "group",
        "cluster",
        "together",
        "cohesive"
    ]

    target_words = [
        "target",
        "prey",
        "moving target"
    ]

    has_encircle_intent = any(word in text for word in encircle_words)
    has_flock_intent = any(word in text for word in flock_words)
    has_target = any(word in text for word in target_words)

    if has_encircle_intent or (("follow" in text or "chase" in text) and has_target):
        task_spec["task"] = "encircle"
        task_spec["target_required"] = True
        task_spec["skills"] = [
            "assigned_encircle_point",
            "move_to_goal",
            "avoid_neighbors",
            "avoid_boundary"
        ]
        task_spec["constraints"].append("maintain_radius_around_target")
        return task_spec

    if has_flock_intent:
        task_spec["task"] = "flock"
        task_spec["target_required"] = False
        task_spec["skills"] = [
            "sense_neighbors",
            "move_to_goal",
            "avoid_neighbors",
            "avoid_boundary",
            "velocity_alignment",
            "damping"
        ]
        task_spec["constraints"].append("maintain_local_cohesion")
        return task_spec

    task_spec["task"] = "flock"
    task_spec["skills"] = [
        "sense_neighbors",
        "move_to_goal",
        "avoid_neighbors",
        "avoid_boundary",
        "velocity_alignment",
        "damping"
    ]
    task_spec["constraints"].append("maintain_local_cohesion")

    return task_spec


def generate_policy_code(instruction, task_spec):
    task = task_spec["task"]

    if task == "flock":
        return """
def generated_policy(robot_id, positions, old_velocities, target):
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
"""

    if task == "encircle":
        return """
def generated_policy(robot_id, positions, old_velocities, target):
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
"""

    return """
def generated_policy(robot_id, positions, old_velocities, target):
    return np.zeros(2)
"""