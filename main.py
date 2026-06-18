import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N_ROBOTS = 8
DT = 0.05
MAX_SPEED = 0.55
SENSE_RADIUS = 2.2
ENCIRCLE_RADIUS = 1.4
ARENA_LIMIT = 4.0
STEPS = 1200

ROBOT_RADIUS = 0.12
ROBOT_COLLISION_DISTANCE = 2 * ROBOT_RADIUS
FLOCK_SEPARATION_DISTANCE = 0.55

USE_GENERATED_POLICY = True

# New improvement: save final simulation results to a file.
SAVE_RUN_SUMMARY = True
RUN_SUMMARY_FILE = "run_summary.txt"


def choose_task_from_instruction(instruction):
    text = instruction.lower()

    if "flock" in text or "group" in text or "cluster" in text:
        return "flock"

    if "encircle" in text or "surround" in text or "circle" in text:
        return "encircle"

    return "flock"


USER_INSTRUCTION = input("Enter swarm instruction: ")
TASK = choose_task_from_instruction(USER_INSTRUCTION)

print("Chosen task:", TASK)


metrics = {
    "total_encircle_error": 0.0,
    "total_spatial_variance": 0.0,
    "frames": 0,
    "robot_collisions": 0
}


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


def clamp_vector(v, max_norm):
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


def generate_policy_code(instruction, task):
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


def validate_policy_code(policy_code):
    banned_words = [
        "import",
        "open",
        "eval",
        "exec",
        "__",
        "os.",
        "sys.",
        "subprocess",
        "input",
        "while True"
    ]

    for word in banned_words:
        if word in policy_code:
            raise ValueError(f"Unsafe generated code found: {word}")

    tree = ast.parse(policy_code)

    function_names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]

    if "generated_policy" not in function_names:
        raise ValueError("Generated code must define generated_policy().")

    allowed_function_calls = {
        "generated_policy",
        "sense_neighbors",
        "move_to_goal",
        "avoid_neighbors",
        "avoid_boundary",
        "assigned_encircle_point",
        "clamp_vector",
        "len",
        "range"
    }

    allowed_np_calls = {
        "mean",
        "zeros",
        "array"
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in allowed_function_calls:
                    raise ValueError(f"Function call not allowed: {node.func.id}")

            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "np":
                        if node.func.attr not in allowed_np_calls:
                            raise ValueError(
                                f"NumPy call not allowed: np.{node.func.attr}"
                            )

    return True


def compile_generated_policy(policy_code):
    validate_policy_code(policy_code)

    namespace = {
        "__builtins__": {
            "len": len,
            "range": range
        },
        "np": np,
        "MAX_SPEED": MAX_SPEED,
        "sense_neighbors": sense_neighbors,
        "move_to_goal": move_to_goal,
        "avoid_neighbors": avoid_neighbors,
        "avoid_boundary": avoid_boundary,
        "assigned_encircle_point": assigned_encircle_point,
        "clamp_vector": clamp_vector
    }

    exec(policy_code, namespace)

    generated = namespace["generated_policy"]

    dummy_positions = np.array([
        [0.0, 0.0],
        [0.5, 0.1],
        [1.0, 0.0],
        [1.5, 0.1],
        [2.0, 0.0],
        [2.5, 0.1],
        [3.0, 0.0],
        [3.5, 0.1]
    ], dtype=float)

    dummy_velocities = np.zeros_like(dummy_positions)
    dummy_target = np.array([0.0, 0.0], dtype=float)

    test_output = generated(
        0,
        dummy_positions,
        dummy_velocities,
        dummy_target
    )

    if not isinstance(test_output, np.ndarray):
        raise ValueError("generated_policy() must return a NumPy array.")

    if test_output.shape != (2,):
        raise ValueError("generated_policy() must return a 2D vector with shape (2,).")

    if np.any(np.isnan(test_output)):
        raise ValueError("generated_policy() returned NaN values.")

    print("Generated policy passed validation.")

    return generated


policy_code = generate_policy_code(USER_INSTRUCTION, TASK)

print("\nGenerated policy code:")
print(policy_code)

generated_policy = compile_generated_policy(policy_code)


def initial_positions_for_task(task):
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


positions = initial_positions_for_task(TASK)
previous_velocities = np.zeros_like(positions)

fig, ax = plt.subplots(figsize=(7, 7))

ax.set_xlim(-ARENA_LIMIT, ARENA_LIMIT)
ax.set_ylim(-ARENA_LIMIT, ARENA_LIMIT)
ax.set_aspect("equal")
ax.grid(True)

robots_plot = ax.scatter(
    positions[:, 0],
    positions[:, 1],
    s=90,
    label="Robots"
)

target_plot = ax.scatter(
    [0],
    [0],
    s=160,
    marker="x",
    label="Target" if TASK == "encircle" else "_nolegend_"
)

circle_line, = ax.plot([], [], linestyle="--", linewidth=1)

ax.legend(loc="upper right")


def update(frame):
    global positions, previous_velocities

    t = frame * DT

    old_positions = positions.copy()
    old_velocities = previous_velocities.copy()
    velocities = np.zeros_like(positions)

    if TASK == "encircle":
        target = target_position(t)

        for i in range(N_ROBOTS):
            if USE_GENERATED_POLICY:
                velocities[i] = generated_policy(
                    i,
                    old_positions,
                    old_velocities,
                    target
                )
            else:
                velocities[i] = encircle_policy(i, old_positions, target)

    elif TASK == "flock":
        target = None

        for i in range(N_ROBOTS):
            if USE_GENERATED_POLICY:
                velocities[i] = generated_policy(
                    i,
                    old_positions,
                    old_velocities,
                    target
                )
            else:
                velocities[i] = flock_policy(i, old_positions, old_velocities)

    else:
        target = None

    positions = positions + velocities * DT
    previous_velocities = velocities.copy()

    robot_collisions_now = count_robot_collisions(positions)

    metrics["frames"] += 1
    metrics["robot_collisions"] += robot_collisions_now

    robots_plot.set_offsets(positions)

    if TASK == "encircle":
        target_plot.set_visible(True)
        target_plot.set_offsets([target])

        theta = np.linspace(0, 2 * np.pi, 100)
        circle = target + ENCIRCLE_RADIUS * np.stack([
            np.cos(theta),
            np.sin(theta)
        ], axis=1)

        circle_line.set_data(circle[:, 0], circle[:, 1])

        error = encircle_error(positions, target)
        metrics["total_encircle_error"] += error

        title_metric = f"Encircle error: {error:.3f}"

    elif TASK == "flock":
        target_plot.set_visible(False)
        circle_line.set_data([], [])

        variance = spatial_variance(positions)
        metrics["total_spatial_variance"] += variance

        title_metric = f"Spatial variance: {variance:.3f}"

    else:
        target_plot.set_visible(False)
        circle_line.set_data([], [])
        title_metric = "Unknown task"

    policy_type = "generated" if USE_GENERATED_POLICY else "manual"

    ax.set_title(
        f"Task: {TASK} | Policy: {policy_type} | Step {frame} | "
        f"{title_metric} | Robot now: {robot_collisions_now} | "
        f"Robot total: {metrics['robot_collisions']}"
    )

    return robots_plot, target_plot, circle_line


def save_run_summary():
    lines = []

    lines.append("GenSwarm-style Simulation Run Summary")
    lines.append("------------------------------------")
    lines.append(f"Instruction: {USER_INSTRUCTION}")
    lines.append(f"Chosen task: {TASK}")
    lines.append(f"Policy mode: {'generated' if USE_GENERATED_POLICY else 'manual'}")
    lines.append(f"Frames executed: {metrics['frames']}")
    lines.append(f"Total robot collisions: {metrics['robot_collisions']}")

    if TASK == "encircle":
        average_error = metrics["total_encircle_error"] / max(metrics["frames"], 1)
        lines.append(f"Average encircle error: {average_error}")

    if TASK == "flock":
        average_variance = metrics["total_spatial_variance"] / max(metrics["frames"], 1)
        lines.append(f"Average spatial variance: {average_variance}")

    with open(RUN_SUMMARY_FILE, "w") as file:
        file.write("\n".join(lines))

    print(f"Run summary saved to {RUN_SUMMARY_FILE}")


ani = FuncAnimation(
    fig,
    update,
    frames=STEPS,
    interval=30,
    blit=False
)

plt.show()

print("Simulation finished")
print("Instruction:", USER_INSTRUCTION)
print("Task:", TASK)
print("Policy mode:", "generated" if USE_GENERATED_POLICY else "manual")

if TASK == "encircle":
    average_error = metrics["total_encircle_error"] / max(metrics["frames"], 1)
    print("Average encircle error:", average_error)

if TASK == "flock":
    average_variance = metrics["total_spatial_variance"] / max(metrics["frames"], 1)
    print("Average spatial variance:", average_variance)

print("Total robot collision count:", metrics["robot_collisions"])

if SAVE_RUN_SUMMARY:
    save_run_summary()