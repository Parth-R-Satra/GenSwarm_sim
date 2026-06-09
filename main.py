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

metrics = {
    "total_encircle_error": 0.0,
    "frames": 0,
    "robot_collisions": 0
}


def encircle_error(positions, target):
    distances = np.linalg.norm(positions - target, axis=1)
    errors = np.abs(distances - ENCIRCLE_RADIUS)
    return np.mean(errors)


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


angles = np.linspace(0, 2 * np.pi, N_ROBOTS, endpoint=False)

positions = np.stack([
    2.7 * np.cos(angles),
    2.7 * np.sin(angles)
], axis=1)

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
    label="Target"
)

circle_line, = ax.plot([], [], linestyle="--", linewidth=1)

ax.legend(loc="upper right")


def update(frame):
    global positions

    t = frame * DT
    target = target_position(t)

    old_positions = positions.copy()
    velocities = np.zeros_like(positions)

    for i in range(N_ROBOTS):
        velocities[i] = encircle_policy(i, old_positions, target)

    positions = positions + velocities * DT

    error = encircle_error(positions, target)
    robot_collisions_now = count_robot_collisions(positions)

    metrics["total_encircle_error"] += error
    metrics["frames"] += 1
    metrics["robot_collisions"] += robot_collisions_now

    robots_plot.set_offsets(positions)
    target_plot.set_offsets([target])

    theta = np.linspace(0, 2 * np.pi, 100)
    circle = target + ENCIRCLE_RADIUS * np.stack([
        np.cos(theta),
        np.sin(theta)
    ], axis=1)

    circle_line.set_data(circle[:, 0], circle[:, 1])

    ax.set_title(
        f"Task: encircle | Step {frame} | "
        f"Encircle error: {error:.3f} | "
        f"Robot now: {robot_collisions_now} | "
        f"Robot total: {metrics['robot_collisions']}"
    )

    return robots_plot, target_plot, circle_line


ani = FuncAnimation(
    fig,
    update,
    frames=STEPS,
    interval=30,
    blit=False
)

plt.show()

average_error = metrics["total_encircle_error"] / max(metrics["frames"], 1)

print("Simulation finished")
print("Task: encircle")
print("Average encircle error:", average_error)
print("Total robot collision count:", metrics["robot_collisions"])