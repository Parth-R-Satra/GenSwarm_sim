import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N_ROBOTS = 8
DT = 0.05
MAX_SPEED = 0.55
ENCIRCLE_RADIUS = 1.4
ARENA_LIMIT = 4.0
STEPS = 1200


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


def encircle_policy(robot_id, positions, target):
    me = positions[robot_id]

    angle = 2.0 * np.pi * robot_id / len(positions)

    goal = target + ENCIRCLE_RADIUS * np.array([
        np.cos(angle),
        np.sin(angle)
    ])

    velocity = 1.8 * (goal - me)

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

    robots_plot.set_offsets(positions)
    target_plot.set_offsets([target])

    theta = np.linspace(0, 2 * np.pi, 100)
    circle = target + ENCIRCLE_RADIUS * np.stack([
        np.cos(theta),
        np.sin(theta)
    ], axis=1)

    circle_line.set_data(circle[:, 0], circle[:, 1])

    ax.set_title(f"Basic Encircling Simulation | Step {frame}")

    return robots_plot, target_plot, circle_line


ani = FuncAnimation(
    fig,
    update,
    frames=STEPS,
    interval=30,
    blit=False
)

plt.show()