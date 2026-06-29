import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from config import (
    ARENA_LIMIT,
    DT,
    ENCIRCLE_RADIUS,
    GENERATED_POLICY_FILE,
    LLM_MAX_RETRIES,
    N_ROBOTS,
    RUN_SUMMARY_FILE,
    SAVE_GENERATED_POLICY,
    SAVE_RUN_SUMMARY,
    STEPS,
    USE_GENERATED_POLICY,
    USE_LLM_GENERATOR
)

from metrics import (
    count_robot_collisions,
    encircle_error,
    spatial_variance
)

from policies import (
    encircle_policy,
    flock_policy,
    initial_positions_for_task
)

from policy_generator import (
    analyze_instruction,
    generate_policy_code as generate_template_policy_code
)

from llm_policy_generator import generate_policy_code_with_llm

from skills import target_position
from validator import compile_generated_policy


USER_INSTRUCTION = input("Enter swarm instruction: ")

TASK_SPEC = analyze_instruction(USER_INSTRUCTION)
TASK = TASK_SPEC["task"]

print("Task analysis:")
print(TASK_SPEC)
print("Chosen task:", TASK)

def save_generated_policy(policy_code):
    with open(GENERATED_POLICY_FILE, "w") as file:
        file.write(policy_code)

    print(f"Generated policy saved to {GENERATED_POLICY_FILE}")


def build_and_validate_policy():
    validation_error = None

    if USE_LLM_GENERATOR:
        for attempt in range(LLM_MAX_RETRIES):
            print(f"\nGenerating policy using Gemini, attempt {attempt + 1}...")

            policy_code = generate_policy_code_with_llm(
                USER_INSTRUCTION,
                TASK_SPEC,
                validation_error=validation_error
            )

            print("\nLLM generated policy code:")
            print(policy_code)

            try:
                generated = compile_generated_policy(policy_code)

                if SAVE_GENERATED_POLICY:
                    save_generated_policy(policy_code)

                return policy_code, generated

            except Exception as error:
                validation_error = str(error)
                print("Generated policy failed validation:")
                print(validation_error)

        print("\nGemini generation failed. Falling back to template policy.")

    policy_code = generate_template_policy_code(USER_INSTRUCTION, TASK_SPEC)

    print("\nTemplate policy code:")
    print(policy_code)

    generated = compile_generated_policy(policy_code)

    if SAVE_GENERATED_POLICY:
        save_generated_policy(policy_code)

    return policy_code, generated


policy_code, generated_policy = build_and_validate_policy()

metrics = {
    "total_encircle_error": 0.0,
    "total_spatial_variance": 0.0,
    "frames": 0,
    "robot_collisions": 0
}

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
    lines.append(f"Target required: {TASK_SPEC['target_required']}")
    lines.append(f"Skills: {', '.join(TASK_SPEC['skills'])}")
    lines.append(f"Constraints: {', '.join(TASK_SPEC['constraints'])}")
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
print("Target required:", TASK_SPEC["target_required"])
print("Skills:", TASK_SPEC["skills"])
print("Constraints:", TASK_SPEC["constraints"])
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