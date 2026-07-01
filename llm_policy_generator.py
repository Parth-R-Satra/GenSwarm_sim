import json

from google import genai

from config import (
    ENCIRCLE_RADIUS,
    LLM_MODEL,
    TARGET_STANDOFF_DISTANCE
)


client = genai.Client()


def clean_llm_code(text):
    text = text.strip()

    if text.startswith("```python"):
        text = text.replace("```python", "", 1).strip()

    if text.startswith("```"):
        text = text.replace("```", "", 1).strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


def clean_llm_json(text):
    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()

    if text.startswith("```"):
        text = text.replace("```", "", 1).strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


def build_task_analysis_prompt(instruction, validation_error=None):
    prompt = f"""
You are analyzing a natural-language instruction for a 2D multi-robot swarm simulator.

User instruction:
{instruction}

Choose exactly one task from this list:
- flock
- encircle
- pursuit
- dispersion

Task meanings:
- flock: robots move together as a cohesive group.
- encircle: robots form a ring around a target.
- pursuit: robots follow or chase a moving target while maintaining a safe standoff distance.
- dispersion: robots spread out or move away from each other.

Important interpretation rules:
- "around the target", "surround target", "orbit target", "circle target" usually means encircle.
- "follow target", "chase target", "track target", "pursue target" usually means pursuit.
- "spread out", "move apart", "scatter", "disperse", even with typos like "dispersw", usually means dispersion.
- "come close", "group together", "cluster", "flock", "stay together" usually means flock.

You must choose the relevant skills yourself from the allowed skills list.
Do not leave skills empty.
Do not invent new skills.
Choose only the skills that are relevant to the instruction.
You are not required to select every possible skill for a task.

You must decide by yourself whether the task requires a target.
Set target_required to true only if the instruction requires a target-based behavior.
Set target_required to false if the task can be performed without a target.

Allowed skills:
- sense_neighbors
- move_to_goal
- avoid_neighbors
- spread_from_neighbors
- keep_distance_from_target
- avoid_boundary
- assigned_encircle_point
- velocity_alignment
- damping

Skill meanings:
- sense_neighbors: detect nearby robots.
- move_to_goal: move toward a point or local goal.
- avoid_neighbors: avoid robot-robot collisions.
- spread_from_neighbors: actively move away from neighbors.
- keep_distance_from_target: follow a target while maintaining safe distance.
- avoid_boundary: stay inside the arena.
- assigned_encircle_point: assign each robot a point on a ring around the target.
- velocity_alignment: match velocity with nearby robots.
- damping: reduce oscillation using negative velocity feedback.

Return ONLY valid JSON. Do not write markdown. Do not explain.

JSON format:
{{
    "task": "flock",
    "target_required": false,
    "description": "{instruction}",
    "confidence": 0.0,
    "skills": [
        "sense_neighbors",
        "avoid_neighbors",
        "avoid_boundary"
    ]
}}
"""

    if validation_error is not None:
        prompt += f"""

The previous JSON failed validation with this error:
{validation_error}

Return corrected valid JSON only.
"""

    return prompt


def generate_task_spec_with_llm(instruction, validation_error=None):
    prompt = build_task_analysis_prompt(
        instruction,
        validation_error=validation_error
    )

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    cleaned_text = clean_llm_json(response.text)

    return json.loads(cleaned_text)


def build_policy_prompt(instruction, task_spec, validation_error=None):
    prompt = f"""
You are generating Python code for a 2D multi-robot swarm simulator.

User instruction:
{instruction}

Structured task analysis selected by the task analyzer:
{task_spec}

Write ONLY Python code. Do not write markdown. Do not explain.

You must define exactly this function:

def generated_policy(robot_id, positions, old_velocities, target):
    ...

The function must return a NumPy array with shape (2,).

Available variables:
- robot_id: current robot index
- positions: NumPy array of robot positions, shape (N, 2)
- old_velocities: NumPy array of previous velocities, shape (N, 2)
- target: NumPy array of target position, or None

Available constants:
- MAX_SPEED
- ENCIRCLE_RADIUS = {ENCIRCLE_RADIUS}
- TARGET_STANDOFF_DISTANCE = {TARGET_STANDOFF_DISTANCE}

Allowed helper functions:
- sense_neighbors(robot_id, positions)
- move_to_goal(me, goal, strength=1.0)
- avoid_neighbors(robot_id, positions, strength=1.8)
- spread_from_neighbors(robot_id, positions, strength=1.2)
- keep_distance_from_target(me, target, desired_distance, strength=1.2)
- avoid_boundary(me)
- assigned_encircle_point(robot_id, total_robots, target)
- clamp_vector(v, MAX_SPEED)

Allowed NumPy calls:
- np.mean(...)
- np.zeros(...)
- np.array(...)

Helper function API details:
- sense_neighbors(robot_id, positions) returns a Python list of tuples.
- Each tuple has this format:
  (neighbor_id, neighbor_position, distance)

Correct sense_neighbors usage:
neighbors = sense_neighbors(robot_id, positions)
if len(neighbors) > 0:
    neighbor_ids = [item[0] for item in neighbors]
    neighbor_positions = positions[neighbor_ids]
    neighbor_velocities = old_velocities[neighbor_ids]

Incorrect sense_neighbors usage:
- Do not unpack sense_neighbors directly into three variables.
- Do not treat sense_neighbors output as a NumPy mask.
- Do not use np.sum on sense_neighbors output.
- Do not use .size on sense_neighbors output.

Important:
- Use only the selected skills from the structured task analysis.
- Do not use swarm helper functions whose corresponding skill was not selected.
- You may always use clamp_vector(...) for final velocity limiting.
- If target_required is false, do not depend on target being available.
- If target_required is true, use the target meaningfully.

Skill semantics:
- damping means negative feedback against current velocity.
  Correct: damping = -0.3 * old_velocities[robot_id]
  Wrong: velocity += 0.5 * old_velocities[robot_id]
- velocity_alignment means matching neighbor velocity, usually:
  average_neighbor_velocity - old_velocities[robot_id]
- avoid_neighbors means robot-robot collision avoidance.
- avoid_boundary means staying inside the arena.
- spread_from_neighbors means actively increasing spacing from neighbors.
- keep_distance_from_target means following a target while maintaining standoff distance.
- assigned_encircle_point means assigning each robot a point on a ring around the target.

Rules:
- Do not import anything.
- Do not use open, eval, exec, input, os, sys, subprocess, or file access.
- Do not use while loops.
- Do not mutate global variables.
- Always clamp the final velocity using clamp_vector(velocity, MAX_SPEED).
- Return only the function code.
"""

    if validation_error is not None:
        prompt += f"""

The previous generated code failed validation with this error:
{validation_error}

Fix the code and follow the selected skills strictly.
"""

    return prompt


def generate_policy_code_with_llm(instruction, task_spec, validation_error=None):
    prompt = build_policy_prompt(
        instruction,
        task_spec,
        validation_error=validation_error
    )

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    return clean_llm_code(response.text)