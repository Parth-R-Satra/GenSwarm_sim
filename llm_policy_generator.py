from google import genai

from config import LLM_MODEL


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


def build_policy_prompt(instruction, task_spec, validation_error=None):
    prompt = f"""
You are generating Python code for a 2D multi-robot swarm simulator.

User instruction:
{instruction}

Structured task analysis:
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

Allowed helper functions:
- sense_neighbors(robot_id, positions)
- move_to_goal(me, goal, strength=1.0)
- avoid_neighbors(robot_id, positions, strength=1.8)
- avoid_boundary(me)
- assigned_encircle_point(robot_id, total_robots, target)
- clamp_vector(v, MAX_SPEED)

Allowed NumPy calls:
- np.mean(...)
- np.zeros(...)
- np.array(...)

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

Fix the code and follow the allowed function list strictly.
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