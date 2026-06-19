import ast
import numpy as np

from config import MAX_SPEED
from skills import (
    assigned_encircle_point,
    avoid_boundary,
    avoid_neighbors,
    clamp_vector,
    move_to_goal,
    sense_neighbors
)


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