import ast
import numpy as np

from config import (
    ENCIRCLE_RADIUS,
    MAX_SPEED,
    TARGET_STANDOFF_DISTANCE
)

from skills import (
    assigned_encircle_point,
    avoid_boundary,
    avoid_neighbors,
    clamp_vector,
    keep_distance_from_target,
    move_to_goal,
    sense_neighbors,
    spread_from_neighbors
)


ALLOWED_FUNCTION_CALLS = {
    "generated_policy",
    "sense_neighbors",
    "move_to_goal",
    "avoid_neighbors",
    "spread_from_neighbors",
    "keep_distance_from_target",
    "avoid_boundary",
    "assigned_encircle_point",
    "clamp_vector",
    "len",
    "range"
}


ALLOWED_NUMPY_CALLS = {
    "mean",
    "zeros",
    "array"
}


BANNED_FUNCTION_CALLS = {
    "open",
    "eval",
    "exec",
    "input",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr"
}


BANNED_NAMES = {
    "os",
    "sys",
    "subprocess",
    "pathlib",
    "shutil",
    "socket",
    "requests"
}


def validate_policy_code(policy_code, task_spec=None):
    tree = ast.parse(policy_code)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Import statements are not allowed.")

        if isinstance(node, ast.While):
            raise ValueError("While loops are not allowed.")

        if isinstance(node, ast.Name):
            if node.id in BANNED_NAMES:
                raise ValueError(f"Unsafe name used: {node.id}")

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                function_name = node.func.id

                if function_name in BANNED_FUNCTION_CALLS:
                    raise ValueError(f"Unsafe function call found: {function_name}")

                if function_name not in ALLOWED_FUNCTION_CALLS:
                    raise ValueError(f"Function call not allowed: {function_name}")

            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    object_name = node.func.value.id
                    attribute_name = node.func.attr

                    if object_name == "np":
                        if attribute_name not in ALLOWED_NUMPY_CALLS:
                            raise ValueError(
                                f"NumPy call not allowed: np.{attribute_name}"
                            )
                    else:
                        raise ValueError(
                            "Attribute calls are not allowed except allowed NumPy calls."
                        )
                else:
                    raise ValueError(
                        "Nested attribute calls are not allowed."
                    )

    function_names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]

    if "generated_policy" not in function_names:
        raise ValueError("Generated code must define generated_policy().")

    return True


def compile_generated_policy(policy_code, task_spec=None):
    validate_policy_code(policy_code, task_spec)

    namespace = {
        "__builtins__": {
            "len": len,
            "range": range
        },
        "np": np,
        "MAX_SPEED": MAX_SPEED,
        "ENCIRCLE_RADIUS": ENCIRCLE_RADIUS,
        "TARGET_STANDOFF_DISTANCE": TARGET_STANDOFF_DISTANCE,
        "sense_neighbors": sense_neighbors,
        "move_to_goal": move_to_goal,
        "avoid_neighbors": avoid_neighbors,
        "spread_from_neighbors": spread_from_neighbors,
        "keep_distance_from_target": keep_distance_from_target,
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

    if task_spec is not None and task_spec.get("target_required") is False:
        dummy_target = None
    else:
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