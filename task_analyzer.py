import re
from difflib import get_close_matches

from config import (
    TASK_ANALYZER_MAX_RETRIES,
    USE_LLM_TASK_ANALYZER
)

from llm_policy_generator import generate_task_spec_with_llm


ALLOWED_TASKS = {
    "flock",
    "encircle",
    "pursuit",
    "dispersion"
}

ALLOWED_SKILLS = {
    "sense_neighbors",
    "move_to_goal",
    "avoid_neighbors",
    "spread_from_neighbors",
    "keep_distance_from_target",
    "avoid_boundary",
    "assigned_encircle_point",
    "velocity_alignment",
    "damping"
}


def remove_duplicates(items):
    cleaned = []

    for item in items:
        if item not in cleaned:
            cleaned.append(item)

    return cleaned


def fallback_task_spec(task, instruction, confidence=1.0):
    """
    Used only if LLM task analysis fails.
    Main LLM path preserves Gemini-selected skills.
    """

    base_spec = {
        "task": task,
        "target_required": False,
        "description": instruction,
        "confidence": confidence,
        "skills": []
    }

    if task == "encircle":
        base_spec["target_required"] = True
        base_spec["skills"] = [
            "assigned_encircle_point",
            "move_to_goal",
            "avoid_neighbors",
            "avoid_boundary"
        ]
        return base_spec

    if task == "pursuit":
        base_spec["target_required"] = True
        base_spec["skills"] = [
            "keep_distance_from_target",
            "avoid_neighbors",
            "avoid_boundary",
            "damping"
        ]
        return base_spec

    if task == "dispersion":
        base_spec["target_required"] = False
        base_spec["skills"] = [
            "spread_from_neighbors",
            "avoid_neighbors",
            "avoid_boundary",
            "damping"
        ]
        return base_spec

    base_spec["task"] = "flock"
    base_spec["target_required"] = False
    base_spec["skills"] = [
        "sense_neighbors",
        "move_to_goal",
        "avoid_neighbors",
        "avoid_boundary",
        "velocity_alignment",
        "damping"
    ]

    return base_spec


def normalize_llm_task_spec(task_spec, instruction):
    return {
        "task": task_spec["task"],
        "target_required": task_spec["target_required"],
        "description": task_spec.get("description", instruction),
        "confidence": task_spec.get("confidence", 0.8),
        "skills": remove_duplicates(task_spec.get("skills", []))
    }


def words_from_text(text):
    return re.findall(r"[a-zA-Z]+", text.lower())


def has_keyword_or_close_match(text, keywords, cutoff=0.78):
    text = text.lower()
    words = words_from_text(text)

    for keyword in keywords:
        if keyword in text:
            return True

    single_word_keywords = [
        keyword
        for keyword in keywords
        if len(keyword.split()) == 1
    ]

    for word in words:
        close_matches = get_close_matches(
            word,
            single_word_keywords,
            n=1,
            cutoff=cutoff
        )

        if len(close_matches) > 0:
            return True

    return False


def fuzzy_task_analysis(instruction):
    text = instruction.lower()

    encircle_words = [
        "encircle",
        "surround",
        "circle",
        "around",
        "ring",
        "orbit"
    ]

    pursuit_words = [
        "pursue",
        "pursuit",
        "follow",
        "chase",
        "track"
    ]

    dispersion_words = [
        "disperse",
        "dispersion",
        "dispersal",
        "spread",
        "spread out",
        "move apart",
        "separate",
        "scatter"
    ]

    flock_words = [
        "flock",
        "group",
        "cluster",
        "together",
        "cohesive",
        "come close",
        "close together"
    ]

    target_words = [
        "target",
        "prey",
        "moving target"
    ]

    has_encircle_intent = has_keyword_or_close_match(text, encircle_words)
    has_pursuit_intent = has_keyword_or_close_match(text, pursuit_words)
    has_dispersion_intent = has_keyword_or_close_match(text, dispersion_words)
    has_flock_intent = has_keyword_or_close_match(text, flock_words)
    has_target = has_keyword_or_close_match(text, target_words)

    if has_dispersion_intent:
        return fallback_task_spec(
            "dispersion",
            instruction,
            confidence=0.85
        )

    if has_encircle_intent:
        return fallback_task_spec(
            "encircle",
            instruction,
            confidence=0.85
        )

    if has_pursuit_intent and has_target:
        return fallback_task_spec(
            "pursuit",
            instruction,
            confidence=0.85
        )

    if has_flock_intent:
        return fallback_task_spec(
            "flock",
            instruction,
            confidence=0.85
        )

    return fallback_task_spec(
        "flock",
        instruction,
        confidence=0.5
    )


def validate_llm_task_spec(task_spec):
    if not isinstance(task_spec, dict):
        raise ValueError("Task spec must be a dictionary.")

    task = task_spec.get("task")

    if task not in ALLOWED_TASKS:
        raise ValueError(f"Invalid task from LLM: {task}")

    target_required = task_spec.get("target_required")

    if not isinstance(target_required, bool):
        raise ValueError("target_required must be boolean.")

    confidence = task_spec.get("confidence", 0.0)

    if not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number.")

    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be between 0 and 1.")

    skills = task_spec.get("skills", [])

    if not isinstance(skills, list):
        raise ValueError("skills must be a list.")

    if len(skills) == 0:
        raise ValueError("LLM must select at least one skill.")

    for skill in skills:
        if skill not in ALLOWED_SKILLS:
            raise ValueError(f"Invalid skill from LLM: {skill}")

    return True


def llm_task_analysis(instruction):
    validation_error = None

    for attempt in range(TASK_ANALYZER_MAX_RETRIES):
        try:
            task_spec = generate_task_spec_with_llm(
                instruction,
                validation_error=validation_error
            )

            validate_llm_task_spec(task_spec)

            return normalize_llm_task_spec(
                task_spec,
                instruction
            )

        except Exception as error:
            validation_error = str(error)
            print("LLM task analysis failed:")
            print(validation_error)

    print("Falling back to fuzzy task analysis.")

    return fuzzy_task_analysis(instruction)


def analyze_instruction(instruction):
    if USE_LLM_TASK_ANALYZER:
        return llm_task_analysis(instruction)

    return fuzzy_task_analysis(instruction)