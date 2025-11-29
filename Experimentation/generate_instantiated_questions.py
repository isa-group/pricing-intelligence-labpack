import argparse
import json
import re
from copy import deepcopy
from pathlib import Path

PLACEHOLDER_RE = re.compile(r"\{\{([^}]+)\}\}")


def replace_placeholders_in_text(text: str, placeholder_values: dict) -> str:
    """Sustituye {{PLACEHOLDER}} en un string por el valor correspondiente."""

    def repl(match: re.Match) -> str:
        name = match.group(1)
        if name not in placeholder_values:
            raise KeyError(f"Missing placeholder value for '{name}' in question text")
        return placeholder_values[name]

    return PLACEHOLDER_RE.sub(repl, text)


def instantiate_plan_with_placeholders(plan_template: dict, placeholder_values: dict) -> dict:
    """
    Aplica sustitución de placeholders a todo el árbol del plan: cualquier string que sea
    exactamente '{{NOMBRE}}' se sustituye por placeholder_values['NOMBRE'].
    """

    def rec(node):
        if isinstance(node, dict):
            return {k: rec(v) for k, v in node.items()}
        elif isinstance(node, list):
            return [rec(v) for v in node]
        elif isinstance(node, str):
            m = PLACEHOLDER_RE.fullmatch(node)
            if m:
                name = m.group(1)
                if name not in placeholder_values:
                    raise KeyError(f"Missing placeholder value for '{name}' in plan")
                return placeholder_values[name]
            return node
        else:
            return node

    return rec(plan_template)


def apply_plan_overrides(plan: dict, overrides: dict | None) -> dict:
    """
    Aplica los overrides mínimos almacenados en el diccionario sobre un plan ya
    instanciado con placeholders.
    """
    if not overrides:
        return plan

    plan = deepcopy(plan)

    # Caso especial: sobreescribir toda la lista de acciones
    if "actions_full" in overrides:
        plan["actions"] = overrides["actions_full"]

    # Caso general: overrides por índice de action
    if "actions" in overrides:
        override_actions = overrides["actions"]
        for idx, override in enumerate(override_actions):
            if override is not None:
                # Se sustituye la acción completa en ese índice
                plan["actions"][idx] = override

    # Otros overrides a nivel top (use_pricing2yaml_spec, etc.)
    for key, value in overrides.items():
        if key in ("actions", "actions_full"):
            continue
        plan[key] = value

    return plan


def generate_instantiated_questions(templates: list[dict], spec: dict) -> list[dict]:
    """
    Genera la lista completa de objetos InstantiatedQuestion a partir de:
    - templates: lista leída de question_action_templates.json
    - spec: diccionario leido de instantiation_spec.json
    """
    instances = spec["instances"]
    out: list[dict] = []

    for inst_spec in instances:
        template_index = inst_spec["template_index"]
        template_entry = templates[template_index]

        placeholder_values = inst_spec["placeholder_values"]
        question_override = inst_spec.get("question_override")
        plan_overrides = inst_spec.get("plan_overrides")
        pricing_paths = inst_spec["pricing_paths"]

        template_question = template_entry["question"]

        # Pregunta final
        if question_override is not None:
            question = question_override
        else:
            question = replace_placeholders_in_text(template_question, placeholder_values)

        # Plan: plantilla + placeholders + overrides
        base_plan = instantiate_plan_with_placeholders(template_entry["plan"], placeholder_values)
        final_plan = apply_plan_overrides(base_plan, plan_overrides)

        out.append(
            {
                "template": template_question,
                "question": question,
                "plan": final_plan,
                "pricing_paths": pricing_paths,
            }
        )

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate InstantiatedQuestions.json from templates + instantiation spec"
    )
    parser.add_argument(
        "--templates",
        required=True,
        help="Path to question_action_templates.json",
    )
    parser.add_argument(
        "--spec",
        required=True,
        help="Path to instantiation_spec.json (dictionary with placeholder values and overrides)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write InstantiatedQuestions.json",
    )
    parser.add_argument(
        "--expected",
        help="Optional path to an existing instantiated_questions.json to verify exact equality",
    )

    args = parser.parse_args()

    templates_path = Path(args.templates)
    spec_path = Path(args.spec)
    output_path = Path(args.output)

    templates = json.loads(templates_path.read_text())
    spec = json.loads(spec_path.read_text())

    instantiated_questions = generate_instantiated_questions(templates, spec)

    # Escribimos la salida
    output_path.write_text(json.dumps(instantiated_questions, indent=2))



if __name__ == "__main__":
    main()
