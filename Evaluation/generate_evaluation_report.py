#!/usr/bin/env python3
"""Genera un JSON de evaluación (estructura similar a `evaluation_report.json`) a
partir de un `experiment_results.json`.

Uso:
  python3 generate_evaluation_report.py --input data/experiment_results.json --output_dir data --outfile evaluation_report_generated.json

La métrica de estructura usa la formulación proporcionada (action-level y parameter-level),
con lambda=0.5. La métrica de contenido es "accuracy" según la especificación del usuario.
"""
import argparse
import json
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def safe_get(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def compute_iqr(values: List[float]) -> float:
    """Compute interquartile range (Q3 - Q1). Return 0.0 if not enough data."""
    if not values or len(values) < 2:
        return 0.0
    try:
        qs = statistics.quantiles(values, n=4)
        # qs -> [Q1, Q2(median), Q3]
        return float(qs[2] - qs[0])
    except Exception:
        # fallback: use median splits
        try:
            m = statistics.median(values)
            lower = [v for v in values if v <= m]
            upper = [v for v in values if v >= m]
            q1 = statistics.median(lower) if lower else m
            q3 = statistics.median(upper) if upper else m
            return float(q3 - q1)
        except Exception:
            return 0.0


def flatten_params(prefix: str, obj: Any, mapping: Dict[str, Any], keys: Set[str]):
    """Recursivamente aplanar parámetros y llenar mapping y keys.
    prefix: camino actual (sin terminar con '.')
    obj: valor a procesar
    mapping: dict path -> value (no JSON stringified)
    keys: conjunto de paths (strings)"""
    # Special handling for certain list-like parameters
    tail = prefix.split(".")[-1]
    if isinstance(obj, dict):
        for k, v in obj.items():
            newp = f"{prefix}.{k}" if prefix else k
            flatten_params(newp, v, mapping, keys)
    elif isinstance(obj, list):
        # features: convert list of strings -> "A;B" (alphabetical, no spaces)
        if tail == "features":
            # keep only string elements
            strs = [str(x) for x in obj if isinstance(x, (str, int, float))]
            strs = sorted(strs)
            combined = ";".join(strs)
            keys.add(prefix)
            mapping[prefix] = combined
            return

        # usageLimits: list of dicts -> "(key,val);(key2,val2)" sorted by key
        if tail == "usageLimits":
            combined_map = {}
            for item in obj:
                if isinstance(item, dict):
                    for k, v in item.items():
                        combined_map[str(k)] = v
            if combined_map:
                parts = []
                for k in sorted(combined_map.keys()):
                    parts.append(f"({k},{combined_map[k]})")
                combined = ";".join(parts)
                keys.add(prefix)
                mapping[prefix] = combined
                return

        # Default: enumerate and flatten each element (keep indices)
        for i, v in enumerate(obj):
            newp = f"{prefix}[{i}]"
            flatten_params(newp, v, mapping, keys)
    else:
        keys.add(prefix)
        mapping[prefix] = obj


def extract_actions(raw_actions: Any) -> List[Dict]:
    """Extrae la lista de acciones desde el objeto `plan`.
    Devuelve lista de dicts: {'name': str, 'param_keys': set(str), 'param_values': dict}
    """
    
    actions = []

    for a in raw_actions:
        if not isinstance(a, dict):
            continue
        name = a.get("name") or a.get("action") or ""
        # Aplanar parámetros: todos los campos excepto 'name' y 'solver'
        param_mapping = {}
        param_keys = set()
        for k, v in a.items():
            if k in ("name", "solver"):
                continue
            # flatenizamos 'filters' y cualquier otro hijo
            if isinstance(v, (dict, list)):
                flatten_params(k, v, param_mapping, param_keys)
            else:
                param_keys.add(k)
                param_mapping[k] = v

        actions.append({
            "name": name,
            "param_keys": param_keys,
            "param_values": param_mapping,
        })

    return actions


def compute_structure_metrics(g_actions: List[Dict], h_actions: List[Dict], lam: float = 0.5) -> Dict[str, float]:
    """Calcula P^{act}, R^{act}, P^{par}, R^{par}, h_p, h_r, h_f1 para un único par (G,H)."""
    A_G = {a.get("name") for a in g_actions if a.get("name")}
    A_H = {a.get("name") for a in h_actions if a.get("name")}

    # action-level
    inter_actions = A_G & A_H
    # prec denom
    if len(A_H) == 0:
        p_act = 1.0 if len(A_G) == 0 else 0.0
    else:
        p_act = len(inter_actions) / len(A_H)
    r_act = len(inter_actions) / len(A_G) if len(A_G) > 0 else 1.0

    # parameter-level: solo acciones emparejadas
    M = inter_actions
    num_intersect_params = 0
    denom_par_prec = 0
    denom_par_rec = 0

    # construir mapping por nombre para acceso
    map_g = {a["name"]: a for a in g_actions if a.get("name")}
    map_h = {a["name"]: a for a in h_actions if a.get("name")}

    for a in M:
        p_i_g_keys = map_g[a]["param_keys"] if a in map_g else set()
        p_i_h_keys = map_h[a]["param_keys"] if a in map_h else set()
        num_intersect_params += len(p_i_g_keys & p_i_h_keys)
        denom_par_prec += len(p_i_h_keys)
        denom_par_rec += len(p_i_g_keys)

    p_par = (num_intersect_params / denom_par_prec) if denom_par_prec > 0 else 1.0
    r_par = (num_intersect_params / denom_par_rec) if denom_par_rec > 0 else 1.0

    h_p = lam * p_act + (1 - lam) * p_par
    h_r = lam * r_act + (1 - lam) * r_par
    if h_p + h_r > 0:
        h_f1 = 2 * h_p * h_r / (h_p + h_r)
    else:
        h_f1 = 0.0

    return {
        "p_act": float(p_act),
        "r_act": float(r_act),
        "p_par": float(p_par),
        "r_par": float(r_par),
        "hierarchical_precision": float(h_p),
        "hierarchical_recall": float(h_r),
        "hierarchical_f1": float(h_f1),
        # "num_actions_G": len(A_G),
        # "num_actions_H": len(A_H),
        # "num_matched_actions": len(M),
        # "num_common_param_instances": int(num_intersect_params),
    }


def compute_content_accuracy(g_actions: List[Dict], h_actions: List[Dict]) -> Dict[str, Any]:
    """Calcula content accuracy según la instrucción del usuario:
    - Tomar los parámetros (instances) que están en común entre G y H (por acción)
    - Comparar el valor; contar matches/total_common
    Devuelve {'accuracy': float, 'num_common': int, 'num_value_matches': int}
    """
    map_g = {a["name"]: a for a in g_actions if a.get("name")}
    map_h = {a["name"]: a for a in h_actions if a.get("name")}

    common_params = 0
    value_matches = 0

    # consideramos solo parámetros de acciones que aparecen en H y en G
    for aname in map_h.keys():
        if aname not in map_g:
            continue
        keys_g = map_g[aname]["param_keys"]
        keys_h = map_h[aname]["param_keys"]
        common_keys = keys_g & keys_h
        for k in common_keys:
            common_params += 1
            v_g = map_g[aname]["param_values"].get(k)
            v_h = map_h[aname]["param_values"].get(k)
            # comparación robusta: igualdad estructural
            if v_g == v_h:
                value_matches += 1

    if common_params == 0:
        accuracy = 1.0
    else:
        accuracy = value_matches / common_params

    return {
      "accuracy": float(accuracy), 
      # "num_common": int(common_params), 
      # "num_matches": int(value_matches)
    }


def aggregate_metrics(items: List[Dict], key_func) -> Dict[str, Dict]:
    """Agrupa items por key_func(item) y promedia las métricas.
    Se espera que cada item tenga 'structure' con hierarchical_precision, hierarchical_recall, hierarchical_f1 y 'content' con accuracy."""
    groups = defaultdict(list)
    for it in items:
        k = key_func(it) or "Unknown"
        groups[k].append(it)

    out = {}
    for k, lst in groups.items():
        n = len(lst)
        sum_p_act = sum(x["structure"]["p_act"] for x in lst)
        sum_r_act = sum(x["structure"]["r_act"] for x in lst)
        sum_p_par = sum(x["structure"]["p_par"] for x in lst)
        sum_r_par = sum(x["structure"]["r_par"] for x in lst)
        sum_h_p = sum(x["structure"]["hierarchical_precision"] for x in lst)
        sum_h_r = sum(x["structure"]["hierarchical_recall"] for x in lst)
        sum_h_f1 = sum(x["structure"]["hierarchical_f1"] for x in lst)
        sum_acc = sum(x["content"]["accuracy"] for x in lst)
        # prepare lists for medians
        list_p_act = [x["structure"]["p_act"] for x in lst]
        list_r_act = [x["structure"]["r_act"] for x in lst]
        list_p_par = [x["structure"]["p_par"] for x in lst]
        list_r_par = [x["structure"]["r_par"] for x in lst]
        list_h_p = [x["structure"]["hierarchical_precision"] for x in lst]
        list_h_r = [x["structure"]["hierarchical_recall"] for x in lst]
        list_h_f1 = [x["structure"]["hierarchical_f1"] for x in lst]
        list_acc = [x["content"]["accuracy"] for x in lst]
        out[k] = {
            "structure_action_precision": sum_p_act / n,
            "structure_action_recall": sum_r_act / n,
            "structure_parameter_precision": sum_p_par / n,
            "structure_parameter_recall": sum_r_par / n,
            "structure_hierarchical_precision": sum_h_p / n,
            "structure_hierarchical_recall": sum_h_r / n,
            "structure_hierarchical_f1": sum_h_f1 / n,
            "content_accuracy": sum_acc / n,
            # medians
            "structure_action_precision_median": float(statistics.median(list_p_act)),
            "structure_action_recall_median": float(statistics.median(list_r_act)),
            "structure_parameter_precision_median": float(statistics.median(list_p_par)),
            "structure_parameter_recall_median": float(statistics.median(list_r_par)),
            "structure_hierarchical_precision_median": float(statistics.median(list_h_p)),
            "structure_hierarchical_recall_median": float(statistics.median(list_h_r)),
            "structure_hierarchical_f1_median": float(statistics.median(list_h_f1)),
            "content_accuracy_median": float(statistics.median(list_acc)),
                # IQRs
                "structure_action_precision_iqr": float(compute_iqr(list_p_act)),
                "structure_action_recall_iqr": float(compute_iqr(list_r_act)),
                "structure_parameter_precision_iqr": float(compute_iqr(list_p_par)),
                "structure_parameter_recall_iqr": float(compute_iqr(list_r_par)),
                "structure_hierarchical_precision_iqr": float(compute_iqr(list_h_p)),
                "structure_hierarchical_recall_iqr": float(compute_iqr(list_h_r)),
                "structure_hierarchical_f1_iqr": float(compute_iqr(list_h_f1)),
                "content_accuracy_iqr": float(compute_iqr(list_acc)),
        }
    return out


def build_report(experiments: List[Dict]) -> Dict:
    details = []

    for idx, e in enumerate(experiments):
        g_plan = safe_get(e, "input", "plan", "actions")
        h_plan = safe_get(e, "api_response", "plan", "actions")
        g_actions = extract_actions(g_plan)
        h_actions = extract_actions(h_plan)

        struct = compute_structure_metrics(g_actions, h_actions, lam=0.5)
        content = compute_content_accuracy(g_actions, h_actions)

        # metadata para agrupar: intentar obtener campos comunes
        template = safe_get(e, "input", "template")
        question = safe_get(e, "input", "question")

        details.append({
            "index": idx,
            "template": template,
            "question": question,
            "structure": struct,
            "content": content,
        })

    # overall aggregates: promedios
    n = len(details) if details else 0
    if n == 0:
        overall = {
            "structure_p_act": 0.0,
            "structure_r_act": 0.0,
            "structure_p_par": 0.0,
            "structure_r_par": 0.0,
            "structure_hierarchical_precision": 0.0,
            "structure_hierarchical_recall": 0.0,
            "structure_hierarchical_f1": 0.0,
            "content_accuracy": 0.0,
            "structure_p_act_median": 0.0,
            "structure_r_act_median": 0.0,
            "structure_p_par_median": 0.0,
            "structure_r_par_median": 0.0,
            "structure_hierarchical_precision_median": 0.0,
            "structure_hierarchical_recall_median": 0.0,
            "structure_hierarchical_f1_median": 0.0,
            "content_accuracy_median": 0.0,
            "structure_p_act_iqr": 0.0,
            "structure_r_act_iqr": 0.0,
            "structure_p_par_iqr": 0.0,
            "structure_r_par_iqr": 0.0,
            "structure_hierarchical_precision_iqr": 0.0,
            "structure_hierarchical_recall_iqr": 0.0,
            "structure_hierarchical_f1_iqr": 0.0,
            "content_accuracy_iqr": 0.0,
        }
    else:
        structure_p_act = sum(d["structure"]["p_act"] for d in details) / n
        structure_r_act = sum(d["structure"]["r_act"] for d in details) / n
        structure_p_par = sum(d["structure"]["p_par"] for d in details) / n
        structure_r_par = sum(d["structure"]["r_par"] for d in details) / n
        structure_precision = sum(d["structure"]["hierarchical_precision"] for d in details) / n
        structure_recall = sum(d["structure"]["hierarchical_recall"] for d in details) / n
        # f1 from averaged precision & recall
        structure_f1 = (2 * structure_precision * structure_recall / (structure_precision + structure_recall)) if (structure_precision + structure_recall) > 0 else 0.0
        content_accuracy = sum(d["content"]["accuracy"] for d in details) / n
        # medians for overall
        list_p_act = [d["structure"]["p_act"] for d in details]
        list_r_act = [d["structure"]["r_act"] for d in details]
        list_p_par = [d["structure"]["p_par"] for d in details]
        list_r_par = [d["structure"]["r_par"] for d in details]
        list_h_p = [d["structure"]["hierarchical_precision"] for d in details]
        list_h_r = [d["structure"]["hierarchical_recall"] for d in details]
        list_h_f1 = [d["structure"]["hierarchical_f1"] for d in details]
        list_acc = [d["content"]["accuracy"] for d in details]
        overall = {
            "structure_p_act": float(structure_p_act),
            "structure_r_act": float(structure_r_act),
            "structure_p_par": float(structure_p_par),
            "structure_r_par": float(structure_r_par),
            "structure_hierarchical_precision": float(structure_precision),
            "structure_hierarchical_recall": float(structure_recall),
            "structure_hierarchical_f1": float(structure_f1),
            "content_accuracy": float(content_accuracy),
            "structure_p_act_median": float(statistics.median(list_p_act)),
            "structure_r_act_median": float(statistics.median(list_r_act)),
            "structure_p_par_median": float(statistics.median(list_p_par)),
            "structure_r_par_median": float(statistics.median(list_r_par)),
            "structure_hierarchical_precision_median": float(statistics.median(list_h_p)),
            "structure_hierarchical_recall_median": float(statistics.median(list_h_r)),
            "structure_hierarchical_f1_median": float(statistics.median(list_h_f1)),
            "content_accuracy_median": float(statistics.median(list_acc)),
            "structure_p_act_iqr": float(compute_iqr(list_p_act)),
            "structure_r_act_iqr": float(compute_iqr(list_r_act)),
            "structure_p_par_iqr": float(compute_iqr(list_p_par)),
            "structure_r_par_iqr": float(compute_iqr(list_r_par)),
            "structure_hierarchical_precision_iqr": float(compute_iqr(list_h_p)),
            "structure_hierarchical_recall_iqr": float(compute_iqr(list_h_r)),
            "structure_hierarchical_f1_iqr": float(compute_iqr(list_h_f1)),
            "content_accuracy_iqr": float(compute_iqr(list_acc)),
        }

    # agrupaciones
    by_template = aggregate_metrics(details, lambda d: d.get("template") or "Unknown")

    report = {
        "overall": overall,
        "by_template": by_template,
        "details": details,
    }

    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to experiment_results.json")
    p.add_argument("--output_dir", required=True, help="Directory to write the report")
    p.add_argument("--outfile", default="evaluation_report_generated.json", help="Output filename")
    args = p.parse_args()

    experiments = load_json(args.input)
    report = build_report(experiments)

    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, args.outfile)
    dump_json(report, outpath)
    print(f"Wrote evaluation report to {outpath}")


if __name__ == "__main__":
    main()
