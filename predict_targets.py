from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, Optional

import pandas as pd
from catboost import CatBoostRegressor, Pool

from preprocess import load_schema, transform_with_schema


logger = logging.getLogger(__name__)


MODEL_FILES_BY_TARGET = {
    "Recommended_Calories": "catboost_reco_calories.cbm",
    "Recommended_Protein": "catboost_reco_protein.cbm",
    "Recommended_Carbs": "catboost_reco_carbs.cbm",
    "Recommended_Fats": "catboost_reco_fats.cbm",
    "TARGET_CALORIES_PROFILE_ONLY_KCAL": "catboost_reco_calories.cbm",
    "TARGET_PROTEIN_PROFILE_ONLY_G": "catboost_reco_protein.cbm",
    "TARGET_CARBS_PROFILE_ONLY_G": "catboost_reco_carbs.cbm",
    "TARGET_FATS_PROFILE_ONLY_G": "catboost_reco_fats.cbm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict nutrition targets and build recipe-model payload."
    )
    parser.add_argument(
        "--mode",
        choices=["profile_only", "profile_plus_intake"],
        default="profile_only",
        help="Feature mode.",
    )
    parser.add_argument(
        "--feature_set",
        choices=["engineered", "raw", "raw_user_form"],
        default="raw_user_form",
        help="Feature set used for training the models.",
    )
    parser.add_argument(
        "--models_dir",
        default="artifacts/profile_only_run/raw/profile_only",
        help=(
            "Directory containing models and schema "
            "(defaults to artifacts/profile_only_run/<feature_set>/<mode>)."
        ),
    )
    parser.add_argument("--input_json", default="schem.json", help="Path to JSON file with user profile.")
    parser.add_argument("--input_str", help="Raw JSON string with user profile.")
    return parser.parse_args()


def _parse_json_input(path: Optional[str], raw: Optional[str]) -> Dict[str, object]:
    if bool(path) == bool(raw):
        raise ValueError("Provide exactly one of --input_json or --input_str.")
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(raw)


def _clamp(value: float, lower: float, upper: Optional[float] = None) -> float:
    value = max(lower, value)
    if upper is not None:
        value = min(upper, value)
    return value


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    payload = _parse_json_input(args.input_json, args.input_str)
    models_dir = args.models_dir or os.path.join(
        "artifacts", "profile_only_run", args.feature_set, args.mode
    )
    schema_path = os.path.join(models_dir, "schema.json")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    schema = load_schema(schema_path)
    target_cols = schema.target_cols

    df_input = pd.DataFrame([payload])
    X_input, _ = transform_with_schema(df_input, args.mode, schema)
    cat_indices = [
        schema.feature_cols.index(col)
        for col in schema.categorical_cols
        if col in schema.feature_cols
    ]
    pool = Pool(X_input, cat_features=cat_indices)

    preds: Dict[str, float] = {}
    for target in target_cols:
        model_name = MODEL_FILES_BY_TARGET.get(target)
        if not model_name:
            raise ValueError(f"No model file mapping for target: {target}")
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = CatBoostRegressor()
        model.load_model(model_path)
        pred = float(model.predict(pool)[0])
        preds[target] = pred

    output_map = {
        "TARGET_CALORIES_PROFILE_ONLY_KCAL": "Recommended_Calories",
        "TARGET_PROTEIN_PROFILE_ONLY_G": "Recommended_Protein",
        "TARGET_CARBS_PROFILE_ONLY_G": "Recommended_Carbs",
        "TARGET_FATS_PROFILE_ONLY_G": "Recommended_Fats",
        "Recommended_Calories": "Recommended_Calories",
        "Recommended_Protein": "Recommended_Protein",
        "Recommended_Carbs": "Recommended_Carbs",
        "Recommended_Fats": "Recommended_Fats",
    }
    output_preds: Dict[str, float] = {}
    for internal_name, pred in preds.items():
        public_name = output_map.get(internal_name)
        if public_name is None:
            continue
        output_preds[public_name] = pred

    required_outputs = [
        "Recommended_Calories",
        "Recommended_Protein",
        "Recommended_Carbs",
        "Recommended_Fats",
    ]
    missing_outputs = [name for name in required_outputs if name not in output_preds]
    if missing_outputs:
        raise ValueError(f"Missing outputs for targets: {missing_outputs}")

    kcal = int(round(output_preds["Recommended_Calories"]))
    kcal = int(_clamp(kcal, 1200, 4500))
    protein_g = round(_clamp(output_preds["Recommended_Protein"], 0.0), 1)
    carbs_g = round(_clamp(output_preds["Recommended_Carbs"], 0.0), 1)
    fat_g = round(_clamp(output_preds["Recommended_Fats"], 0.0), 1)

    output = {
        "Recommended_Calories": kcal,
        "Recommended_Protein": protein_g,
        "Recommended_Carbs": carbs_g,
        "Recommended_Fats": fat_g,
        "mode": args.mode,
        "feature_set": args.feature_set,
    }

    print(json.dumps(output, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
