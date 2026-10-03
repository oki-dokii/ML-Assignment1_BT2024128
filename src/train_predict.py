from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from common import FINAL_CONFIGS, ROLL_NUMBER, ROOT, load_dataset, make_pipeline


def main() -> None:
    predictions_dir = ROOT / "predictions"
    models_dir = ROOT / "models"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    for problem, config in FINAL_CONFIGS.items():
        train = load_dataset(problem, "train")
        test = load_dataset(problem, "test")
        features = list(config["features"])
        model = make_pipeline(config["degree"])
        model.fit(train.loc[:, features], train["y"])
        predictions = model.predict(test.loc[:, features])

        if len(predictions) != len(test):
            raise RuntimeError(f"Prediction length mismatch for {problem}")
        if not np.isfinite(predictions).all():
            raise RuntimeError(f"Non-finite predictions produced for {problem}")

        output_path = predictions_dir / f"{ROLL_NUMBER}_pred_{problem}.csv"
        pd.DataFrame({"y": predictions}).to_csv(output_path, index=False)
        joblib.dump(model, models_dir / f"{ROLL_NUMBER}_{problem}_polynomial_model.joblib")
        print(
            f"{problem}: degree={config['degree']}, features={features}, "
            f"rows={len(predictions)}, min={predictions.min():.6f}, "
            f"max={predictions.max():.6f}, output={output_path}"
        )


if __name__ == "__main__":
    main()
