"""Small CPU-only ML-engineering task templates."""

from __future__ import annotations

import json
import random
from textwrap import dedent


def _classification_csv(seed: int, n: int = 80, x_names: tuple[str, str] = ("x1", "x2")) -> str:
    rng = random.Random(seed)
    rows = [f"{x_names[0]},{x_names[1]},label"]
    for _ in range(n):
        x1 = rng.uniform(-2.0, 2.0)
        x2 = rng.uniform(-2.0, 2.0)
        label = int(x1 - 0.45 * x2 + rng.uniform(-0.1, 0.1) > 0.0)
        rows.append(f"{x1:.5f},{x2:.5f},{label}")
    return "\n".join(rows) + "\n"


def _age_csv(seed: int, n: int = 70) -> str:
    rng = random.Random(seed)
    rows = ["age,income,label"]
    for _ in range(n):
        age = rng.randint(18, 75)
        income = rng.randint(25_000, 130_000)
        label = int(income > 72_000)
        rows.append(f"{age},{income},{label}")
    return "\n".join(rows) + "\n"


def _latency_csv(seed: int, n: int = 60) -> str:
    rng = random.Random(seed)
    rows = ["x1,x2,label"]
    for _ in range(n):
        x1 = rng.uniform(-1.5, 1.5)
        x2 = rng.uniform(-1.5, 1.5)
        label = int(x1 + x2 > 0.0)
        rows.append(f"{x1:.5f},{x2:.5f},{label}")
    return "\n".join(rows) + "\n"


def _hidden_validation(metric_name: str, threshold: float, extra_python: str = "") -> str:
    return dedent(
        f"""
        import json
        import subprocess
        import sys


        def _run(cmd):
            result = subprocess.run(cmd, text=True, capture_output=True, check=False)
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            if result.returncode != 0:
                raise SystemExit(result.returncode)
            return result.stdout


        _run([sys.executable, "-m", "pytest", "-q"])
        _run([sys.executable, "train.py"])
        output = _run([sys.executable, "eval.py"])
        metrics = {{}}
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("{{") and line.endswith("}}"):
                metrics.update(json.loads(line))
        value = float(metrics.get("{metric_name}", -1.0))
        assert value >= {threshold!r}, metrics
        {extra_python}
        print(json.dumps({{"hidden_validation_passed": True, "{metric_name}": value}}))
        """
    ).strip() + "\n"


def shape_mismatch_training_crash(seed: int) -> tuple[dict[str, str], dict[str, str], str, float]:
    files = {
        "data/train.csv": _classification_csv(seed, 90),
        "data/test.csv": _classification_csv(seed + 10_000, 40),
        "train.py": dedent(
            """
            import json
            from pathlib import Path

            import numpy as np
            import pandas as pd
            from sklearn.linear_model import LogisticRegression


            def load_training_data():
                df = pd.read_csv("data/train.csv")
                X = df[["x1", "x2"]].to_numpy()
                y = df["label"].to_numpy()
                return X, y


            def main():
                X, y = load_training_data()
                weights = np.ones(3)
                _ = X @ weights
                model = LogisticRegression(random_state=0, solver="liblinear")
                model.fit(X, y)
                Path("artifacts").mkdir(exist_ok=True)
                payload = {
                    "coef": model.coef_[0].tolist(),
                    "intercept": float(model.intercept_[0]),
                    "features": ["x1", "x2"],
                }
                Path("artifacts/model.json").write_text(json.dumps(payload))
                print(json.dumps({"train_accuracy": float(model.score(X, y))}))


            if __name__ == "__main__":
                main()
            """
        ).lstrip(),
        "eval.py": dedent(
            """
            import json
            from pathlib import Path

            import numpy as np
            import pandas as pd


            def sigmoid(z):
                return 1.0 / (1.0 + np.exp(-z))


            def main():
                model = json.loads(Path("artifacts/model.json").read_text())
                df = pd.read_csv("data/test.csv")
                X = df[model["features"]].to_numpy()
                y = df["label"].to_numpy()
                logits = X @ np.asarray(model["coef"]) + model["intercept"]
                preds = (sigmoid(logits) >= 0.5).astype(int)
                accuracy = float((preds == y).mean())
                print(json.dumps({"accuracy": accuracy}))


            if __name__ == "__main__":
                main()
            """
        ).lstrip(),
        "smoke_test.py": dedent(
            """
            from train import load_training_data


            X, y = load_training_data()
            assert X.shape[0] == y.shape[0]
            assert X.shape[1] == 2
            print("smoke ok")
            """
        ).lstrip(),
        "tests/test_training.py": dedent(
            """
            import subprocess
            import sys
            from pathlib import Path


            def test_training_completes_and_writes_model():
                result = subprocess.run([sys.executable, "train.py"], text=True, capture_output=True)
                assert result.returncode == 0, result.stderr + result.stdout
                assert Path("artifacts/model.json").exists()
            """
        ).lstrip(),
        "task_metadata.json": json.dumps({"family": "shape_mismatch_training_crash", "seed": seed}) + "\n",
    }
    hidden = {"hidden_validation.py": _hidden_validation("accuracy", 0.75)}
    return files, hidden, "accuracy", 0.75


def missing_column_preprocessing(seed: int) -> tuple[dict[str, str], dict[str, str], str, float]:
    files = {
        "data/train.csv": _age_csv(seed, 80),
        "data/test.csv": _age_csv(seed + 10_000, 35),
        "preprocess.py": dedent(
            """
            FEATURE_COLUMNS = ["age_years", "income"]


            def make_features(df):
                return df[FEATURE_COLUMNS].to_numpy(dtype=float)
            """
        ).lstrip(),
        "train.py": dedent(
            """
            import json
            from pathlib import Path

            import pandas as pd
            from sklearn.linear_model import LogisticRegression

            from preprocess import make_features


            def main():
                df = pd.read_csv("data/train.csv")
                X = make_features(df)
                y = df["label"].to_numpy()
                model = LogisticRegression(random_state=0, solver="liblinear")
                model.fit(X, y)
                Path("artifacts").mkdir(exist_ok=True)
                Path("artifacts/model.json").write_text(json.dumps({
                    "coef": model.coef_[0].tolist(),
                    "intercept": float(model.intercept_[0]),
                    "features": ["age_years", "income"],
                }))
                print(json.dumps({"train_accuracy": float(model.score(X, y))}))


            if __name__ == "__main__":
                main()
            """
        ).lstrip(),
        "eval.py": dedent(
            """
            import json
            from pathlib import Path

            import numpy as np
            import pandas as pd

            from preprocess import make_features


            def main():
                model = json.loads(Path("artifacts/model.json").read_text())
                df = pd.read_csv("data/test.csv")
                X = make_features(df)
                y = df["label"].to_numpy()
                logits = X @ np.asarray(model["coef"]) + model["intercept"]
                preds = (logits >= 0.0).astype(int)
                print(json.dumps({"accuracy": float((preds == y).mean())}))


            if __name__ == "__main__":
                main()
            """
        ).lstrip(),
        "smoke_test.py": dedent(
            """
            import pandas as pd

            from preprocess import make_features


            df = pd.read_csv("data/train.csv").head(3)
            X = make_features(df)
            assert X.shape == (3, 2)
            print("smoke ok")
            """
        ).lstrip(),
        "tests/test_preprocess.py": dedent(
            """
            import pandas as pd

            from preprocess import make_features


            def test_preprocess_accepts_training_schema():
                df = pd.read_csv("data/train.csv").head(5)
                X = make_features(df)
                assert X.shape == (5, 2)
            """
        ).lstrip(),
        "task_metadata.json": json.dumps({"family": "missing_column_preprocessing", "seed": seed}) + "\n",
    }
    hidden = {"hidden_validation.py": _hidden_validation("accuracy", 0.75)}
    return files, hidden, "accuracy", 0.75


def reproducibility_failure(seed: int) -> tuple[dict[str, str], dict[str, str], str, float]:
    data = _classification_csv(seed, 120)
    files = {
        "data/train.csv": data,
        "data/test.csv": _classification_csv(seed + 10_000, 40),
        "train.py": dedent(
            """
            import json
            from pathlib import Path

            import pandas as pd
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split


            def train_once():
                df = pd.read_csv("data/train.csv")
                X = df[["x1", "x2"]].to_numpy()
                y = df["label"].to_numpy()
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
                model = RandomForestClassifier(n_estimators=15)
                model.fit(X_train, y_train)
                accuracy = float(model.score(X_val, y_val))
                return {"accuracy": accuracy, "n_estimators": 15}


            def main():
                result = train_once()
                Path("artifacts").mkdir(exist_ok=True)
                Path("artifacts/model.json").write_text(json.dumps(result))
                print(json.dumps(result))


            if __name__ == "__main__":
                main()
            """
        ).lstrip(),
        "eval.py": dedent(
            """
            import json

            import train


            def main():
                values = [train.train_once()["accuracy"] for _ in range(3)]
                stable = max(values) - min(values) <= 1e-12
                print(json.dumps({
                    "accuracy": float(sum(values) / len(values)),
                    "reproducibility_score": 1.0 if stable else 0.0,
                }))


            if __name__ == "__main__":
                main()
            """
        ).lstrip(),
        "smoke_test.py": "import train\nprint(train.train_once())\n",
        "tests/test_reproducibility.py": dedent(
            """
            import train


            def test_training_is_reproducible():
                values = [train.train_once()["accuracy"] for _ in range(3)]
                assert max(values) - min(values) <= 1e-12, values
            """
        ).lstrip(),
        "task_metadata.json": json.dumps({"family": "reproducibility_failure", "seed": seed}) + "\n",
    }
    extra = 'assert metrics.get("reproducibility_score", 0.0) == 1.0, metrics'
    hidden = {"hidden_validation.py": _hidden_validation("reproducibility_score", 1.0, extra)}
    return files, hidden, "reproducibility_score", 1.0


def metric_regression(seed: int) -> tuple[dict[str, str], dict[str, str], str, float]:
    files = {
        "data/train.csv": _classification_csv(seed, 100),
        "data/test.csv": _classification_csv(seed + 10_000, 45),
        "features.py": dedent(
            """
            def make_features(df):
                return df[["x1", "x2"]].to_numpy()
            """
        ).lstrip(),
        "train.py": dedent(
            """
            import json
            from pathlib import Path

            import pandas as pd
            from sklearn.linear_model import LogisticRegression

            from features import make_features


            def main():
                df = pd.read_csv("data/train.csv")
                X = make_features(df)
                target = 1 - df["label"].to_numpy()
                model = LogisticRegression(random_state=0, solver="liblinear")
                model.fit(X, target)
                Path("artifacts").mkdir(exist_ok=True)
                Path("artifacts/model.json").write_text(json.dumps({
                    "coef": model.coef_[0].tolist(),
                    "intercept": float(model.intercept_[0]),
                    "features": ["x1", "x2"],
                }))
                print(json.dumps({"train_accuracy": float(model.score(X, target))}))


            if __name__ == "__main__":
                main()
            """
        ).lstrip(),
        "eval.py": dedent(
            """
            import json
            from pathlib import Path

            import numpy as np
            import pandas as pd

            from features import make_features


            def main():
                model = json.loads(Path("artifacts/model.json").read_text())
                df = pd.read_csv("data/test.csv")
                X = make_features(df)
                y = df["label"].to_numpy()
                logits = X @ np.asarray(model["coef"]) + model["intercept"]
                preds = (logits >= 0.0).astype(int)
                print(json.dumps({"accuracy": float((preds == y).mean())}))


            if __name__ == "__main__":
                main()
            """
        ).lstrip(),
        "smoke_test.py": "import pandas as pd\nfrom features import make_features\nprint(make_features(pd.read_csv('data/train.csv').head()).shape)\n",
        "tests/test_metric.py": dedent(
            """
            import json
            import subprocess
            import sys


            def test_eval_metric_above_threshold():
                train = subprocess.run([sys.executable, "train.py"], text=True, capture_output=True)
                assert train.returncode == 0, train.stderr + train.stdout
                result = subprocess.run([sys.executable, "eval.py"], text=True, capture_output=True)
                assert result.returncode == 0, result.stderr + result.stdout
                metrics = {}
                for line in result.stdout.splitlines():
                    if line.startswith("{"):
                        metrics.update(json.loads(line))
                assert metrics["accuracy"] >= 0.75, metrics
            """
        ).lstrip(),
        "task_metadata.json": json.dumps({"family": "metric_regression", "seed": seed}) + "\n",
    }
    hidden = {"hidden_validation.py": _hidden_validation("accuracy", 0.75)}
    return files, hidden, "accuracy", 0.75


def inference_latency_regression(seed: int) -> tuple[dict[str, str], dict[str, str], str, float]:
    files = {
        "data/train.csv": _latency_csv(seed, 70),
        "data/test.csv": _latency_csv(seed + 10_000, 30),
        "train.py": dedent(
            """
            import json
            from pathlib import Path

            import pandas as pd


            def main():
                df = pd.read_csv("data/train.csv")
                coef = [1.0, 1.0]
                Path("artifacts").mkdir(exist_ok=True)
                Path("artifacts/model.json").write_text(json.dumps({"coef": coef, "intercept": 0.0}))
                print(json.dumps({"train_rows": int(len(df))}))


            if __name__ == "__main__":
                main()
            """
        ).lstrip(),
        "inference.py": dedent(
            """
            import json
            import time
            from pathlib import Path

            import numpy as np


            def _load_model():
                time.sleep(0.03)
                return json.loads(Path("artifacts/model.json").read_text())


            def predict_batch(rows):
                preds = []
                for row in rows:
                    model = _load_model()
                    score = float(np.dot(row, model["coef"]) + model["intercept"])
                    preds.append(1 if score >= 0.0 else 0)
                return preds
            """
        ).lstrip(),
        "eval.py": dedent(
            """
            import json
            import time

            import pandas as pd

            from inference import predict_batch


            def main():
                df = pd.read_csv("data/test.csv")
                rows = df[["x1", "x2"]].to_numpy()
                start = time.perf_counter()
                preds = predict_batch(rows)
                runtime = time.perf_counter() - start
                accuracy = sum(int(p == y) for p, y in zip(preds, df["label"])) / len(df)
                print(json.dumps({"accuracy": float(accuracy), "latency_seconds": float(runtime)}))


            if __name__ == "__main__":
                main()
            """
        ).lstrip(),
        "smoke_test.py": "import train\ntrain.main()\nfrom inference import predict_batch\nprint(predict_batch([[1.0, 1.0], [-1.0, -1.0]]))\n",
        "tests/test_latency.py": dedent(
            """
            import time

            import train
            from inference import predict_batch


            def test_batch_prediction_is_fast_and_correct():
                train.main()
                rows = [[1.0, 1.0], [-1.0, -1.0]] * 10
                start = time.perf_counter()
                preds = predict_batch(rows)
                runtime = time.perf_counter() - start
                assert preds[:2] == [1, 0]
                assert runtime < 0.25, runtime
            """
        ).lstrip(),
        "task_metadata.json": json.dumps({"family": "inference_latency_regression", "seed": seed}) + "\n",
    }
    extra = 'assert metrics.get("latency_seconds", 99.0) < 0.25, metrics'
    hidden = {"hidden_validation.py": _hidden_validation("accuracy", 0.75, extra)}
    return files, hidden, "accuracy", 0.75


TEMPLATE_BUILDERS = {
    "shape_mismatch_training_crash": shape_mismatch_training_crash,
    "missing_column_preprocessing": missing_column_preprocessing,
    "reproducibility_failure": reproducibility_failure,
    "metric_regression": metric_regression,
    "inference_latency_regression": inference_latency_regression,
}
