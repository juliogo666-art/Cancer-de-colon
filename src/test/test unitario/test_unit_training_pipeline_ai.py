import pandas as pd
import pytest

from src.pipelines.training_pipeline import DEFAULT_FEATURES, TrainingPipeline


class MemorizingClassifier:
    def __init__(self):
        self.fit_was_called = False
        self.classes_ = [0, 1, 2]

    def fit(self, X, y):
        self.fit_was_called = True
        return self

    def predict(self, X):
        return [0, 1, 2] * (len(X) // 3) + [0] * (len(X) % 3)

    def predict_proba(self, X):
        base = [
            [0.80, 0.15, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.20, 0.75],
        ]
        return base * (len(X) // 3) + [base[0]] * (len(X) % 3)


def make_training_dataframe(rows_per_class=5):
    rows = []
    for risk_level in [0, 1, 2]:
        for index in range(rows_per_class):
            row = {
                feature: float(index + risk_level + 1)
                for feature in DEFAULT_FEATURES
            }
            row["Risk_Level_n"] = risk_level
            rows.append(row)
    return pd.DataFrame(rows)


def test_training_pipeline_loads_features_and_stratified_split(tmp_path):
    csv_path = tmp_path / "risk_training.csv"
    make_training_dataframe().to_csv(csv_path, index=False)

    pipeline = TrainingPipeline(csv_path=str(csv_path), test_size=0.2, random_state=7)
    pipeline.load_and_prepare()

    assert pipeline.X_train.shape == (12, len(DEFAULT_FEATURES))
    assert pipeline.X_test.shape == (3, len(DEFAULT_FEATURES))
    assert sorted(set(pipeline.y_test.tolist())) == [0, 1, 2]
    assert pipeline.features == DEFAULT_FEATURES


def test_training_pipeline_rejects_csv_with_missing_ai_feature(tmp_path):
    csv_path = tmp_path / "risk_training_missing_feature.csv"
    df = make_training_dataframe()
    df = df.drop(columns=[DEFAULT_FEATURES[0]])
    df.to_csv(csv_path, index=False)

    pipeline = TrainingPipeline(csv_path=str(csv_path))

    with pytest.raises(ValueError, match="Columnas no encontradas"):
        pipeline.load_and_prepare()


def test_training_pipeline_trains_and_returns_metrics(tmp_path):
    csv_path = tmp_path / "risk_training.csv"
    make_training_dataframe().to_csv(csv_path, index=False)
    pipeline = TrainingPipeline(csv_path=str(csv_path), test_size=0.2, random_state=7)
    pipeline.load_and_prepare()
    model = MemorizingClassifier()

    result = pipeline.train_and_evaluate(model, "Modelo unitario")

    assert model.fit_was_called is True
    assert result["model_name"] == "Modelo unitario"
    assert result["model"] is model
    assert "Accuracy" in result["metrics"]
    assert "F2-Score (macro)" in result["metrics"]
