import numpy as np

from src.pipelines.evaluation_pipeline import ModelEvaluationPipeline


class ConstantMetric:
    name = "Constant Metric"

    def compute(self, y_true, y_pred, y_proba=None):
        return 0.875


class DictMetric:
    name = "Dict Metric"

    def compute(self, y_true, y_pred, y_proba=None):
        return {"matrix": [[2, 0], [1, 3]], "labels": [0, 1]}


class FailingMetric:
    name = "Failing Metric"

    def compute(self, y_true, y_pred, y_proba=None):
        raise RuntimeError("metric failed")


def test_evaluation_pipeline_stores_each_metric_result():
    pipeline = ModelEvaluationPipeline(metrics=[ConstantMetric(), DictMetric()])
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    result = pipeline.evaluate_model("Modelo IA", y_true, y_pred)

    assert result["Constant Metric"] == 0.875
    assert result["Dict Metric"]["matrix"] == [[2, 0], [1, 3]]
    assert pipeline.results["Modelo IA"] == result


def test_evaluation_pipeline_converts_failed_metric_to_zero():
    pipeline = ModelEvaluationPipeline(metrics=[ConstantMetric(), FailingMetric()])
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 0, 1])

    result = pipeline.evaluate_model("Modelo con fallo", y_true, y_pred)

    assert result["Constant Metric"] == 0.875
    assert result["Failing Metric"] == 0.0


def test_summary_dataframe_keeps_only_scalar_metrics():
    pipeline = ModelEvaluationPipeline(metrics=[ConstantMetric(), DictMetric()])
    pipeline.evaluate_model(
        "Modelo IA",
        y_true=np.array([0, 1]),
        y_pred=np.array([0, 1]),
    )

    summary = pipeline.get_summary_dataframe()

    assert list(summary["Modelo"]) == ["Modelo IA"]
    assert summary.loc[0, "Constant Metric"] == 0.875
    assert "Dict Metric" not in summary.columns
