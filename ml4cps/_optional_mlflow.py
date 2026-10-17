"""Optional MLflow support shared by the learning modules."""

try:
    import mlflow
except ModuleNotFoundError as exc:
    if exc.name != "mlflow":
        raise

    class _UnavailableMLflow:
        def __getattr__(self, name):
            raise ModuleNotFoundError(
                'MLflow support requires pip install "ml4cps[mlflow]".',
                name="mlflow",
            )

    mlflow = _UnavailableMLflow()
