from src.mlops.pipelines.mlops_pipeline_mixin import MLOpsPipelineMixin
from src.pipelines.churn_pipeline import ChurnPipeline


class ChurnMLOpsPipeline(MLOpsPipelineMixin, ChurnPipeline):
    mlflow_experiment_name = "churn"
    registry_model_names = ["churn_random_forest", "churn_xgboost"]
