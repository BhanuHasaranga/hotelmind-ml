from src.mlops.pipelines.mlops_pipeline_mixin import MLOpsPipelineMixin
from src.pipelines.occupancy_pipeline import OccupancyPipeline


class OccupancyMLOpsPipeline(MLOpsPipelineMixin, OccupancyPipeline):
    mlflow_experiment_name = "occupancy"
    registry_model_names = ["occupancy_prophet", "occupancy_xgboost"]
