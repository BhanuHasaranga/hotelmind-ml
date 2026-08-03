from src.mlops.pipelines.mlops_pipeline_mixin import MLOpsPipelineMixin
from src.pipelines.staffing_pipeline import StaffingPipeline


class StaffingMLOpsPipeline(MLOpsPipelineMixin, StaffingPipeline):
    mlflow_experiment_name = "staffing"
    registry_model_names = ["staffing_regression"]
