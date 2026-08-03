from src.mlops.pipelines.mlops_pipeline_mixin import MLOpsPipelineMixin
from src.pipelines.pricing_pipeline import PricingPipeline


class PricingMLOpsPipeline(MLOpsPipelineMixin, PricingPipeline):
    mlflow_experiment_name = "pricing"
    registry_model_names = ["pricing_xgboost"]
