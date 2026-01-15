import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import ModelStep 
from sagemaker.workflow.parameters import ParameterString
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession  # NEW IMPORT
import boto3

# --- AWS & SageMaker Infrastructure Setup ---
region = boto3.Session().region_name
role = "arn:aws:iam::461627999973:role/service-role/AmazonSageMaker-ExecutionRole-20240808T215869"

# FIX: Use PipelineSession instead of sagemaker.session.Session()
pipeline_session = PipelineSession() 

# --- Pipeline Runtime Parameters ---
input_data = ParameterString(
    name="InputData", 
    default_value="s3://sagemaker-cicd-lab-unus/churn.csv"
)

# --- Step 1: Data Processing ---
processor = SKLearnProcessor(
    framework_version="1.2-1", 
    role=role, 
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=pipeline_session  # Use pipeline_session
)

step_process = ProcessingStep(
    name="ProcessData",
    processor=processor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=input_data, 
            destination="/opt/ml/processing/input"
        )
    ],
    code="process.py",
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="train_data", 
            source="/opt/ml/processing/output", 
            destination="s3://sagemaker-cicd-lab-unus/processed"
        )
    ]
)

# --- Step 2: Model Training ---
image_uri = sagemaker.image_uris.retrieve("xgboost", region, version="1.7-1")

estimator = Estimator(
    image_uri=image_uri,
    instance_type="ml.m5.large",
    instance_count=1,
    role=role,
    output_path="s3://sagemaker-cicd-lab-unus/output/", 
    sagemaker_session=pipeline_session  # Use pipeline_session
)

estimator.set_hyperparameters(objective="binary:logistic", num_round=100)

step_train = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)

# --- Step 3: Model Registration ---
model = Model(
    image_uri=image_uri,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    sagemaker_session=pipeline_session  # Use pipeline_session
)

step_register = ModelStep(
    name="RegisterChurnModel",
    step_args=model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name="ChurnPredictionGroup",
        approval_status="PendingManualApproval"
    )
)

# --- Pipeline Assembly ---
pipeline = Pipeline(
    name="ChurnPredictionPipeline",
    parameters=[input_data],
    steps=[step_process, step_train, step_register],
    sagemaker_session=pipeline_session  # Use pipeline_session
)

# --- Pipeline Execution Trigger ---
if __name__ == "__main__":
    # For the upsert and start, we use the role ARN and pipeline session
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    print(f"Execution started.")
    execution.wait()
    print("Execution completed.")