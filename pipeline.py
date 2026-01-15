##start
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
import boto3

# --- AWS & SageMaker Infrastructure Setup ---

# Get the AWS region from the environment (e.g., us-east-1)
region = boto3.Session().region_name

# The IAM Execution Role providing SageMaker permission to access S3 and other AWS resources
role = "arn:aws:iam::461627999973:role/service-role/AmazonSageMaker-ExecutionRole-20240808T215869"

# The SageMaker Session handles the underlying API calls to AWS services
sagemaker_session = sagemaker.session.Session()

# --- Pipeline Runtime Parameters ---

# ParameterString allows us to change the input S3 path without modifying the code itself
input_data = ParameterString(
    name="InputData", 
    default_value="s3://sagemaker-cicd-lab-unus/churn.csv"
)

# --- Step 1: Data Processing ---

# Initialize the Scikit-Learn processor container
processor = SKLearnProcessor(
    framework_version="1.2-1", 
    role=role, 
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=sagemaker_session
)

# Define the Processing Step logic: runs 'process.py' inside the container
step_process = ProcessingStep(
    name="ProcessData",
    processor=processor,
    inputs=[
        # Maps the S3 input file to the container's internal local path
        sagemaker.processing.ProcessingInput(
            source=input_data, 
            destination="/opt/ml/processing/input"
        )
    ],
    code="process.py", # Local script that performs cleaning/feature engineering
    outputs=[
        # Maps the container's output folder to an S3 bucket for persistent storage
        sagemaker.processing.ProcessingOutput(
            output_name="train_data", # Key used to reference this data in the next step
            source="/opt/ml/processing/output", 
            destination="s3://sagemaker-cicd-lab-unus/processed"
        )
    ]
)

# --- Step 2: Model Training ---

# Automatically find the Docker Image URI for the official SageMaker XGBoost algorithm
image_uri = sagemaker.image_uris.retrieve("xgboost", region, version="1.7-1")

# Define the Training Estimator configuration
estimator = Estimator(
    image_uri=image_uri,
    instance_type="ml.m5.large",
    instance_count=1,
    role=role,
    output_path="s3://sagemaker-cicd-lab-unus/output/", # Where 'model.tar.gz' will be uploaded
    sagemaker_session=sagemaker_session
)

# Set algorithm-specific hyperparameters (Binary classification for Churn)
estimator.set_hyperparameters(objective="binary:logistic", num_round=100)

# Define the Training Step
step_train = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            # DYNAMIC LINKING: This step waits for 'step_process' to finish and
            # automatically pulls the S3 path from its 'train_data' output.
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)

# --- Pipeline Assembly ---

# Create the Directed Acyclic Graph (DAG) by defining the sequence of steps
pipeline = Pipeline(
    name="ChurnPredictionPipeline",
    parameters=[input_data],
    steps=[step_process, step_train],
    sagemaker_session=sagemaker_session
)

# --- Pipeline Execution Trigger ---

if __name__ == "__main__":
    print("Upserting and starting pipeline...")
    
    # upsert() creates the pipeline definition in AWS or updates it if it already exists
    pipeline.upsert(role_arn=role)
    
    # Triggers the actual execution of the workflow
    execution = pipeline.start()
    print(f"Execution started: {execution.arn}")
    
    # wait() pauses the local script and prints logs until the cloud job is done
    execution.wait()
    print(f"Final Status: {execution.describe()['PipelineExecutionStatus']}")
##end
