# src/ml_data_pipeline/config.py
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any

class DataLoaderConfig(BaseModel):
    """Configuration for the data loader.

    Attributes:
        file_path (str): The path to the data file.
        file_type (str): The type of the data file (csv or json).
    """

    file_path: str
    file_type: str

    @field_validator("file_type")
    def validate_file_type(cls, value: str) -> str:
        """Validates the file type.

        Args:
            value (str): The file type to validate.

        Returns:
            str: The validated file type.

        Raises:
            ValueError: If the file type is not 'csv' or 'json'.
        """
        if value not in {"csv", "json"}:
            raise ValueError("file_type must be 'csv' or 'json'")
        return value


class TransformationConfig(BaseModel):
    """Configuration for the data transformation.

    Attributes:
        normalize (bool): Whether to normalize the data.
        scaling_method (str): The method to use for scaling (standard or minmax).
    """

    normalize: bool
    scaling_method: str

    @field_validator("scaling_method")
    def validate_scaling_method(cls, value: str) -> str:
        """Validates the scaling method.

        Args:
            value (str): The scaling method to validate.

        Returns:
            str: The validated scaling method.

        Raises:
            ValueError: If the scaling method is not 'standard' or 'minmax'.
        """
        if value not in {"standard", "minmax"}:
            raise ValueError("scaling_method must be 'standard' or 'minmax'")
        return value


class ModelConfig(BaseModel):
    """Configuration for the model.

    Attributes:
        type (str): The type of the model (logistic or decisiontree).
        params (Dict[str, Any]): Additional parameters for the model.
    """

    type: str
    params: Dict[str, Any] = {}
    @field_validator("type")
    def validate_model_type(cls, value: str) -> str:
        """Validates the model type.

        Args:
            value (str): The model type to validate.

        Returns:
            str: The validated model type.

        Raises:
            ValueError: If the model type is not 'logistic' or 'decisiontree'.
        """
        if value not in {"decisiontree", "logistic"}:
            raise ValueError("model type must be 'logistic' or 'decisiontree'")
        return value


class SplittingConfig(BaseModel):
    """Configuration for data splitting.

    Attributes:
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used for random number generation.
    """

    test_size: float = Field(
        ..., ge=0.0, le=1.0, description="Proportion of test data."
    )

    @field_validator("test_size")
    def validate_test_size(cls, value: float) -> float:
        """Validates the test size.

        Args:
            value (float): The test size to validate.

        Returns:
            float: The validated test size.

        Raises:
            ValueError: If the test size is not between 0 and 1.
        """
        if not (0.0 < value < 1.0):
            raise ValueError("test_size must be between 0 and 1.")
        return value

class MLflowConfig(BaseModel):
    """Configuration for MLflow.

    Attributes:
        tracking_uri (str): The URI for the MLflow tracking server. 
            Must be a valid URI.
        experiment_name (str): The name of the MLflow experiment. 
            Must not be empty.
    """

    tracking_uri: str
    experiment_name: str

    @field_validator("tracking_uri")
    def validate_tracking_uri(cls, value: str) -> str:
        """Validates the tracking URI.

        Args:
            value (str): The tracking URI to validate.

        Returns:
            str: The validated tracking URI.

        Raises:
            ValueError: If the URI is invalid.
        """
        if not value.startswith(("http://", "https://", "file://")):
            raise ValueError(
                "tracking_uri must start with 'http://', 'https://', or 'file://'."
            )
        return value

    @field_validator("experiment_name")
    def validate_experiment_name(cls, value: str) -> str:
        """Validates the experiment name.

        Args:
            value (str): The experiment name to validate.

        Returns:
            str: The validated experiment name.

        Raises:
            ValueError: If the name is empty or invalid.
        """
        if not value.strip():
            raise ValueError("experiment_name must not be empty.")
        return value



class Config(BaseModel):
    """Overall configuration for the pipeline.

    Attributes:
        data_loader (DataLoaderConfig): Configuration for the data loader.
        transformation (TransformationConfig): Configuration for the data transformation.
        model (ModelConfig): Configuration for the model.
        splitting (SplittingConfig): Configuration for data splitting.
        mlflow (MLflowConfig): Configuration for MLflow.
    """

    data_loader: DataLoaderConfig
    transformation: TransformationConfig
    model: ModelConfig
    splitting: SplittingConfig
    mlflow: MLflowConfig


def load_config(config_path: str) -> Config:
    """Loads the configuration from a file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        Config: The loaded configuration.
    """
    raw_config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(raw_config, resolve=True)
    return Config.model_validate(config_dict)
