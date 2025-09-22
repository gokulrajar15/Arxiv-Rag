import os
from dotenv import load_dotenv
from enum import Enum
from pathlib import Path

# Define environment types
class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"

# Function to determine environment
def get_environment() -> Environment:
    """Get the current environment."""
    env = os.getenv("APP_ENV", "development").lower()
    if env in ["production", "prod"]:
        return Environment.PRODUCTION
    if env in ["staging", "stage"]:
        return Environment.STAGING
    if env == "test":
        return Environment.TEST
    return Environment.DEVELOPMENT

# Load environment-specific .env file
def load_env_file():
    """Load the correct environment-specific .env file."""
    env = get_environment()
    print(f"Loading environment: {env}")
    base_dir = Path(__file__).resolve().parent.parent.parent
    env_files = [
        base_dir / f".env.{env.value}.local",
        base_dir / f".env.{env.value}",
        base_dir / ".env.local",
        base_dir / ".env"
    ]

    for env_file in env_files:
        if env_file.is_file():
            load_dotenv(dotenv_path=env_file)
            print(f"Loaded environment from {env_file}")
            return env_file
    return None

ENV_FILE = load_env_file()

# Parsing utility functions
def parse_list_from_env(env_key, default=None):
    """Parse a comma-separated list from an environment variable."""
    value = os.getenv(env_key)
    if not value:
        return default or []
    value = value.strip("\"'")
    if "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]

def parse_dict_of_lists_from_env(prefix, default_dict=None):
    """Parse dictionary of lists from environment variables with a common prefix."""
    result = default_dict or {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            endpoint = key[len(prefix):].lower()
            if value:
                value = value.strip("\"'")
                result[endpoint] = [item.strip() for item in value.split(",") if item.strip()]
    return result

class Settings:
    """Application settings class."""

    def __init__(self):
        """Initialize settings from environment variables."""
        self.ENVIRONMENT = get_environment()

        # Application Settings
        self.PROJECT_NAME = os.getenv("PROJECT_NAME", "Arxiv RAG")
        self.PROJECT_VERSION = os.getenv("VERSION", "1.0.0")
        self.PROJECT_DESCRIPTION = os.getenv(
            "DESCRIPTION",
            "The Arxiv RAG ."
        )
        self.PROJECT_SUMMARY = os.getenv(
            "PROJECT_SUMMARY",
            "An AI-powered research assistant for exploring Arxiv papers."
        )

        self.azure_openai_41_endpoint = os.getenv("azure_openai_41_endpoint")
        self.azure_openai_41_api_key = os.getenv("azure_openai_41_api_key")
        self.azure_openai_41_api_version = os.getenv("azure_openai_41_api_version") 

        self.azure_openai_41_mini_endpoint = os.getenv("azure_openai_41_mini_endpoint")
        self.azure_openai_41_mini_api_key = os.getenv("azure_openai_41_mini_api_key")
        self.azure_openai_41_mini_api_version = os.getenv("azure_openai_41_mini_api_version")
        
        self.db_host = os.getenv("HOST")
        self.db_port = os.getenv("PORT")
        self.db_database = os.getenv("DATABASE")
        self.db_user = os.getenv("USER")
        self.db_password = os.getenv("PASSWORD")
        self.db_default_schema = os.getenv("DEFAULT_SCHEMA", "public")

# Create settings instance
settings = Settings()


class Config:
    """Configuration class for the application."""
    
    @staticmethod
    def get_evaluation_config() -> dict:
        """Get evaluation configuration with default values."""
        return {
            "evaluation_model": os.getenv("DEEPEVAL_MODEL", "gpt-4.1"),
            "async_mode": os.getenv("DEEPEVAL_ASYNC", "true").lower() == "true",
            "save_to_db": os.getenv("DEEPEVAL_SAVE_DB", "true").lower() == "true",
            "langsmith_enabled": os.getenv("LANGSMITH_ENABLED", "false").lower() == "true",
            "langfuse_enabled": os.getenv("LANGFUSE_ENABLED", "false").lower() == "true",
            "thresholds": {
                "answer_relevancy_threshold": float(os.getenv("ANSWER_RELEVANCY_THRESHOLD", "0.7")),
                "contextual_precision_threshold": float(os.getenv("CONTEXTUAL_PRECISION_THRESHOLD", "0.7")),
                "contextual_recall_threshold": float(os.getenv("CONTEXTUAL_RECALL_THRESHOLD", "0.7")),
                "contextual_relevancy_threshold": float(os.getenv("CONTEXTUAL_RELEVANCY_THRESHOLD", "0.7")),
                "faithfulness_threshold": float(os.getenv("FAITHFULNESS_THRESHOLD", "0.7")),
                "tool_correctness_threshold": float(os.getenv("TOOL_CORRECTNESS_THRESHOLD", "0.7")),
                "agent_purpose_threshold": float(os.getenv("AGENT_PURPOSE_THRESHOLD", "0.7")),
                "toxicity_threshold": float(os.getenv("TOXICITY_THRESHOLD", "0.5")),
                "bias_threshold": float(os.getenv("BIAS_THRESHOLD", "0.5")),
                "hallucination_threshold": float(os.getenv("HALLUCINATION_THRESHOLD", "0.5")),
                "overall_threshold": float(os.getenv("OVERALL_THRESHOLD", "0.7"))
            }
        }
