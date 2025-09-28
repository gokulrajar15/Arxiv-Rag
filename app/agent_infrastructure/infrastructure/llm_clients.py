# Third-party imports
from langchain_openai import AzureChatOpenAI

from app.core.config import settings


gpt_41 = AzureChatOpenAI(
    azure_deployment=settings.azure_openai_41_deployment_name,
    api_version=settings.azure_openai_41_api_version,
    azure_endpoint=settings.azure_openai_41_endpoint,
    api_key=settings.azure_openai_41_api_key
)

gpt_41_mini = AzureChatOpenAI(
    azure_deployment=settings.azure_openai_41_mini_deployment_name,
    api_version=settings.azure_openai_41_mini_api_version,
    azure_endpoint=settings.azure_openai_41_mini_endpoint,
    api_key=settings.azure_openai_41_mini_api_key
)
