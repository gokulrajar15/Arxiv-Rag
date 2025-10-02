import asyncio
import json
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
    BiasMetric,
    ToxicityMetric,
    HallucinationMetric
)
from typing import List, Dict, Any
from deepeval.models import AzureOpenAIModel
from deepeval import evaluate
from app.db.client import Database
from app.core.config import settings


def _to_text_context(retrieved_docs: List[Any]) -> List[str]:
    """
    Convert your retrieved docs to strings for DeepEval.
    Handles both dict format and Document objects with metadata and page_content.
    """
    if not retrieved_docs:
        return []
    
    out = []
    
    # Flatten if it's a nested list
    docs_to_process = retrieved_docs
    if retrieved_docs and isinstance(retrieved_docs[0], list):
        docs_to_process = retrieved_docs[0]
    
    for d in docs_to_process:
        text = ""
        
        # Handle Document objects (with metadata and page_content attributes)
        if hasattr(d, 'metadata') and hasattr(d, 'page_content'):
            title = d.metadata.get("title", "") if d.metadata else ""
            category = d.metadata.get("category", "") if d.metadata else ""
            abstract = d.page_content or ""
            text = abstract or f"{title} [{category}]"
        
        # Handle dictionary format
        elif isinstance(d, dict):
            abstract = d.get("abstract") or ""
            title = d.get("title") or ""
            cat = d.get("category") or ""
            text = abstract or f"{title} [{cat}]"
        
        if text:
            out.append(text)
    
    return out


async def _insert_metrics_async(user_query: str, agent_output: str, results: Dict[str, Any]) -> None:
    """Async helper to insert metrics (assumes DB is already initialized)."""
    try:
        json_list = []
        for test in results:
            test_dict = {
                "name": test.name,
                "success": test.success,
                "metrics_data": []
            }
            for metric in test.metrics_data:
                metric_dict = {
                    "name": metric.name,
                    "threshold": metric.threshold,
                    "success": metric.success,
                    "score": metric.score,
                    "reason": metric.reason
                }
                test_dict["metrics_data"].append(metric_dict)
            json_list.append(test_dict)
 
        await Database.execute(
            """
            INSERT INTO metrics (query, output, metrics)
            VALUES ($1, $2, $3::jsonb)
            """,
            user_query,
            agent_output,
            json.dumps(json_list),
        )
    except Exception as e:
        print(f"❌ Error inserting metrics into DB: {e}")
        raise


async def run_deep_eval(user_query: str,
                             agent_output: str,
                             retrieved_docs: List[Any],
                             tools_used: List[str] | None = None) -> Dict[str, Dict[str, Any]]:
    """
    Run DeepEval metrics asynchronously and return the results.
    """
    try:
        print("⚡ DeepEval started...")

        azure_openai = AzureOpenAIModel(
            model_name=settings.azure_openai_41_mini_deployment_name,
            deployment_name=settings.azure_openai_41_mini_deployment_name,
            azure_openai_api_key=settings.azure_openai_41_mini_api_key,
            openai_api_version=settings.azure_openai_41_mini_api_version,
            azure_endpoint=settings.azure_openai_41_mini_endpoint,
            temperature=0
        )

        tool_calls = [ToolCall(name=name) for name in (tools_used or [])]

        rag_context = _to_text_context(retrieved_docs)

        test_case = [LLMTestCase(
            input=user_query,
            actual_output=agent_output,
            context=rag_context,
            retrieval_context=rag_context,
            tools_called=tool_calls
        )]

        metrics = [
            # Response generation metrics
            AnswerRelevancyMetric(model=azure_openai),
            FaithfulnessMetric(model=azure_openai),
            # Retrieval metrics
            ContextualRelevancyMetric(model=azure_openai),
            # Task completion metrics
            TaskCompletionMetric(model=azure_openai),
            # ToolCorrectnessMetric(),
            # Safety metrics
            BiasMetric(model=azure_openai),
            ToxicityMetric(model=azure_openai),
            HallucinationMetric(model=azure_openai),
        ]
        results = evaluate(test_cases=test_case, metrics=metrics)
        await _insert_metrics_async(user_query, agent_output, results.test_results)
        return results

    except Exception as e:
        print(f"❌ Error in deepeval running: {e}")
        raise
