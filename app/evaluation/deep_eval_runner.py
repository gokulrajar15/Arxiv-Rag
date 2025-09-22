import os
import threading
import asyncio
from deepeval import evaluate
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
from app.db.client import Database
from app.core.config import settings


def _to_text_context(retrieved_docs: List[Dict[str, Any]]) -> List[str]:
    """
    Convert your retrieved docs to strings for DeepEval.
    Expecting items like {"title":..., "category":..., "abstract":...}
    """
    if not retrieved_docs:
        return []
    out = []
    for d in retrieved_docs:
        abstract = d.get("abstract") or ""
        title = d.get("title") or ""
        cat = d.get("category") or ""
        text = abstract or f"{title} [{cat}]"
        if text:
            out.append(text)
    return out


async def _insert_metrics_async(user_query: str, agent_output: str, results: Dict[str, Any]) -> None:
    """Async helper to init DB and insert metrics."""
    await Database.init()
    await Database.execute(
        """
        INSERT INTO metrics (query, output, metrics)
        VALUES ($1, $2, $3::jsonb)
        """,
        user_query,
        agent_output,
        results,
    )


def run_deep_eval(user_query: str,
                  agent_output: str,
                  retrieved_docs: List[Dict[str, Any]],
                  tools_used: List[str] | None = None):
    """
    Run DeepEval metrics in a background thread and insert into DB.
    """

    def task():
        try:
            print("⚡ DeepEval started...")


            azure_openai = AzureOpenAIModel(
                model_name="gpt-4.1", 
                deployment_name="gpt-4.1", 
                azure_openai_api_key=settings.azure_openai_41_api_key,
                openai_api_version=settings.azure_openai_41_api_version,
                azure_endpoint=settings.azure_openai_41_endpoint,
                temperature=0
        )

            tool_calls = [ToolCall(name=name) for name in (tools_used or [])]

            rag_context = _to_text_context(retrieved_docs)
            test_case = LLMTestCase(
                input=user_query,
                actual_output=agent_output,
                context=rag_context,
                retrieval_context=rag_context,
                tools_called=tool_calls,
                expected_tools=tool_calls or [ToolCall(name="document_retriever")],
            )

            metrics = [
                AnswerRelevancyMetric(model=azure_openai),
                FaithfulnessMetric(model=azure_openai),
                ContextualRelevancyMetric(model=azure_openai),
                TaskCompletionMetric(model=azure_openai),
                ToolCorrectnessMetric(),
                BiasMetric(model=azure_openai),
                ToxicityMetric(model=azure_openai),
                HallucinationMetric(model=azure_openai),
            ]


            # deep_evaluate = evaluate(test_case=[test_case], metrics=metrics)
            # Measure individually to capture scores + reasons for DB
            results: Dict[str, Dict[str, Any]] = {}
            for m in metrics:
                m.measure(test_case) 
                results[m.__class__.__name__] = {
                    "score": getattr(m, "score", None),
                    "reason": getattr(m, "reason", None),
                }

            print(f"✅ DeepEval finished. Results: {results}")

            asyncio.run(_insert_metrics_async(user_query, agent_output, results))
            print("✅ DeepEval metrics inserted into DB")

        except Exception as e:
            print("Error in deepeval running")

    threading.Thread(target=task, daemon=True).start()