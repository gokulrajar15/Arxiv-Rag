from typing import List, Optional
from app.agent_infrastructure.guardrails.guardrails_models import GuardrailsModels
from guardrails import AsyncGuard, OnFailAction
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
import asyncio

@register_validator(name="constrain_topic", data_type="string")
class ConstrainTopic(Validator):
    def __init__(
        self,
        banned_topics: Optional[list[str]] = ["politics"],
        threshold: float = 0.8,
        guard_models: Optional[GuardrailsModels] = None,
        **kwargs
    ):
        self.topics = banned_topics
        self.threshold = threshold
        self.guard_models = guard_models or GuardrailsModels()
        super().__init__(**kwargs)

    def _validate(
        self, value: str, metadata: Optional[dict[str, str]] = None
    ) -> ValidationResult:
        detected_topics = self.guard_models.detect_topic(
            value, self.topics, self.threshold
        )
        if detected_topics:
            return FailResult(
                error_message=f"Sorry I cannot assist with that request as it contains the following banned topics: {detected_topics}"
            )
        return PassResult()

@register_validator(name="constrain_bias", data_type="string")
class ConstrainBias(Validator):
    def __init__(
        self,
        threshold: float = 0.8,
        guard_models: Optional[GuardrailsModels] = None,
        **kwargs
    ):
        self.threshold = threshold
        self.guard_models = guard_models or GuardrailsModels()
        super().__init__(**kwargs)

    def _validate(
        self, value: str, metadata: Optional[dict[str, str]] = None
    ) -> ValidationResult:
        detected_bias = self.guard_models.detect_bias(
            value, self.threshold
        )
        if detected_bias:
            return FailResult(
                error_message=f"Sorry I cannot assist with that request as it contains the following banned bias: {detected_bias}"
            )
        return PassResult()

@register_validator(name="constrain_toxic", data_type="string")
class ConstrainToxic(Validator):
    def __init__(
        self,
        threshold: float = 0.8,
        guard_models: Optional[GuardrailsModels] = None,
        **kwargs
    ):
        self.threshold = threshold
        self.guard_models = guard_models or GuardrailsModels()
        super().__init__(**kwargs)

    def _validate(
        self, value: str, metadata: Optional[dict[str, str]] = None
    ) -> ValidationResult:
        detected_toxic = self.guard_models.detect_toxic(
            value, self.threshold
        )
        if detected_toxic:
            return FailResult(
                error_message=f"Sorry I cannot assist with that request as it contains the following banned toxic: {detected_toxic}"
            )
        return PassResult()
 

guard = AsyncGuard(name='topic_guard').use_many(
    ConstrainTopic(
        banned_topics=["nudity", "violence", "adult content", "illegal", "hate speech", "offensive"],
        on_fail=OnFailAction.NOOP,
    ),
    ConstrainBias(
        on_fail=OnFailAction.NOOP,
    ),
    ConstrainToxic(
        on_fail=OnFailAction.NOOP,
    )
)


async def guardrails_validator(text: str):
    """Validate text using guardrails, it'll detect topics, bias, and toxicity."""
    try:
        result = await guard.validate(text)
        return result.to_dict()
    except Exception as e:
        print(f"Validation failed: {e}")
        return {"validationPassed": True, "error": str(e)}

async def main():
    test_texts = [
        "This is a text about politics.",
        "This is a neutral text about technology.",
        "This text contains hate speech and offensive content.",
        "This is a biased text favoring one side.",
        "This is a toxic text with harmful language."
    ]
    
    for text in test_texts:
        print(f"Testing text: {text}")
        result = await guardrails_validator(text)
        print(f"Validation result: {result}\n")

if __name__ == "__main__":
    asyncio.run(main())
