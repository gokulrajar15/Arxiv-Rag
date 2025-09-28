import httpx
from app.core.config import settings
from typing import List


class GuardrailsModels:
    """
    A class to interact with Guardrails classification models asynchronously.

    Attributes:
        guardrails_models_auth_token (str): Authorization token for the API.
        topic_classification_endpoint (str): URL for topic detection model.
        bias_classification_endpoint (str): URL for bias detection model.
        toxic_classification_endpoint (str): URL for toxic classification model.
    """

    def __init__(self):
        """
        Initialize the GuardrailsModels instance with endpoints and auth token.
        """
        self.guardrails_models_auth_token = settings.guardrails_auth_token
        self.topic_classification_endpoint = settings.topic_detection_model_url
        self.bias_classification_endpoint = settings.bias_detection_model_url
        self.toxic_classification_endpoint = settings.toxic_classification_model_url

    def _topic_detection_model(self, text: str, topics: list) -> dict:
        """
        Call the topic detection model API to classify text into topics.

        Args:
            text (str): The input text to classify.
            topics (list): List of topics to classify against.

        Returns:
            dict: The API response containing classification results.

        Raises:
            RuntimeError: If there is a network or HTTP error.
        """
        headers = {
            "Authorization": f"Bearer {self.guardrails_models_auth_token}",
            "Content-Type": "application/json"
        }
        payload = {"text": text, "topics": topics}

        try:
            with httpx.Client(timeout=10) as client:
                response = client.post(self.topic_classification_endpoint, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            raise RuntimeError(f"Network error: {e}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP error: {e.response.status_code} - {e.response.text}")

    def _toxic_classification_model(self, texts: str | list) -> dict:
        """
        Call the toxic classification model API to determine if text is toxic or non-toxic.

        Args:
            texts (str | list): The input text(s) to classify.

        Returns:
            dict: The API response containing classification results.

        Raises:
            RuntimeError: If there is a network error.
        """
        headers = {
            "Authorization": f"Bearer {self.guardrails_models_auth_token}",
            "Content-Type": "application/json"
        }
        try:
            with httpx.Client(timeout=10) as client:
                response = client.post(self.toxic_classification_endpoint, headers=headers, json=texts)
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            raise RuntimeError(f"Network error: {e}")

    def _bias_classification_model(self, texts: str | list) -> dict:
        """
        Call the bias classification model API to determine if text is biased or neutral.

        Args:
            texts (str | list): The input text(s) to classify.

        Returns:
            dict: The API response containing classification results.

        Raises:
            RuntimeError: If there is a network error.
        """
        headers = {
            "Authorization": f"Bearer {self.guardrails_models_auth_token}",
            "Content-Type": "application/json"
        }
        try:
            with httpx.Client(timeout=10) as client:
                response = client.post(self.bias_classification_endpoint, headers=headers, json=texts)
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            raise RuntimeError(f"Network error: {e}")
    
        
    def detect_topic(
            self,
            text: str,
            topics: List[str],
            threshold: float = 0.8
    ) -> List[str]:
        """Detect topics in text above the given threshold."""
        try:
            result = self._topic_detection_model(text=text, topics=topics)
            
            if not isinstance(result, dict) or "labels" not in result or "scores" not in result:
                print(f"Unexpected topic detection response format: {result}")
                return []
                
            return [topic
                for topic, score in zip(result["labels"], result["scores"])
                if score > threshold]
        except Exception as e:
            print(f"Error in detect_topic: {e}")
            return []
    
    def detect_bias(
            self,
            text: str,
            threshold: float = 0.8
    ) -> List[str]:
        """Detect bias in text above the given threshold."""
        try:
            result = self._bias_classification_model(texts=text)
            
            if not isinstance(result, list) or not result:
                print(f"Unexpected bias detection response format: {result}")
                return []
                
            return [
                item["label"]
                for item in result
                if item["score"] > threshold and item["label"] == 'BIASED'
            ]
        except Exception as e:
            print(f"Error in detect_bias: {e}")
            return []
    
    def detect_toxic(
            self,
            text: str,
            threshold: float = 0.8
    ) -> List[str]:
        """Detect toxicity in text above the given threshold."""
        try:
            result = self._toxic_classification_model(texts=text)
            
            if not isinstance(result, list) or not result:
                print(f"Unexpected toxic detection response format: {result}")
                return []
                
            return [
                item["label"]
                for item in result
                if item["score"] > threshold and item["label"] == 'toxic'
            ]
        except Exception as e:
            print(f"Error in detect_toxic: {e}")
            return []


# Example usage
if __name__ == "__main__":
    def main():
        guard_models = GuardrailsModels()
        print("---Topic classification model---")
        topics = ["politics", "sports", "technology"]
        text = "This is a text about politics."

        prediction = guard_models.detect_topic(text, topics)
        print(prediction)

        print("---Toxic classification model---")

        text = "This is a text about politics."

        prediction = guard_models.detect_toxic(text)
        print(prediction)

        print("---Bias classification model---")

        text = "This is a text about politics."

        prediction = guard_models.detect_bias(text)
        print(prediction)

    main()
