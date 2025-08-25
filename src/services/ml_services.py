"""
Machine Learning Service Module.

This module encapsulates all the logic related to the machine learning model.
It follows a singleton pattern by creating a single global instance of the
MLService class.

Responsibilities:
- Loading the fine-tuned transformer model and tokenizer from disk.
- Providing a clean `predict` method to perform inference on raw text.
- Managing the model's device placement (CPU/GPU).

This separation of concerns keeps the ML logic isolated from the API/web layer,
making the application more modular, testable, and maintainable.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, model_path: Path):
        """Loads the model and tokenizer from the specified path."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found at {model_path}")
        
        logger.info(f"Loading model and tokenizer from: {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Model and tokenizer loaded successfully.")

    def predict(self, text: str) -> tuple[str, float]:
        """Performs a prediction on the given text."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded. Call the load() method first.")

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        confidence_score, predicted_class_id = torch.max(probabilities, dim=1)
        predicted_category = self.model.config.id2label[predicted_class_id.item()]
        
        return predicted_category, confidence_score.item()

# Create a single, global instance of the service to be used across the application
ml_service = MLService()
