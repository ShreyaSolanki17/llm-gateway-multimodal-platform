import re
from typing import Tuple
from app.schemas.chat import ChatCompletionRequest
from app.router.schemas import ComplexityLevel, RequestType


class RequestAnalyzer:
    """Rule-based analyzer for request complexity and input classification."""

    COMPLEX_KEYWORDS = [
        "explain step by step",
        "analyze",
        "compare and contrast",
        "refactor",
        "debug",
        "architecture",
        "proof",
        "mathematical",
        "solve",
        "implement",
    ]

    def analyze(self, request: ChatCompletionRequest) -> Tuple[ComplexityLevel, RequestType]:
        # Extract combined text of user messages
        user_texts = [m.content for m in request.messages if m.role == "user"]
        full_text = " ".join(user_texts)

        # 1. Detect Request Type (Vision vs Text)
        # (In Milestone 7 we'll inspect base64/URL images, here we check text indicators)
        is_vision = any("image:" in t.lower() or "[image]" in t.lower() for t in user_texts)
        req_type = RequestType.VISION if is_vision else RequestType.TEXT

        # 2. Analyze Complexity
        # Word count & code block detection
        word_count = len(full_text.split())
        has_code_blocks = "```" in full_text
        has_complex_keywords = any(re.search(r"\b" + re.escape(kw) + r"\b", full_text, re.IGNORECASE) for kw in self.COMPLEX_KEYWORDS)

        if is_vision or word_count > 250 or (has_code_blocks and word_count > 100) or (has_complex_keywords and word_count > 50):
            complexity = ComplexityLevel.COMPLEX
        elif word_count > 60 or has_code_blocks or has_complex_keywords:
            complexity = ComplexityLevel.MODERATE
        else:
            complexity = ComplexityLevel.SIMPLE

        return complexity, req_type
