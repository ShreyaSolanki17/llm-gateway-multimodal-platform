import base64
import binascii
import re
import time
import uuid
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5MB decoded image cap
DATA_URI_PATTERN = re.compile(r"^data:image/(png|jpeg|jpg|gif|webp);base64,(.+)$", re.IGNORECASE | re.DOTALL)


class ImageURL(BaseModel):
    url: str = Field(..., description="Either an http(s) URL or a data:image/...;base64,... URI")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if v.startswith("http://") or v.startswith("https://"):
            return v

        match = DATA_URI_PATTERN.match(v)
        if not match:
            raise ValueError("image_url.url must be an http(s) URL or a data:image/...;base64,... URI")

        try:
            decoded = base64.b64decode(match.group(2), validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("image_url.url contains invalid base64 image data")

        if len(decoded) > MAX_IMAGE_BYTES:
            raise ValueError(f"image exceeds maximum size of {MAX_IMAGE_BYTES} bytes")

        return v


class ContentPart(BaseModel):
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (e.g. system, user, assistant)")
    content: Union[str, List[ContentPart]] = Field(
        ..., description="Message content: plain text, or a list of text/image parts"
    )

    def get_text(self) -> str:
        """Extract the text portion of this message, ignoring any image parts."""
        if isinstance(self.content, str):
            return self.content
        return " ".join(part.text for part in self.content if part.type == "text" and part.text)

    def has_image(self) -> bool:
        """Whether this message carries at least one image part."""
        if isinstance(self.content, str):
            return False
        return any(part.type == "image_url" for part in self.content)


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="default-model", description="ID of the model to use")
    messages: List[ChatMessage] = Field(..., min_length=1, description="List of messages in the conversation")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=1000, gt=0, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(default=False, description="Whether to stream back partial responses")
    use_rag: Optional[bool] = Field(default=False, description="Augment the prompt with retrieved context from ingested documents")


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
