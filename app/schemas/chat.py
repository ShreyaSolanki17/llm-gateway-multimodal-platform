import time
import uuid
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (e.g. system, user, assistant)")
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="default-model", description="ID of the model to use")
    messages: List[ChatMessage] = Field(..., min_length=1, description="List of messages in the conversation")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=1000, gt=0, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(default=False, description="Whether to stream back partial responses")


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
