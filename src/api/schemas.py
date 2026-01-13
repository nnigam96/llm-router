"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ExpertType(str, Enum):
    """Available expert types"""
    PROFESSOR = "professor"
    ZOOMER = "zoomer"


class RouteRequest(BaseModel):
    """Request schema for routing a query to an expert"""
    query: str = Field(..., min_length=1, max_length=2000, description="The user query to route")
    user_id: Optional[str] = Field(None, description="Optional user ID for tracking")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and sanitize query"""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty or whitespace only")
        return v


class RouteResponse(BaseModel):
    """Response schema for routing decision"""
    query: str = Field(..., description="The original query")
    expert: ExpertType = Field(..., description="The selected expert")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Routing confidence score")
    latency_ms: float = Field(..., ge=0.0, description="Routing latency in milliseconds")
    strategy_used: str = Field(..., description="The strategy that made the routing decision (keyword/semantic/fallback)")
    interaction_id: str = Field(..., description="Unique ID for this interaction")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the routing decision")


class ChatRequest(BaseModel):
    """Request schema for getting a response from an expert"""
    query: str = Field(..., min_length=1, max_length=2000, description="The user query")
    expert: Optional[ExpertType] = Field(None, description="Specific expert to use (if None, will auto-route)")
    user_id: Optional[str] = Field(None, description="Optional user ID for tracking")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: Optional[int] = Field(512, ge=1, le=4096, description="Maximum tokens to generate")

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and sanitize query"""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty or whitespace only")
        return v


class ChatResponse(BaseModel):
    """Response schema for expert chat response"""
    query: str = Field(..., description="The original query")
    expert: ExpertType = Field(..., description="The expert that generated the response")
    response: str = Field(..., description="The expert's response")
    routing_confidence: float = Field(..., ge=0.0, le=1.0, description="Routing confidence score")
    routing_latency_ms: float = Field(..., ge=0.0, description="Routing latency in milliseconds")
    inference_latency_ms: float = Field(..., ge=0.0, description="Inference latency in milliseconds")
    total_latency_ms: float = Field(..., ge=0.0, description="Total latency in milliseconds")
    strategy_used: str = Field(..., description="The strategy that made the routing decision")
    interaction_id: str = Field(..., description="Unique ID for this interaction")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the response")


class FeedbackRequest(BaseModel):
    """Request schema for providing feedback on a response"""
    interaction_id: str = Field(..., description="The interaction ID to provide feedback for")
    feedback: int = Field(..., ge=0, le=1, description="Feedback: 1 for positive, 0 for negative")
    comment: Optional[str] = Field(None, max_length=500, description="Optional comment")


class FeedbackResponse(BaseModel):
    """Response schema for feedback submission"""
    interaction_id: str = Field(..., description="The interaction ID")
    status: str = Field(..., description="Status of the feedback submission")
    message: str = Field(..., description="Human-readable message")


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str = Field(..., description="Health status (healthy/unhealthy)")
    version: str = Field(..., description="API version")
    routing_strategies: list[str] = Field(..., description="Available routing strategies")
    experts: list[str] = Field(..., description="Available experts")
    ollama_status: str = Field(..., description="Ollama connection status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of health check")


class ErrorResponse(BaseModel):
    """Response schema for errors"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the error")
