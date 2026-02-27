"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Enum for message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class VisualizationType(str, Enum):
    """Supported visualization types."""
    HISTOGRAM = "histogram"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    BOX_PLOT = "box_plot"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    LINE_CHART = "line_chart"


class ChatMessage(BaseModel):
    """Schema for a chat message."""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChatRequest(BaseModel):
    """Schema for incoming chat requests."""
    message: str = Field(..., min_length=1, max_length=2000, description="User's message")
    session_id: str = Field(..., description="Unique session identifier")
    include_reasoning: bool = Field(default=False, description="Include agent reasoning in response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What was the survival rate for women?",
                "session_id": "user-123-session-456",
                "include_reasoning": True
            }
        }


class VisualizationData(BaseModel):
    """Schema for visualization data."""
    chart_type: VisualizationType
    title: str
    image_base64: str
    description: str
    data_summary: Optional[dict] = None


class ChatResponse(BaseModel):
    """Schema for chat responses."""
    message: str = Field(..., description="Agent's text response")
    visualizations: list[VisualizationData] = Field(
        default_factory=list, 
        description="Generated visualizations"
    )
    reasoning: Optional[str] = Field(None, description="Agent's reasoning process")
    suggested_questions: list[str] = Field(
        default_factory=list, 
        description="Follow-up question suggestions"
    )
    execution_time: float = Field(..., description="Response generation time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "The survival rate for women was approximately 74.2%.",
                "visualizations": [],
                "reasoning": "Used pandas to filter by Sex=='female' and calculated mean of Survived column.",
                "suggested_questions": [
                    "How does this compare to men?",
                    "What about survival by class?"
                ],
                "execution_time": 1.23
            }
        }


class DatasetStats(BaseModel):
    """Schema for dataset statistics."""
    total_passengers: int
    total_survived: int
    survival_rate: float
    columns: list[str]
    missing_values: dict[str, int]
    class_distribution: dict[str, int]
    gender_distribution: dict[str, int]
    age_stats: dict[str, float]
    fare_stats: dict[str, float]


class SessionInfo(BaseModel):
    """Schema for session information."""
    session_id: str
    created_at: datetime
    message_count: int
    last_activity: datetime


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: Literal["healthy", "unhealthy"]
    version: str
    timestamp: datetime
    components: dict[str, bool]