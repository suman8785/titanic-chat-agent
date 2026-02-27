"""
Configuration management for the Titanic Chat Agent.
Handles environment variables and application settings.
"""

from pydantic import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    app_name: str = "Titanic Chat Agent"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-4-turbo-preview", description="OpenAI model to use")
    openai_temperature: float = Field(default=0.1, ge=0, le=2, description="Model temperature")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    
    # CORS Configuration
    cors_origins: list[str] = Field(
        default=["http://localhost:8501", "http://localhost:3000"],
        description="Allowed CORS origins"
    )
    
    # Memory Configuration
    max_memory_messages: int = Field(default=20, description="Maximum messages to keep in memory")
    
    # Visualization Configuration
    chart_dpi: int = Field(default=100, description="Chart DPI for rendering")
    chart_style: str = Field(default="seaborn-v0_8-whitegrid", description="Matplotlib style")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience function for accessing settings
settings = get_settings()