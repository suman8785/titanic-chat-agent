"""
FastAPI backend for the Titanic Chat Agent.
Provides REST API endpoints for the chat interface.
"""

import logging
import time
import math
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from backend.config import settings
from backend.schemas import (
    ChatRequest, 
    ChatResponse, 
    DatasetStats, 
    HealthResponse,
    SessionInfo
)
from backend.agent import get_agent, TitanicChatAgent
from backend.data_loader import get_data_loader
from backend.memory import memory_manager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize an object for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting Titanic Chat Agent API...")
    
    # Pre-load data
    logger.info("📊 Loading Titanic dataset...")
    get_data_loader()
    
    logger.info("✅ Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Titanic Chat Agent API...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="""
    A production-ready conversational AI agent for exploring the Titanic dataset.
    
    ## Features
    - Natural language queries about the Titanic dataset
    - Automatic visualization generation
    - Statistical analysis
    - Conversation memory
    - Intelligent insights
    
    ## Usage
    Send POST requests to `/api/chat` with your questions about the Titanic dataset.
    """,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================
# Dependency Injection
# ===========================================

def get_agent_dependency() -> TitanicChatAgent:
    """Dependency for getting the agent instance."""
    return get_agent()


# ===========================================
# API Endpoints
# ===========================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Titanic Dataset Chat Agent API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        # Check data loader
        loader = get_data_loader()
        data_healthy = loader.dataframe is not None and len(loader.dataframe) > 0
        
        return HealthResponse(
            status="healthy" if data_healthy else "unhealthy",
            version=settings.app_version,
            timestamp=datetime.utcnow(),
            components={
                "data_loader": data_healthy,
                "memory_manager": True
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version=settings.app_version,
            timestamp=datetime.utcnow(),
            components={
                "data_loader": False,
                "memory_manager": False
            }
        )


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    agent: TitanicChatAgent = Depends(get_agent_dependency)
):
    """
    Process a chat message and return the agent's response.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Chat request from session {request.session_id}: {request.message[:100]}...")
        
        # Process the message
        response_text, visualizations, reasoning, suggested_questions = await agent.chat(
            message=request.message,
            session_id=request.session_id,
            include_reasoning=request.include_reasoning
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Build response
        response = ChatResponse(
            message=response_text,
            visualizations=visualizations,
            reasoning=reasoning,
            suggested_questions=suggested_questions,
            execution_time=round(execution_time, 3)
        )
        
        logger.info(f"Chat response generated in {execution_time:.2f}s with {len(visualizations)} visualizations")
        
        return response
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}"
        )


@app.get("/api/dataset/stats", response_model=DatasetStats, tags=["Dataset"])
async def get_dataset_stats():
    """Get comprehensive statistics about the Titanic dataset."""
    try:
        loader = get_data_loader()
        stats = loader.get_statistics()
        
        # Sanitize the stats for JSON
        sanitized_stats = sanitize_for_json(stats)
        
        return DatasetStats(**sanitized_stats)
        
    except Exception as e:
        logger.error(f"Dataset stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dataset/columns", tags=["Dataset"])
async def get_dataset_columns():
    """Get information about all columns in the dataset."""
    try:
        loader = get_data_loader()
        column_info = loader.get_column_info()
        return sanitize_for_json(column_info)
        
    except Exception as e:
        logger.error(f"Dataset columns error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dataset/sample", tags=["Dataset"])
async def get_dataset_sample(n: int = 5):
    """Get a sample of rows from the dataset."""
    try:
        loader = get_data_loader()
        df = loader.dataframe.sample(n=min(n, 100))
        
        # Convert to dict and handle NaN values
        sample_data = df.to_dict(orient='records')
        sanitized_sample = sanitize_for_json(sample_data)
        
        return {"sample": sanitized_sample, "total_rows": len(loader.dataframe)}
        
    except Exception as e:
        logger.error(f"Dataset sample error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}", response_model=SessionInfo, tags=["Session"])
async def get_session_info(session_id: str):
    """Get information about a specific session."""
    try:
        info = memory_manager.get_session_info(session_id)
        return SessionInfo(
            session_id=info["session_id"],
            created_at=datetime.fromisoformat(info["created_at"]),
            message_count=info["message_count"],
            last_activity=datetime.fromisoformat(info["last_activity"])
        )
    except Exception as e:
        logger.error(f"Session info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}/history", tags=["Session"])
async def get_session_history(session_id: str):
    """Get the chat history for a specific session."""
    try:
        history = memory_manager.get_chat_history_as_text(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"Session history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/session/{session_id}", tags=["Session"])
async def clear_session(session_id: str):
    """Clear the chat history for a specific session."""
    try:
        memory_manager.clear_session(session_id)
        return {"status": "success", "message": f"Session {session_id} cleared"}
    except Exception as e:
        logger.error(f"Clear session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions/cleanup", tags=["Session"])
async def cleanup_sessions(max_age_hours: int = 24, background_tasks: BackgroundTasks = None):
    """Clean up inactive sessions."""
    try:
        removed = memory_manager.cleanup_inactive_sessions(max_age_hours)
        return {"status": "success", "sessions_removed": removed}
    except Exception as e:
        logger.error(f"Cleanup sessions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================
# Error Handlers
# ===========================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred",
            "error_type": type(exc).__name__
        }
    )


# ===========================================
# Main Entry Point
# ===========================================

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info"
    )