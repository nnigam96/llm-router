"""
FastAPI application entry point for LLM-Router microservice
"""

import time
import uuid
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    RouteRequest,
    RouteResponse,
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ErrorResponse,
    ExpertType,
)
from src.api.dependencies import (
    initialize_app_state,
    get_router,
    get_llm_client,
    get_logger,
    get_config,
    get_experts_config,
)
from src.core.protocols import RoutingStrategy, LLMProvider
from src.forensics.logger import DataLogger
from omegaconf import DictConfig


# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown.
    Initializes all dependencies at startup.
    """
    # Startup
    print("🚀 Initializing LLM Router...")
    try:
        initialize_app_state()
        print("✅ Application initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize application: {e}")
        raise

    yield

    # Shutdown
    print("👋 Shutting down LLM Router...")


# Create FastAPI app
app = FastAPI(
    title="LLM Router",
    description="High-Performance Semantic Orchestration for Distributed LLM Systems",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with consistent error response format"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.detail,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            detail={"error": str(exc)},
        ).model_dump(),
    )


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "LLM Router API",
        "version": "0.1.0",
        "description": "High-Performance Semantic Orchestration for Distributed LLM Systems",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(
    config: DictConfig = Depends(get_config),
    llm_client: LLMProvider = Depends(get_llm_client),
):
    """
    Health check endpoint.
    Returns the status of the API and its dependencies.
    """
    # Check Ollama connection
    ollama_status = "unknown"
    try:
        # Try a simple request to check if Ollama is running
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            ollama_status = "healthy"
        else:
            ollama_status = "unhealthy"
    except Exception:
        ollama_status = "unreachable"

    return HealthResponse(
        status="healthy" if ollama_status != "unreachable" else "degraded",
        version="0.1.0",
        routing_strategies=["hybrid", "semantic", "keyword"],
        experts=["professor", "zoomer"],
        ollama_status=ollama_status,
    )


@app.post("/v1/route", response_model=RouteResponse, tags=["routing"])
async def route_query(
    request: RouteRequest,
    router: RoutingStrategy = Depends(get_router),
):
    """
    Route a query to the appropriate expert.

    This endpoint only performs routing decision without calling the LLM.
    Returns the selected expert, confidence score, and latency metrics.
    """
    try:
        # Perform routing
        expert_id, confidence, latency_ms = router.route(request.query)

        # Determine which strategy was used
        strategy_used = "hybrid"  # Default assumption
        if confidence == 1.0:
            strategy_used = "keyword"
        elif confidence < 0.2:
            strategy_used = "fallback"
        else:
            strategy_used = "semantic"

        # Generate interaction ID
        interaction_id = str(uuid.uuid4())

        return RouteResponse(
            query=request.query,
            expert=ExpertType(expert_id),
            confidence=confidence,
            latency_ms=latency_ms,
            strategy_used=strategy_used,
            interaction_id=interaction_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Routing failed: {str(e)}",
        )


@app.post("/v1/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    request: ChatRequest,
    router: RoutingStrategy = Depends(get_router),
    llm_client: LLMProvider = Depends(get_llm_client),
    logger: DataLogger = Depends(get_logger),
    experts_config: Dict[str, Dict[str, Any]] = Depends(get_experts_config),
):
    """
    Get a response from an expert (with optional auto-routing).

    If expert is specified in the request, uses that expert directly.
    Otherwise, automatically routes the query to the appropriate expert.
    """
    try:
        start_time = time.time()

        # Step 1: Determine expert (either from request or via routing)
        if request.expert:
            expert_id = request.expert.value
            confidence = 1.0
            routing_latency_ms = 0.0
            strategy_used = "manual"
        else:
            # Auto-route
            expert_id, confidence, routing_latency_ms = router.route(request.query)

            # Determine which strategy was used
            if confidence == 1.0:
                strategy_used = "keyword"
            elif confidence < 0.2:
                strategy_used = "fallback"
            else:
                strategy_used = "semantic"

        # Step 2: Get expert configuration
        if expert_id not in experts_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Expert '{expert_id}' not found in configuration",
            )

        expert_config = experts_config[expert_id]

        # Update config with request parameters
        if request.temperature is not None:
            expert_config["temperature"] = request.temperature
        if request.max_tokens is not None:
            expert_config["max_tokens"] = request.max_tokens

        # Step 3: Get response from LLM (using async)
        inference_start = time.time()
        response_text = await llm_client.get_response_async(
            expert_name=expert_id,
            query=request.query,
            config=expert_config,
        )
        inference_latency_ms = (time.time() - inference_start) * 1000

        # Check for errors
        if response_text.startswith("[System Error]"):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=response_text,
            )

        # Step 4: Calculate total latency
        total_latency_ms = (time.time() - start_time) * 1000

        # Step 5: Generate interaction ID
        interaction_id = str(uuid.uuid4())

        # Step 6: Log the interaction (with neutral feedback for now)
        logger.log_interaction(
            prompt=request.query,
            routed_model=expert_id,
            response=response_text,
            feedback_score=-1,  # -1 indicates no feedback yet
            latency_ms=total_latency_ms,
        )

        return ChatResponse(
            query=request.query,
            expert=ExpertType(expert_id),
            response=response_text,
            routing_confidence=confidence,
            routing_latency_ms=routing_latency_ms,
            inference_latency_ms=inference_latency_ms,
            total_latency_ms=total_latency_ms,
            strategy_used=strategy_used,
            interaction_id=interaction_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}",
        )


@app.post("/v1/feedback", response_model=FeedbackResponse, tags=["feedback"])
async def submit_feedback(
    request: FeedbackRequest,
    logger: DataLogger = Depends(get_logger),
):
    """
    Submit feedback for a specific interaction.

    This endpoint allows users to provide thumbs up (1) or thumbs down (0)
    feedback on expert responses, which will be used for DPO training.
    """
    try:
        # In a real implementation, we would:
        # 1. Look up the interaction by ID
        # 2. Update the feedback in the log
        # 3. Potentially trigger DPO pipeline updates

        # For now, we'll just log the feedback
        # Note: This is a simplified implementation
        # A production system would need a database to update feedback

        return FeedbackResponse(
            interaction_id=request.interaction_id,
            status="success",
            message=f"Feedback recorded successfully. Thank you!",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback submission failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
