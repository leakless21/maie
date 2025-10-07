"""Route definitions for the MAIE API."""

from typing import List
from litestar import Controller, get, post
from litestar.params import Parameter

from src.api.dependencies import api_key_auth
from src.api.schemas import (
    ProcessRequestSchema,
    StatusResponseSchema,
    ModelsResponseSchema,
    TemplatesResponseSchema
)


class ProcessController(Controller):
    """Controller for processing endpoints."""
    
    @post(
        "/v1/process",
        summary="Process audio file",
        description="Submit an audio file for processing using specified parameters",
        tags=["Processing"]
    )
    async def process_audio(
        self,
        data: ProcessRequestSchema
    ) -> StatusResponseSchema:
        """
        Process an audio file with the specified parameters.
        
        Args:
            data: Processing request parameters
            
        Returns:
            Status response with task ID and initial status
        """
        # This is a skeleton implementation - actual processing logic
        # would be implemented in future iterations
        pass


class StatusController(Controller):
    """Controller for status checking endpoints."""
    
    @get(
        "/v1/status/{task_id:uuid}",
        summary="Get processing status",
        description="Check the status of a processing task by its ID",
        tags=["Status"]
    )
    async def get_status(
        self,
        task_id: str = Parameter(
            ..., 
            description="UUID of the processing task"
        )
    ) -> StatusResponseSchema:
        """
        Get the current status of a processing task.
        
        Args:
            task_id: Unique identifier of the processing task
            
        Returns:
            Status information for the specified task
        """
        # This is a skeleton implementation - actual status checking logic
        # would be implemented in future iterations
        pass


class ModelsController(Controller):
    """Controller for models endpoints."""
    
    @get(
        "/v1/models",
        summary="Get available models",
        description="Retrieve a list of available audio processing models",
        tags=["Models"]
    )
    async def get_models(self) -> ModelsResponseSchema:
        """
        Get a list of available audio processing models.
        
        Returns:
            List of available models with their information
        """
        # This is a skeleton implementation - actual model listing logic
        # would be implemented in future iterations
        pass


class TemplatesController(Controller):
    """Controller for templates endpoints."""
    
    @get(
        "/v1/templates",
        summary="Get available templates",
        description="Retrieve a list of available processing templates",
        tags=["Templates"]
    )
    async def get_templates(self) -> TemplatesResponseSchema:
        """
        Get a list of available processing templates.
        
        Returns:
            List of available templates with their information
        """
        # This is a skeleton implementation - actual template listing logic
        # would be implemented in future iterations
        pass


# Define route handlers for the app
route_handlers: List = [
    ProcessController,
    StatusController,
    ModelsController,
    TemplatesController,
]