"""Main Litestar application entry point for the Modular Audio Intelligence Engine (MAIE) API."""

from litestar import Litestar
from litestar.openapi import OpenAPIConfig
from litestar.openapi.spec import Info

from src.api.routes import route_handlers


# OpenAPI configuration
openapi_config = OpenAPIConfig(
    title="Modular Audio Intelligence Engine (MAIE) API",
    version="1.0.0",
    info=Info(
        title="Modular Audio Intelligence Engine (MAIE) API",
        version="1.0.0",
        description="API for the Modular Audio Intelligence Engine providing audio processing capabilities",
    ),
)


# Create the Litestar app instance
app = Litestar(
    route_handlers=route_handlers,
    openapi_config=openapi_config,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0", port=8000)