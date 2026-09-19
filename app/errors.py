"""Domain exceptions and their HTTP translations.

Services raise these; routers never build error responses by hand.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


class ServiceError(Exception):
    """Base class for expected, client-reportable failures."""

    status_code: int = status.HTTP_400_BAD_REQUEST
    code: str = "service_error"

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.message = message
        self.context = context


class UnknownTickerError(ServiceError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "unknown_ticker"


class NoMarketDataError(ServiceError):
    status_code = status.HTTP_502_BAD_GATEWAY
    code = "no_market_data"


class InsufficientDataError(ServiceError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "insufficient_data"


class UnknownBackendError(ServiceError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "unknown_backend"


class OptimizationFailedError(ServiceError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "optimization_failed"


class JobNotFoundError(ServiceError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "job_not_found"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def _handle_service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "context": {k: str(v) for k, v in exc.context.items()},
                }
            },
        )
