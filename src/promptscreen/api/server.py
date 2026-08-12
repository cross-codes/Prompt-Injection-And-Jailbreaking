import logging
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from ..defence.abstract_defence import AbstractDefence

logger = logging.getLogger(__name__)

# Hard cap on request body size, applied before FastAPI/pydantic parses the
# body. Guards like PromptLengthGuard only run *after* the JSON body has
# already been fully parsed into memory, so they can't protect against an
# oversized request on their own.
_DEFAULT_MAX_BODY_BYTES = 1_000_000  # 1 MB


class _MaxBodySizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, max_body_bytes: int) -> None:
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > self.max_body_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": (
                        f"Request body of {content_length} bytes exceeds the "
                        f"{self.max_body_bytes}-byte limit."
                    )
                },
            )
        return await call_next(request)


def create_app(
    guards: dict[str, AbstractDefence],
    *,
    api_key: Optional[str] = None,
    allowed_origins: Optional[list[str]] = None,
    max_body_bytes: int = _DEFAULT_MAX_BODY_BYTES,
) -> FastAPI:
    """Build the evaluation API.

    Parameters
    ----------
    guards:
        Guard instances keyed by CLI-style name, e.g. from
        ``promptscreen.cli.AVAILABLE_GUARDS``.
    api_key:
        If set, every request must include a matching ``X-API-Key`` header
        or receive ``401 Unauthorized``. ``None`` (the default) disables
        auth -- suitable for local/dev use only. Set this (e.g. from an
        environment variable) before exposing the API beyond localhost.
    allowed_origins:
        If set, enables CORS for these origins via ``CORSMiddleware``.
        ``None`` (the default) leaves CORS disabled.
    max_body_bytes:
        Hard cap on request body size, rejected with ``413`` before the
        body is parsed. Defaults to 1 MB.
    """

    class EvaluationRequest(BaseModel):
        prompt: str
        defences: list[str]
        mode: str

    class DefenceResult(BaseModel):
        is_safe: bool
        details: str
        confidence: float = -1.0  # -1 means the guard does not produce a score

    async def _verify_api_key(
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> None:
        if api_key is not None and x_api_key != api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    app = FastAPI(
        title="LLM Defence Suite API",
        description="Evaluates prompts against security defences.",
        dependencies=[Depends(_verify_api_key)],
    )

    app.add_middleware(_MaxBodySizeMiddleware, max_body_bytes=max_body_bytes)

    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    def _run_guard(name: str, prompt: str) -> DefenceResult:
        """Run a single guard, returning a structured result on any error."""
        guard = guards.get(name)
        if not guard:
            return DefenceResult(
                is_safe=False,
                details=f"Error: Defence '{name}' not available.",
            )
        try:
            analysis = guard.analyse(prompt)
            return DefenceResult(
                is_safe=analysis.get_verdict(),
                details=analysis.get_type(),
                confidence=(
                    analysis.confidence if analysis.confidence is not None else -1.0
                ),
            )
        except Exception:
            logger.exception("Guard '%s' raised an unexpected error", name)
            return DefenceResult(
                is_safe=False,
                details=f"Error: Defence '{name}' encountered an internal error.",
            )

    @app.post("/evaluate", response_model=dict[str, DefenceResult])
    async def evaluate_prompt(request: EvaluationRequest) -> dict[str, DefenceResult]:
        mode = request.mode.lower()

        if mode == "separate":
            return {name: _run_guard(name, request.prompt) for name in request.defences}

        elif mode == "chain":
            # Accumulate results so the response shape is always {guard_name: result},
            # regardless of whether a guard fails or all pass.
            chain_results: dict[str, DefenceResult] = {}
            for name in request.defences:
                result = _run_guard(name, request.prompt)
                chain_results[name] = result
                if not result.is_safe:
                    return chain_results  # early-stop; client sees which guard failed
            return chain_results

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode '{request.mode}'. Supported modes: 'separate', 'chain'.",
            )

    @app.get("/defences", response_model=list[str])
    async def get_available_defences() -> list[str]:
        return list(guards.keys())

    return app
