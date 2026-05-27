import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..defence.abstract_defence import AbstractDefence

logger = logging.getLogger(__name__)


def create_app(guards: dict[str, AbstractDefence]) -> FastAPI:
    class EvaluationRequest(BaseModel):
        prompt: str
        defences: list[str]
        mode: str

    class DefenceResult(BaseModel):
        is_safe: bool
        details: str
        confidence: float = -1.0  # -1 means the guard does not produce a score

    app = FastAPI(
        title="LLM Defence Suite API",
        description="Evaluates prompts against security defences.",
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
            for name in request.defences:
                result = _run_guard(name, request.prompt)
                if not result.is_safe:
                    return {name: result}

            return {
                "ChainResult": DefenceResult(
                    is_safe=True,
                    details="All defences passed in chain evaluation.",
                )
            }

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode '{request.mode}'. Supported modes: 'separate', 'chain'.",
            )

    @app.get("/defences", response_model=list[str])
    async def get_available_defences() -> list[str]:
        return list(guards.keys())

    return app
