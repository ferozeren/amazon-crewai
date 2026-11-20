import json
import os
import queue
import threading
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .paths import ENV_PATH, FRONTEND_DIR, ASSETS_ROOT

load_dotenv(ENV_PATH)

from .crew import (
    build_fallback_recommendation,
    build_fallback_summary,
    create_laptop_crew,
)
from .data_manager import clean_dataset, get_dataset_stats, get_laptops_data, get_metadata

app = FastAPI(
    title="AI Laptop Recommendation System",
    description="CrewAI and FastAPI Laptop Recommendation Dashboard",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    provider: str  # 'local', 'openrouter', 'openai', 'gemini' or 'nvidia'
    major: str
    budget: float
    ram: Optional[float] = None
    brand: Optional[str] = "Any"
    os_name: Optional[str] = "Any"
    details: Optional[str] = ""
    model: Optional[str] = None
    gpu: Optional[str] = "Any"  # dedicated / integrated / RTX 4060 etc
    include_unavailable: Optional[bool] = False


NO_MATCH_PATTERNS = (
    "no laptops found",
    "no matching laptops",
    "no laptops match",
    "no matching models",
    "did not find any",
    "could not find any laptop",
)


def _indicates_no_match(text: str) -> bool:
    """Heuristic: did the AI conclude there are no matching laptops?"""
    if not text:
        return False
    lowered = text.lower()
    return any(pattern in lowered for pattern in NO_MATCH_PATTERNS)


def _sse(payload: dict) -> str:
    """Format a dict as a Server-Sent Events data frame."""
    return f"data: {json.dumps(payload)}\n\n"


_TOOL_LABELS = {
    "query_laptops_tool": "Searching the laptop catalog",
    "get_dataset_stats_tool": "Reviewing the dataset stats",
}


def _shorten(text: str, limit: int = 72) -> str:
    """Trim a status line to a compact, readable length."""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _describe_step(step: Any) -> str:
    """Turn a CrewAI step callback payload (AgentAction | AgentFinish) into a
    short, human-friendly progress message for the UI."""
    tool = getattr(step, "tool", None)
    if tool:
        name = str(tool).strip()
        label = _TOOL_LABELS.get(name)
        if label:
            return label
        return _shorten(name.replace("_", " ").replace(" tool", "").strip(), 72)
    log = getattr(step, "log", None)
    if log:
        text = str(log).strip().splitlines()[0].strip() if str(log).strip() else ""
        if text:
            return _shorten(text, 72)
    return "Agent is thinking…"


def _describe_task(output: Any) -> str:
    """Turn a CrewAI task callback payload (TaskOutput) into a short, friendly
    progress label. Only short task *names* are ever surfaced to the UI — the
    long task description stays out of the stream so it can be shown under a
    "Show more details" toggle instead of cluttering the loading view."""
    name = getattr(output, "name", None)
    agent = getattr(output, "agent", None)
    role = getattr(agent, "role", None) if agent is not None else None
    label = str(name).strip() if name else ""
    if not label:
        label = "Running the AI advisor"
    elif label.lower().startswith("search"):
        label = "Searching the laptop catalog"
    elif label.lower().startswith(("recommend", "write")):
        label = "Writing your personalized recommendation"
    elif label.lower().startswith("analy"):
        label = "Analyzing your profile"
    if role:
        return f"{label} — {role}"
    return label


class CleanRequest(BaseModel):
    pass




@app.post("/api/clean")
def clean_data():
    """Trigger the data cleaning process on the raw dataset."""
    try:
        res = clean_dataset()
        if "error" in res:
            raise HTTPException(status_code=400, detail=res["error"])
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data cleaning failed: {str(e)}")


@app.get("/api/stats")
def get_stats():
    """Retrieve statistics about the cleaned dataset."""
    try:
        stats = get_dataset_stats()
        if "error" in stats:
            raise HTTPException(status_code=400, detail=stats["error"])
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/api/metadata")
def get_meta():
    """Last updated / last trained timestamps and dataset health."""
    try:
        meta = get_metadata()
        stats = get_dataset_stats()
        if "error" not in stats:
            meta["total_records"] = stats.get("total_records")
            meta["available_count"] = stats.get("available_count")
            meta["unavailable_count"] = stats.get("unavailable_count")
            meta["dedicated_gpu_count"] = stats.get("dedicated_gpu_count")
        return meta
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")


@app.get("/api/laptops")
def get_laptops(include_unavailable: bool = False):
    """Retrieve all cleaned laptops in the database.

    By default only available laptops are returned; pass ?include_unavailable=true to see all.
    """
    try:
        rows = get_laptops_data()
        if not include_unavailable:
            filtered = [r for r in rows if str(r.get("Is_Available", True)).lower() in ("true", "1", "yes", "1.0")]
            if filtered or any("Is_Available" in r for r in rows):
                return filtered
        return rows
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load laptops list: {str(e)}"
        )


@app.post("/api/refresh")
def refresh_dataset(scrape: bool = False, pages: int = 2):
    """Refresh dataset: re-clean, optionally scrape Amazon, retrain model."""
    try:
        result = {}
        if scrape:
            try:
                from scrape_laptops import scrape_queries
                scrape_queries(pages=pages)
                result["scrape"] = "done"
            except Exception as e:
                result["scrape_error"] = str(e)
        clean_res = clean_dataset()
        result["clean"] = clean_res
        try:
            import train_model
            train_model.train()
            result["train"] = "done"
        except Exception as e:
            result["train_error"] = str(e)
        result["metadata"] = get_metadata()
        result["stats"] = get_dataset_stats()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refresh failed: {str(e)}")


@app.post("/api/train")
def train_model_endpoint():
    """Retrain the ML ranker on the current clean dataset."""
    try:
        import train_model
        train_model.train()
        return {"status": "success", "metadata": get_metadata(), "stats": get_dataset_stats()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Train failed: {str(e)}")


def _fallback_result(
    user_profile: dict, error_message: str, warning: str, details: str, reason: str = "ai_error"
) -> dict:
    """Shared fallback response payload (status=fallback) used by the SSE stream.

    ``reason`` tells the UI *why* we fell back so it can show a distinct
    message (missing API key vs. an AI failure vs. an AI no-match result).
    """
    return {
        "status": "fallback",
        "fallback_reason": reason,
        "recommendation": build_fallback_recommendation(
            user_profile=user_profile, error_message=error_message
        ),
        "summary": build_fallback_summary(user_profile),
        "warning": warning,
        "details": details,
    }


@app.post("/api/recommend")
def recommend_laptops(req: RecommendRequest, x_api_key: Optional[str] = Header(None)):
    """
    Run CrewAI agents to recommend laptops based on student major and specs.

    Responds as a Server-Sent Events stream so the frontend can show live
    progress (which agent is working, which tool is being called) while the
    crew runs. The final frame carries {status, recommendation, summary, ...}.
    """
    provider_str = req.provider.lower()
    if "local" in provider_str:
        provider = "local"
    elif "openrouter" in provider_str:
        provider = "openrouter"
    elif "gemini" in provider_str or "google" in provider_str:
        provider = "gemini"
    elif "openai" in provider_str:
        provider = "openai"
    elif "nvidia" in provider_str:
        provider = "nvidia"
    else:
        provider = provider_str

    api_key = x_api_key
    if not api_key and provider != "local":
        if provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        elif provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
        elif provider == "nvidia":
            api_key = os.environ.get("NVIDIA_API_KEY")

    user_profile = {
        "major": req.major,
        "budget": req.budget,
        "ram": req.ram if req.ram else "Any",
        "brand": req.brand if req.brand else "Any",
        "os": req.os_name if req.os_name else "Any",
        "details": req.details if req.details else "None",
        "gpu": req.gpu if req.gpu else "Any",
        "include_unavailable": bool(req.include_unavailable),
    }

    def event_stream():
        events: "queue.Queue[tuple[str, str]]" = queue.Queue()

        def step_cb(step: Any) -> None:
            events.put(("step", _describe_step(step)))

        def task_cb(output: Any) -> None:
            events.put(("task", _describe_task(output)))

        # No key, no LLM: serve catalog picks directly.
        if provider == "local":
            yield _sse(
                {"type": "status", "message": "Using the local laptop catalog…"}
            )
            yield _sse(
                {
                    "type": "result",
                    **_fallback_result(
                        user_profile,
                        "Local catalog mode — no AI provider selected.",
                        "Showing picks matched from the local laptop catalog.",
                        "Select an AI provider and add its API key for a personalized written recommendation.",
                        reason="local",
                    ),
                }
            )
            return

        # A missing key is not an AI failure; tag it so the UI says so.
        if not api_key:
            yield _sse(
                {
                    "type": "status",
                    "message": "No API key found — showing local picks.",
                }
            )
            yield _sse(
                {
                    "type": "result",
                    **_fallback_result(
                        user_profile,
                        f"No {provider} API key was supplied.",
                        f"No API key for {provider} was found. These picks come from the local laptop database — add a key to {provider} to get AI recommendations.",
                        f"Set the {provider} API key in your .env file or pass it via the X-Api-Key header.",
                        reason="missing_api_key",
                    ),
                }
            )
            return

        yield _sse(
            {
                "type": "status",
                "message": "Analyzing your profile…",
            }
        )

        try:
            crew = create_laptop_crew(
                api_key=api_key,
                provider=provider,
                user_profile=user_profile,
                model=req.model,
                step_callback=step_cb,
                task_callback=task_cb,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            yield _sse(
                {
                    "type": "status",
                    "message": "AI service unavailable — showing local picks.",
                }
            )
            yield _sse(
                {
                    "type": "result",
                    **_fallback_result(
                        user_profile,
                        str(e),
                        "The AI service hit an error before it could start. These picks come from the local laptop database.",
                        str(e),
                        reason="ai_error",
                    ),
                }
            )
            return

        if crew is None:
            yield _sse(
                {
                    "type": "status",
                    "message": "AI service unavailable — showing local picks.",
                }
            )
            yield _sse(
                {
                    "type": "result",
                    **_fallback_result(
                        user_profile,
                        "LLM initialization failed. Please check API key and provider.",
                        "The AI service could not be initialized, so these picks come from the local laptop database.",
                        "LLM initialization failed. Please check your API key and provider.",
                        reason="ai_error",
                    ),
                }
            )
            return

        outcome: dict = {}

        def run() -> None:
            try:
                outcome["result"] = crew.kickoff()
            except Exception as e:  # noqa: BLE001 - surfaced via fallback below
                outcome["error"] = e

        worker = threading.Thread(target=run, daemon=True)
        worker.start()

        while not outcome:
            try:
                kind, message = events.get(timeout=0.5)
                yield _sse({"type": kind, "message": message})
            except queue.Empty:
                yield ": keepalive\n\n"  # keep the connection alive

        while True:
            try:
                kind, message = events.get(timeout=0.1)
            except queue.Empty:
                break
            yield _sse({"type": kind, "message": message})

        worker.join(timeout=10)

        if "error" in outcome:
            import traceback

            traceback.print_exc()
            error = outcome["error"]
            yield _sse(
                {
                    "type": "result",
                    **_fallback_result(
                        user_profile,
                        str(error),
                        "The AI service ran into a problem mid-recommendation, so these picks come from the local laptop database.",
                        str(error),
                        reason="ai_error",
                    ),
                }
            )
            return

        result_text = getattr(outcome["result"], "raw", str(outcome["result"]))

        # A "no match" report is usually a tool-calling miss; use real picks.
        if _indicates_no_match(result_text):
            yield _sse(
                {
                    "type": "status",
                    "message": "Widening the search — using verified local picks.",
                }
            )
            yield _sse(
                {
                    "type": "result",
                    **_fallback_result(
                        user_profile,
                        "The AI advisor could not filter the catalog to matching laptops.",
                        "The AI advisor couldn't narrow the catalog, so these picks come from the local laptop database.",
                        "The AI returned a no-match result; using verified local picks instead.",
                        reason="no_match",
                    ),
                }
            )
            return

        yield _sse({"type": "result", "status": "success", "recommendation": result_text})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


if ASSETS_ROOT.exists():
    app.mount("/assets", StaticFiles(directory=ASSETS_ROOT), name="assets")

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    @app.get("/")
    def read_root_dummy():
        return HTMLResponse("<h1>Laptop Recommendation Backend</h1>")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
