import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import APIError

from .api import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("career_assistant")

app = FastAPI(title="Career Intelligence Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.exception_handler(APIError)
async def openai_error_handler(request: Request, exc: APIError):
    # Never leak upstream error bodies (may contain request internals) back to the client.
    logger.error("OpenAI API error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=502,
        content={"detail": "The AI service is temporarily unavailable or misconfigured. Please try again shortly."},
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    logger.error("Runtime error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Server configuration error. Please contact the administrator."})

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
