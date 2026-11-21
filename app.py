import os
import sys
from pathlib import Path

# Ensure application root is in python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import gradio as gr
import uvicorn
from backend.app.main import app as fastapi_app
from backend.app.config import settings

# Initialize runtime directories
settings.init_directories()

# Build Gradio control interface
with gr.Blocks(title="Clizzer AI Clipper - API & Dashboard") as demo:
    gr.Markdown(
        """
        # 🎬 Clizzer AI Video Processing API
        ### Autonomous Multimodal Stream & Video Clipper

        The backend engine is active on **Hugging Face Spaces** and ready to serve your Vercel frontend.

        - **FastAPI Endpoints:**
          - `POST /jobs` — Submit YouTube URL or video file for autonomous clipping
          - `GET /jobs/{id}` — Poll job status, progress, timings & generated clips
          - `GET /jobs/{id}/clips/{idx}/download` — Stream high-quality 9:16 vertical render
          - `GET /health` — Diagnostic health check
        - **Documentation:** [Interactive Swagger UI](/docs) | [ReDoc](/redoc)
        """
    )

    with gr.Accordion("Engine Diagnostics", open=True):
        with gr.Row():
            check_btn = gr.Button("🔍 Ping Backend Health", variant="primary")
        health_output = gr.JSON(label="System Status")

        def get_health():
            return {
                "status": "healthy",
                "llm_provider": settings.LLM_PROVIDER,
                "groq_configured": bool(settings.GROQ_API_KEY),
                "hf_token_configured": bool(settings.HF_TOKEN),
                "jobs_dir": str(settings.JOBS_DIR),
                "cache_dir": str(settings.CACHE_DIR),
            }

        check_btn.click(fn=get_health, outputs=health_output)

# Mount Gradio app into FastAPI at root '/'
# FastAPI routes (/jobs, /docs, /health, /openapi.json) are handled with top priority
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
