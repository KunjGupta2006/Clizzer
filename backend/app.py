"""
CLIZZER - AI-Powered Video Clipping Engine (GTA Vice City Edition)
Main Gradio Blocks application mounted on FastAPI with custom GTA Vice City aesthetic.
"""

import os
import sys
import re
import time
import queue
import zipfile
import inspect
import tempfile
import threading
from typing import Optional, List, Any

# Ensure backend root and workspace root are on sys.path for direct executions
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import logging
logger = logging.getLogger("clizzer.app")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import gradio as gr

# Import ClipperPipeline from core.pipeline
try:
    from core.pipeline import ClipperPipeline
except ImportError as err1:
    try:
        from backend.core.pipeline import ClipperPipeline
    except ImportError as err2:
        logger.error("Failed to import ClipperPipeline from core or backend.core: %s / %s", err1, err2, exc_info=True)
        raise ImportError(f"Failed to import ClipperPipeline: {err1} / {err2}")


# ==============================================================================
# GTA VICE CITY THEME & CUSTOM STYLING
# ==============================================================================

CUSTOM_CSS = """
/* GTA Vice City Font & Global Styling */
@import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@400;600;700;800;900&family=Montserrat:wght@400;600;700&display=swap');

:root {
    --bg-dark: #0b0b0b;
    --panel-bg: #1a1a2e;
    --neon-pink: #ff007f;
    --gold: #f5b800;
    --neon-cyan: #00f0ff;
    --text-white: #ffffff;
}

/* Dark Background #0b0b0b */
body, .gradio-container, gradio-app, .gradio-container-wrap {
    background-color: #0b0b0b !important;
    color: #ffffff !important;
    font-family: 'Big Shoulders Display', 'Montserrat', sans-serif !important;
}

/* Headings in gold #f5b800, uppercase */
h1, h2, h3, h4, h5, h6, .gr-heading, .title-text {
    color: #f5b800 !important;
    text-transform: uppercase !important;
    letter-spacing: 2.5px !important;
    font-family: 'Big Shoulders Display', sans-serif !important;
    font-weight: 800 !important;
    text-shadow: 0 0 10px rgba(245, 184, 0, 0.35) !important;
}

/* Panels with #1a1a2e bg and #ff007f border */
.gr-panel, .gr-box, .gr-block, .block, .gr-form, .gr-group, fieldset,
div[data-testid="block"], div[class*="block"] {
    background-color: #1a1a2e !important;
    border: 1.5px solid #ff007f !important;
    border-radius: 8px !important;
    box-shadow: 0 0 16px rgba(255, 0, 127, 0.2) !important;
}

/* Labels and input text */
label span, .label-text, .gr-form label {
    color: #00f0ff !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    font-family: 'Big Shoulders Display', sans-serif !important;
    font-size: 1.05rem !important;
}

input[type="text"], textarea, .gr-text-input {
    background-color: #0d0d18 !important;
    color: #ffffff !important;
    border: 1px solid #ff007f !important;
    border-radius: 6px !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 1rem !important;
}

input[type="text"]:focus, textarea:focus {
    border-color: #f5b800 !important;
    box-shadow: 0 0 10px rgba(245, 184, 0, 0.5) !important;
}

/* Primary buttons: gradient pink to gold, uppercase, bold, letter-spacing */
button.primary, .gr-button-primary, button[variant="primary"], #start-mission-btn {
    background: linear-gradient(135deg, #ff007f 0%, #f5b800 100%) !important;
    color: #0b0b0b !important;
    text-transform: uppercase !important;
    font-weight: 900 !important;
    letter-spacing: 3px !important;
    font-size: 1.3rem !important;
    font-family: 'Big Shoulders Display', sans-serif !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: 0 0 25px rgba(255, 0, 127, 0.55), 0 0 12px rgba(245, 184, 0, 0.45) !important;
    padding: 14px 28px !important;
    cursor: pointer !important;
    transition: all 0.25s ease-in-out !important;
}

button.primary:hover, .gr-button-primary:hover, button[variant="primary"]:hover, #start-mission-btn:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 0 35px rgba(255, 0, 127, 0.8), 0 0 20px rgba(245, 184, 0, 0.7) !important;
}

/* Secondary button styling */
button.secondary, .gr-button-secondary, button[variant="secondary"], #download-all-btn {
    background-color: #1a1a2e !important;
    color: #f5b800 !important;
    border: 2px solid #f5b800 !important;
    text-transform: uppercase !important;
    font-weight: 800 !important;
    letter-spacing: 2px !important;
    font-size: 1.15rem !important;
    font-family: 'Big Shoulders Display', sans-serif !important;
    border-radius: 6px !important;
    box-shadow: 0 0 15px rgba(245, 184, 0, 0.25) !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
}

button.secondary:hover, .gr-button-secondary:hover, button[variant="secondary"]:hover, #download-all-btn:hover {
    background-color: #f5b800 !important;
    color: #0b0b0b !important;
    box-shadow: 0 0 25px rgba(245, 184, 0, 0.7) !important;
}

/* Sliders */
input[type="range"] {
    accent-color: #ff007f !important;
}

/* Video previews container */
video, .gr-video {
    border-radius: 6px !important;
    border: 1px solid #ff007f !important;
    background-color: #000000 !important;
}
"""

vice_city_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.pink,
    font=[gr.themes.GoogleFont("Big Shoulders Display"), "sans-serif"],
).set(
    body_background_fill="#0b0b0b",
    body_background_fill_dark="#0b0b0b",
    block_background_fill="#1a1a2e",
    block_background_fill_dark="#1a1a2e",
    block_border_color="#ff007f",
    block_border_color_dark="#ff007f",
    button_primary_background_fill="linear-gradient(135deg, #ff007f 0%, #f5b800 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #ff007f 0%, #f5b800 100%)",
    button_primary_text_color="#0b0b0b",
    button_primary_text_color_dark="#0b0b0b",
)


# ==============================================================================
# GTA WANTED STARS HELPER
# ==============================================================================

def render_wanted_stars(stars: int = 0, message: str = "MISSION STANDBY") -> str:
    """Format GTA Vice City wanted stars display."""
    stars = max(0, min(5, int(stars)))
    gold_star = "<span style='color: #f5b800; text-shadow: 0 0 12px #f5b800;'>★</span>"
    empty_star = "<span style='color: #43435c;'>☆</span>"
    stars_html = (gold_star * stars) + (empty_star * (5 - stars))

    return f"""
    <div style="background-color: #1a1a2e; border: 2px solid #ff007f; border-radius: 8px; padding: 18px; margin: 15px 0; text-align: center; box-shadow: 0 0 18px rgba(255, 0, 127, 0.3);">
        <div style="color: #f5b800; font-family: 'Big Shoulders Display', sans-serif; font-size: 1.6rem; font-weight: 900; letter-spacing: 4px; text-transform: uppercase;">
            WANTED LEVEL
        </div>
        <div style="font-size: 2.7rem; letter-spacing: 12px; margin: 6px 0;">
            {stars_html}
        </div>
        <div style="color: #00f0ff; font-family: 'Big Shoulders Display', sans-serif; font-size: 1.25rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;">
            {message}
        </div>
    </div>
    """


# ==============================================================================
# PIPELINE INVOCATION HELPER
# ==============================================================================

import asyncio

def run_coroutine_sync(coro):
    """Safely execute a coroutine from synchronous threads."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def execute_pipeline(pipeline_instance: Any, source: str, num_clips: int, max_duration: int, callback=None) -> List[str]:
    """Execute pipeline instance with dynamic signature matching and await async execution."""
    if hasattr(pipeline_instance, "hype_engine") and pipeline_instance.hype_engine is not None:
        pipeline_instance.hype_engine.num_clips = int(num_clips)
        pipeline_instance.hype_engine.max_clip_duration = float(max_duration)
    if hasattr(pipeline_instance, "config") and isinstance(pipeline_instance.config, dict):
        pipeline_instance.config["num_clips"] = int(num_clips)
        pipeline_instance.config["max_clip_duration"] = float(max_duration)

    target_func = None
    if hasattr(pipeline_instance, "process"):
        target_func = pipeline_instance.process
    elif hasattr(pipeline_instance, "run"):
        target_func = pipeline_instance.run
    elif callable(pipeline_instance):
        target_func = pipeline_instance
    else:
        raise AttributeError("ClipperPipeline does not expose a callable process or run method.")

    sig = inspect.signature(target_func)
    params = sig.parameters
    kwargs = {}

    if "progress_callback" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        kwargs["progress_callback"] = callback
    elif "callback" in params:
        kwargs["callback"] = callback

    if "num_clips" in params:
        kwargs["num_clips"] = int(num_clips)
    if "max_duration" in params:
        kwargs["max_duration"] = int(max_duration)

    try:
        raw_res = target_func(source, **kwargs)
    except TypeError:
        first_param = list(params.keys())[0] if params else "video_input"
        kwargs[first_param] = source
        raw_res = target_func(**kwargs)

    # Await if coroutine was returned
    if inspect.iscoroutine(raw_res):
        raw_res = run_coroutine_sync(raw_res)

    output_paths: List[str] = []
    if isinstance(raw_res, dict) and "clips" in raw_res:
        clips_data = raw_res["clips"]
        if isinstance(clips_data, list):
            for item in clips_data:
                if isinstance(item, dict) and "clip_path" in item:
                    p = str(item["clip_path"])
                    if os.path.exists(p):
                        output_paths.append(p)
                elif isinstance(item, str) and os.path.exists(item):
                    output_paths.append(item)
    elif isinstance(raw_res, list):
        for item in raw_res:
            if isinstance(item, dict) and "clip_path" in item:
                p = str(item["clip_path"])
                if os.path.exists(p):
                    output_paths.append(p)
            elif isinstance(item, str) and os.path.exists(item):
                output_paths.append(item)

    return output_paths


# ==============================================================================
# GRADIO PROCESSING & DOWNLOAD FUNCTIONS
# ==============================================================================

def process_clizzer_mission(url: Optional[str], file: Optional[Any], num_clips: int, max_duration: int):
    """
    Determine input (url or file), validate, and run ClipperPipeline
    with progress callback updating wanted stars. Yields up to 4 clip paths.
    """
    input_source = None

    # Determine input: uploaded file takes priority if present, else URL
    if file is not None:
        if isinstance(file, str):
            input_source = file
        elif hasattr(file, "name"):
            input_source = file.name
        elif isinstance(file, dict) and "name" in file:
            input_source = file["name"]
        else:
            input_source = str(file)

    if not input_source and url and url.strip():
        input_source = url.strip()

    # Input validation
    if not input_source or not str(input_source).strip():
        yield (
            render_wanted_stars(0, "MISSION ABORTED • PROVIDE A VALID URL OR VIDEO FILE"),
            None, None, None, None,
        )
        return

    progress_queue: queue.Queue = queue.Queue()

    def progress_callback(*args, **kwargs):
        stars = 1
        msg = "Engaging AI Clipping Mission..."

        if len(args) == 1:
            val = args[0]
            if isinstance(val, (int, float)):
                stars = int(val) if val <= 5 else int(round((val / 100.0) * 5))
            elif isinstance(val, str):
                msg = val
        elif len(args) == 2:
            if isinstance(args[0], (int, float)) and isinstance(args[1], str):
                stars = int(args[0])
                msg = args[1]
            elif isinstance(args[0], (int, float)) and isinstance(args[1], (int, float)):
                stars = int(round((args[0] / max(args[1], 1)) * 5))
        elif len(args) >= 3:
            curr, tot, msg_arg = args[0], args[1], args[2]
            stars = int(round((curr / max(tot, 1)) * 5))
            msg = str(msg_arg)

        if "stars" in kwargs:
            stars = kwargs["stars"]
        if "message" in kwargs:
            msg = kwargs["message"]

        progress_queue.put(("progress", (stars, msg)))

    execution_state = {"clips": [], "error": None}

    def run_worker():
        try:
            pipeline = ClipperPipeline()
            clips = execute_pipeline(
                pipeline_instance=pipeline,
                source=str(input_source).strip(),
                num_clips=int(num_clips),
                max_duration=int(max_duration),
                callback=progress_callback,
            )
            execution_state["clips"] = clips or []
        except Exception as exc:
            logger.error("Clizzer pipeline execution failure: %s", exc, exc_info=True)
            execution_state["error"] = str(exc)
        finally:
            progress_queue.put(("done", None))

    worker_thread = threading.Thread(target=run_worker, daemon=True)
    worker_thread.start()

    # Initial Progress Yield
    yield (
        render_wanted_stars(1, "MISSION LAUNCHED • INGESTING SOURCE FOOTAGE"),
        None, None, None, None,
    )

    while worker_thread.is_alive() or not progress_queue.empty():
        try:
            event_type, event_data = progress_queue.get(timeout=0.15)
            if event_type == "progress":
                st, m = event_data
                yield (
                    render_wanted_stars(st, m),
                    None, None, None, None,
                )
            elif event_type == "done":
                break
        except queue.Empty:
            continue

    worker_thread.join(timeout=2.0)

    if execution_state["error"]:
        yield (
            render_wanted_stars(0, f"MISSION FAILED • {execution_state['error']}"),
            None, None, None, None,
        )
        return

    clips = execution_state["clips"]
    c1 = clips[0] if len(clips) > 0 else None
    c2 = clips[1] if len(clips) > 1 else None
    c3 = clips[2] if len(clips) > 2 else None
    c4 = clips[3] if len(clips) > 3 else None

    count = len(clips)
    status_text = f"MISSION PASSED! {count} FIRE CLIPS GENERATED" if count > 0 else "MISSION PASSED • NO CLIPS IDENTIFIED"
    yield (
        render_wanted_stars(5, status_text),
        c1, c2, c3, c4,
    )


def download_all_clips(clip1: Optional[str], clip2: Optional[str], clip3: Optional[str], clip4: Optional[str]):
    """Bundle all generated clips into a downloadable ZIP archive."""
    clips = [c for c in [clip1, clip2, clip3, clip4] if c and os.path.exists(str(c))]
    if not clips:
        return gr.update(value=None, visible=False)

    archive_dir = os.path.join(tempfile.gettempdir(), "clizzer_archives")
    os.makedirs(archive_dir, exist_ok=True)
    zip_path = os.path.join(archive_dir, f"clizzer_clips_{int(time.time())}.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for index, clip_file in enumerate(clips, 1):
            arcname = f"clip_{index}_{os.path.basename(clip_file)}"
            zipf.write(clip_file, arcname=arcname)

    return gr.update(value=zip_path, visible=True)


def update_source_preview(url: Optional[str], file: Optional[Any]) -> str:
    """Generate reactive GTA Vice City live preview embed HTML for YouTube or uploaded videos."""
    # 1. File Upload preview
    if file is not None:
        file_path = file if isinstance(file, str) else getattr(file, "name", str(file))
        file_name = os.path.basename(file_path)
        return f"""
        <div style="max-width: 720px; margin: 10px auto; background: #1a1a2e; border: 2px solid #ff007f; border-radius: 10px; padding: 14px; box-shadow: 0 0 20px rgba(255, 0, 127, 0.45); text-align: center;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,0,127,0.3); padding-bottom: 6px; margin-bottom: 10px;">
                <span style="color: #f5b800; font-weight: 800; font-size: 1.1rem; letter-spacing: 1.5px; font-family: 'Big Shoulders Display', sans-serif;">📁 LOCAL SURVEILLANCE TAPE LOADED</span>
                <span style="color: #00f0ff; font-weight: 700; font-size: 0.85rem; letter-spacing: 1px;">LOCAL FILE</span>
            </div>
            <p style="color: #ffffff; font-size: 1.2rem; font-weight: 700; margin: 8px 0; font-family: 'Montserrat', sans-serif;">{file_name}</p>
            <p style="color: #00f0ff; font-size: 0.95rem; margin: 0;">Tape ready for AI hype scanning. Adjust sliders & hit START MISSION.</p>
        </div>
        """

    # 2. YouTube URL preview
    if url and url.strip():
        u = url.strip()
        yt_match = re.search(r'(?:v=|\/|youtu\.be\/|embed\/)([0-9A-Za-z_-]{11})', u)
        if yt_match:
            yt_id = yt_match.group(1)
            return f"""
            <div style="max-width: 720px; margin: 10px auto; background: #1a1a2e; border: 2px solid #00f0ff; border-radius: 10px; overflow: hidden; box-shadow: 0 0 25px rgba(0, 240, 255, 0.45); padding: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 4px 6px 8px 6px; border-bottom: 1px solid rgba(0,240,255,0.3); margin-bottom: 8px;">
                    <span style="color: #f5b800; font-weight: 800; font-size: 1.15rem; letter-spacing: 1.5px; font-family: 'Big Shoulders Display', sans-serif;">📡 RADAR TARGET ACQUIRED • LIVE FEED</span>
                    <span style="color: #ff007f; font-weight: 700; font-size: 0.85rem; letter-spacing: 1px;">YOUTUBE ({yt_id})</span>
                </div>
                <div style="position: relative; width: 100%; padding-bottom: 56.25%; height: 0;">
                    <iframe style="position: absolute; top:0; left: 0; width: 100%; height: 100%; border: none; border-radius: 6px;" src="https://www.youtube-nocookie.com/embed/{yt_id}" title="Source Video Preview" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
                </div>
            </div>
            """
        else:
            return f"""
            <div style="max-width: 720px; margin: 10px auto; background: #1a1a2e; border: 2px solid #f5b800; border-radius: 10px; padding: 14px; box-shadow: 0 0 15px rgba(245, 184, 0, 0.35); text-align: center;">
                <p style="color: #f5b800; font-size: 1.15rem; font-weight: 800; letter-spacing: 1.5px; margin: 0 0 6px 0; font-family: 'Big Shoulders Display', sans-serif;">🌐 WEB STREAM DETECTED</p>
                <p style="color: #ffffff; font-size: 0.95rem; margin: 0; word-break: break-all;">{u}</p>
                <p style="color: #00f0ff; font-size: 0.9rem; margin-top: 6px;">URL verified for AI ingest. Press START MISSION to cut highlights.</p>
            </div>
            """

    # 3. Default standby placeholder
    return """
    <div style="max-width: 720px; margin: 10px auto; background: rgba(26, 26, 46, 0.6); border: 1.5px dashed rgba(255, 0, 127, 0.45); border-radius: 10px; padding: 18px; text-align: center;">
        <p style="color: #00f0ff; font-size: 1.15rem; font-weight: 800; letter-spacing: 2px; margin: 0; font-family: 'Big Shoulders Display', sans-serif;">
            🎯 RADAR SCANNER STANDBY • PASTE A LINK ABOVE TO ENGAGE LIVE PREVIEW
        </p>
    </div>
    """


# ==============================================================================
# GRADIO BLOCKS APPLICATION
# ==============================================================================

with gr.Blocks(theme=vice_city_theme, css=CUSTOM_CSS, title="CLIZZER - Vice City Clips") as demo:
    # Header Section
    gr.HTML(
        """
        <div style="text-align: center; margin-bottom: 24px; padding-top: 10px;">
            <h1 style="color: #f5b800; font-size: 4.2rem; margin: 0; font-weight: 900; letter-spacing: 6px; text-shadow: 3px 3px 0px #ff007f, 6px 6px 0px #1a1a2e;">
                CLIZZER
            </h1>
            <p style="color: #00f0ff; font-size: 1.35rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-top: 8px;">
                Paste a link. Get fire clips. Zero subscriptions.
            </p>
        </div>
        """
    )

    # Input Row: URL input & File upload
    with gr.Row():
        url_input = gr.Textbox(
            label="URL INPUT",
            placeholder="Paste video link here (YouTube, Twitter/X, Twitch, etc.)...",
            lines=1,
            scale=3,
            elem_id="url-input",
        )
        file_input = gr.File(
            label="FILE UPLOAD FOR VIDEO",
            file_types=["video"],
            file_count="single",
            type="filepath",
            scale=2,
            elem_id="file-input",
        )

    # Live Source Radar Preview
    preview_box = gr.HTML(
        value=update_source_preview(None, None),
        elem_id="live-source-preview",
    )

    # Configuration Row: num_clips and max_duration sliders
    with gr.Row():
        num_clips_slider = gr.Slider(
            minimum=1,
            maximum=6,
            value=4,
            step=1,
            label="NUMBER OF CLIPS",
            elem_id="num-clips-slider",
        )
        max_duration_slider = gr.Slider(
            minimum=10,
            maximum=90,
            value=30,
            step=5,
            label="MAX DURATION (SECONDS)",
            elem_id="max-duration-slider",
        )

    # Primary Action Button
    start_btn = gr.Button(
        value="START MISSION",
        variant="primary",
        size="lg",
        elem_id="start-mission-btn",
    )

    # Wanted Stars Status Markdown
    status_md = gr.Markdown(
        value=render_wanted_stars(0, "READY FOR MISSION • INSERT SOURCE TO PROCEED"),
        elem_id="status-wanted-stars",
    )

    # Results Section: 4 gr.Video previews in a row
    gr.HTML("<h3 style='color: #f5b800; margin-top: 24px; margin-bottom: 8px; font-size: 1.45rem; letter-spacing: 2px;'>MISSION RESULTS • CLIP PREVIEWS</h3>")
    with gr.Row(elem_id="results-row"):
        video_1 = gr.Video(label="Clip #1", interactive=False, elem_id="video-clip-1")
        video_2 = gr.Video(label="Clip #2", interactive=False, elem_id="video-clip-2")
        video_3 = gr.Video(label="Clip #3", interactive=False, elem_id="video-clip-3")
        video_4 = gr.Video(label="Clip #4", interactive=False, elem_id="video-clip-4")

    # Download Button & File Output
    with gr.Row():
        download_btn = gr.Button(
            value="DOWNLOAD ALL CLIPS",
            variant="secondary",
            size="lg",
            elem_id="download-all-btn",
        )

    download_file = gr.File(
        label="DOWNLOAD FIRE CLIPS ARCHIVE (.ZIP)",
        interactive=False,
        visible=False,
        elem_id="download-archive-file",
    )

    # Event Handlers
    url_input.change(
        fn=update_source_preview,
        inputs=[url_input, file_input],
        outputs=[preview_box],
    )

    file_input.change(
        fn=update_source_preview,
        inputs=[url_input, file_input],
        outputs=[preview_box],
    )

    start_btn.click(
        fn=process_clizzer_mission,
        inputs=[url_input, file_input, num_clips_slider, max_duration_slider],
        outputs=[status_md, video_1, video_2, video_3, video_4],
    )

    download_btn.click(
        fn=download_all_clips,
        inputs=[video_1, video_2, video_3, video_4],
        outputs=[download_file],
    )


def create_gradio_app():
    """Return the initialized Gradio Blocks application."""
    return demo


# ==============================================================================
# FASTAPI APPLICATION & ENDPOINTS
# ==============================================================================

from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File, Form
import shutil
from pathlib import Path

app = FastAPI(
    title="Clizzer API",
    description="GTA Vice City-themed AI Video Clipper API & Server",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount clips output directory for direct HTTP video streaming and downloads
output_dir = Path("output").resolve()
clips_dir = output_dir / "clips"
clips_dir.mkdir(parents=True, exist_ok=True)
app.mount("/clips", StaticFiles(directory=str(clips_dir)), name="clips")


class ClipItem(BaseModel):
    clip_path: str
    thumbnail_path: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    hype_score: float = 5.0


class ProcessRequest(BaseModel):
    url: Optional[str] = Field(None, description="Video URL (e.g. YouTube, Twitch)")
    video_path: Optional[str] = Field(None, description="Local or uploaded video file path")
    num_clips: int = Field(default=4, ge=1, le=6, description="Number of clips to generate")
    max_duration: int = Field(default=30, ge=10, le=90, description="Max duration in seconds")
    reframe: bool = Field(default=True, description="Enable 9:16 vertical crop")
    subtitle_style: str = Field(default="karaoke", description="Subtitle animation style")


class ProcessResponse(BaseModel):
    status: str
    message: str
    clips: List[Any]
    num_clips: int
    video_title: Optional[str] = None


@app.get("/health")
async def health_check():
    """Health status endpoint."""
    return {
        "status": "healthy",
        "service": "Clizzer Backend",
        "version": "1.0.0",
        "timestamp": time.time(),
    }


@app.post("/process", response_model=ProcessResponse)
async def process_video_endpoint(payload: ProcessRequest):
    """Process video via API and return generated clip file paths and web URLs."""
    input_source = payload.video_path if payload.video_path else payload.url
    if not input_source or not str(input_source).strip():
        raise HTTPException(
            status_code=400,
            detail="Either 'url' or 'video_path' must be provided.",
        )

    try:
        pipeline = ClipperPipeline()
        if hasattr(pipeline, "hype_engine") and pipeline.hype_engine is not None:
            pipeline.hype_engine.num_clips = int(payload.num_clips)
            pipeline.hype_engine.max_clip_duration = float(payload.max_duration)
        if hasattr(pipeline, "config") and isinstance(pipeline.config, dict):
            pipeline.config["num_clips"] = int(payload.num_clips)
            pipeline.config["max_clip_duration"] = float(payload.max_duration)

        res = await pipeline.process(str(input_source).strip())
        raw_clips = res.get("clips", []) if isinstance(res, dict) else []
        video_title = res.get("video_title", "highlight_video") if isinstance(res, dict) else "video"

        formatted_clips = []
        for idx, item in enumerate(raw_clips):
            if isinstance(item, dict):
                local_path = item.get("clip_path", "")
                fname = os.path.basename(local_path)
                web_clip_url = f"http://localhost:7860/clips/{fname}"
                thumb_local = item.get("thumbnail_path")
                thumb_url = f"http://localhost:7860/clips/{os.path.basename(thumb_local)}" if thumb_local else None

                formatted_clips.append({
                    "clip_path": web_clip_url,
                    "thumbnail_path": thumb_url,
                    "start_time": float(item.get("start_time", 0.0)),
                    "end_time": float(item.get("end_time", 0.0)),
                    "duration": float(item.get("duration", 0.0)),
                    "hype_score": float(item.get("hype_score", 4.5)),
                })
            elif isinstance(item, str):
                fname = os.path.basename(item)
                formatted_clips.append({
                    "clip_path": f"http://localhost:7860/clips/{fname}",
                    "thumbnail_path": None,
                    "start_time": 0.0,
                    "end_time": float(payload.max_duration),
                    "duration": float(payload.max_duration),
                    "hype_score": 4.5,
                })

        return ProcessResponse(
            status="success",
            message=f"Successfully generated {len(formatted_clips)} clips",
            clips=formatted_clips,
            num_clips=len(formatted_clips),
            video_title=video_title,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/process-upload")
async def process_upload_endpoint(
    file: UploadFile = File(...),
    num_clips: int = Form(4),
    max_duration: int = Form(30),
    reframe: bool = Form(True),
    subtitle_style: str = Form("karaoke"),
):
    """Process uploaded video file."""
    upload_dir = Path("output/uploads").resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_file = upload_dir / f"upload_{int(time.time())}_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    req = ProcessRequest(
        video_path=str(temp_file),
        num_clips=num_clips,
        max_duration=max_duration,
        reframe=reframe,
        subtitle_style=subtitle_style,
    )
    return await process_video_endpoint(req)


# Alias endpoint for /api/process and /api/health
@app.get("/api/health", include_in_schema=False)
async def api_health_alias():
    return await health_check()


@app.post("/api/process", response_model=ProcessResponse, include_in_schema=False)
async def api_process_alias(payload: ProcessRequest):
    return await process_video_endpoint(payload)


# Mount Gradio Blocks application onto FastAPI
app = gr.mount_gradio_app(app, demo, path="/")


# ==============================================================================
# SERVER LAUNCHER
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
