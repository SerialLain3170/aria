"""Small web UI for native AniSora V3.2 image-to-video inference."""

from __future__ import annotations

import argparse
import cgi
import html
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from urllib.parse import parse_qs, urlparse

from PIL import Image


DEFAULT_PROMPT = (
    "A 2D hand-drawn Japanese anime character in a quiet visual-novel scene, "
    "subtle facial expression, gentle hair movement, soft cinematic lighting, "
    "single continuous shot. aesthetic score: 5.5. motion score: 2.5. "
    "There is no text in the video."
)


@dataclass
class ServerConfig:
    host: str
    port: int
    root: Path
    anisora_code: Path
    ckpt_dir: Path
    gpu_id: str
    size: str
    frame_num: int
    sample_steps: int
    sample_shift: float
    sample_guide_scale: float

    @property
    def jobs_dir(self) -> Path:
        return self.root / "web" / "jobs"


@dataclass
class Job:
    job_id: str
    prompt: str
    image_path: Path
    work_dir: Path
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    returncode: int | None = None
    output_path: Path | None = None
    error: str | None = None


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()


def normalize_prompt(prompt: str) -> str:
    prompt = " ".join(prompt.strip().split())
    if not prompt:
        prompt = DEFAULT_PROMPT
    lower = prompt.lower()
    if "aesthetic score:" not in lower:
        prompt += " aesthetic score: 5.5."
    if "motion score:" not in lower:
        prompt += " motion score: 2.5."
    if "there is no text in the video" not in lower:
        prompt += " There is no text in the video."
    return prompt


def save_uploaded_image(upload_item, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".upload")
    with tmp.open("wb") as f:
        shutil.copyfileobj(upload_item.file, f)
    with Image.open(tmp) as img:
        img.convert("RGB").save(dest)
    tmp.unlink(missing_ok=True)


def build_command(config: ServerConfig, job: Job) -> list[str]:
    prompt_list = job.work_dir / "prompt.txt"
    save_dir = job.work_dir / "outputs"
    prompt_list.write_text(f"{job.prompt}@@{job.image_path}&&0\n", encoding="utf-8")
    save_dir.mkdir(parents=True, exist_ok=True)

    return [
        "python",
        str(config.anisora_code / "generate_txt_new.py"),
        "--task",
        "i2v-A14B",
        "--size",
        config.size,
        "--ckpt_dir",
        str(config.ckpt_dir),
        "--prompt_list",
        str(prompt_list),
        "--save_dir",
        str(save_dir),
        "--sample_steps",
        str(config.sample_steps),
        "--sample_shift",
        str(config.sample_shift),
        "--sample_guide_scale",
        str(config.sample_guide_scale),
        "--ckpt_dir_lowname",
        "low_noise_model",
        "--ckpt_dir_highname",
        "high_noise_model",
        "--base_seed",
        "4096",
        "--frame_num",
        str(config.frame_num),
        "--offload_model",
        "True",
        "--t5_cpu",
        "--convert_model_dtype",
    ]


def run_job(config: ServerConfig, job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job.status = "waiting"

    with INFERENCE_LOCK:
        with JOBS_LOCK:
            job = JOBS[job_id]
            job.status = "running"
            job.started_at = time.time()

        log_path = job.work_dir / "anisora.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = config.gpu_id
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        env["PYTHONPATH"] = str(config.anisora_code)

        try:
            cmd = build_command(config, job)
            with log_path.open("w", encoding="utf-8") as log:
                log.write("$ " + " ".join(cmd) + "\n\n")
                log.flush()
                proc = subprocess.run(
                    cmd,
                    cwd=str(config.anisora_code),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

            outputs = sorted((job.work_dir / "outputs").glob("*.mp4"))
            with JOBS_LOCK:
                job.returncode = proc.returncode
                job.finished_at = time.time()
                if proc.returncode == 0 and outputs:
                    job.status = "done"
                    job.output_path = outputs[0]
                else:
                    job.status = "failed"
                    job.error = f"AniSora exited with code {proc.returncode}."
        except Exception as exc:
            with JOBS_LOCK:
                job.status = "failed"
                job.finished_at = time.time()
                job.error = str(exc)


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}s"


def page_shell(title: str, body: str, refresh: bool = False) -> bytes:
    refresh_tag = '<meta http-equiv="refresh" content="8">' if refresh else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {refresh_tag}
  <title>{html.escape(title)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Baloo+2:wght@600;700;800&family=M+PLUS+Rounded+1c:wght@400;500;700;800&display=swap" rel="stylesheet">
  <style>
    :root {{
      --pink: #ff6fb5;
      --hot-pink: #ff3d94;
      --yellow: #ffd93d;
      --sky: #4fc3f7;
      --purple: #a66dff;
      --mint: #4ce0b3;
      --ink: #3a2b4d;
      --muted: #7c6690;
      --panel: rgba(255, 255, 255, .9);
      --line: #ffd3ea;
      --soft: #fff4fb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: linear-gradient(120deg, #ffe6f5 0%, #ffe9c7 25%, #dff6ff 55%, #eee0ff 100%);
      background-size: 200% 200%;
      animation: bgshift 18s ease-in-out infinite;
      color: var(--ink);
      font: 15px/1.45 "M PLUS Rounded 1c", "Baloo 2", ui-rounded, system-ui, sans-serif;
      overflow-x: hidden;
    }}
    @keyframes bgshift {{ 0%,100% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} }}
    header {{ text-align: center; padding: 30px 20px 16px; }}
    main {{ width: min(1120px, calc(100vw - 32px)); margin: 0 auto 44px; }}
    h1 {{
      margin: 0;
      font-family: "Baloo 2", sans-serif;
      font-weight: 800;
      font-size: clamp(42px, 7vw, 76px);
      line-height: .92;
      letter-spacing: 0;
      background: linear-gradient(90deg, var(--hot-pink), var(--purple), var(--sky));
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
      text-shadow: 3px 3px 0 rgba(255,255,255,.62);
    }}
    h2 {{
      font-family: "Baloo 2", sans-serif;
      font-size: 22px;
      line-height: 1.1;
      margin: 0 0 14px;
      color: var(--hot-pink);
    }}
    p {{ color: var(--muted); margin: 8px 0; font-weight: 700; }}
    .grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(280px, 380px); gap: 20px; align-items: start; }}
    .panel {{
      background: var(--panel);
      border: 3px solid #fff;
      border-radius: 28px;
      box-shadow: 0 12px 0 rgba(255,111,181,.22), 0 22px 42px rgba(166,109,255,.16);
      padding: 20px;
      backdrop-filter: blur(8px);
      overflow: hidden;
    }}
    label {{ display: block; color: #6b4b8a; font-size: 12px; font-weight: 900; margin: 14px 0 7px; }}
    input[type=file], textarea {{
      width: 100%;
      border: 2px solid var(--line);
      border-radius: 16px;
      background: #fffaff;
      color: var(--ink);
      font: inherit;
      padding: 12px 13px;
      min-height: 46px;
    }}
    input:focus, textarea:focus {{ outline: none; border-color: var(--sky); box-shadow: 0 0 0 4px rgba(79,195,247,.22); }}
    textarea {{ min-height: 170px; resize: vertical; }}
    button, .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 46px;
      border: 0;
      border-radius: 16px;
      background: linear-gradient(135deg, var(--hot-pink), var(--purple));
      color: white;
      padding: 0 16px;
      text-decoration: none;
      font-family: "Baloo 2", sans-serif;
      font-weight: 800;
      cursor: pointer;
      box-shadow: 0 6px 0 #c23f8e;
    }}
    button:hover, .button:hover {{ transform: translateY(-2px); }}
    button:active, .button:active {{ transform: translateY(3px); box-shadow: 0 2px 0 #c23f8e; }}
    .button.secondary {{ background: linear-gradient(135deg, var(--sky), var(--mint)); box-shadow: 0 6px 0 #2b9bc7; }}
    dl {{ display: grid; grid-template-columns: 120px 1fr; gap: 8px 12px; margin: 0; }}
    dt {{ color: #6b4b8a; font-weight: 800; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    .status {{ display: inline-flex; border-radius: 999px; padding: 6px 12px; background: #fff4fb; color: var(--muted); font-weight: 900; }}
    .status.done {{ background: #dffaf3; color: #15987c; }}
    .status.failed {{ background: #ffe2ec; color: #d33267; }}
    video, img {{
      width: 100%;
      max-height: 70vh;
      border-radius: 22px;
      border: 4px solid var(--yellow);
      background: #fff;
      box-shadow: 0 10px 24px rgba(0,0,0,.12);
    }}
    pre {{
      max-height: 360px;
      overflow: auto;
      white-space: pre-wrap;
      border: 2px solid var(--line);
      border-radius: 16px;
      background: #fffaff;
      padding: 12px;
      color: var(--ink);
    }}
    @media (max-width: 820px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header><h1>AniSora I2V Lab</h1><p>First-frame anime video generation with the verified V3.2 smoke-test settings.</p></header>
  <main>{body}</main>
</body>
</html>""".encode("utf-8")


def form_page(config: ServerConfig) -> bytes:
    body = f"""
<div class="grid">
  <section class="panel">
    <h2>Generate From First Frame</h2>
    <form method="post" action="/generate" enctype="multipart/form-data">
      <label for="image">Input image</label>
      <input id="image" name="image" type="file" accept="image/*" required>
      <label for="prompt">Prompt</label>
      <textarea id="prompt" name="prompt">{html.escape(DEFAULT_PROMPT)}</textarea>
      <p>Image position is fixed to first frame. Generation runs as a queued job.</p>
      <button type="submit">Generate video</button>
    </form>
  </section>
  <aside class="panel">
    <h2>Current Settings</h2>
    <dl>
      <dt>Backbone</dt><dd>AniSora V3.2 i2v-A14B</dd>
      <dt>Size</dt><dd>{html.escape(config.size)}</dd>
      <dt>Frames</dt><dd>{config.frame_num}</dd>
      <dt>Steps</dt><dd>{config.sample_steps}</dd>
      <dt>Shift</dt><dd>{config.sample_shift}</dd>
      <dt>Guidance</dt><dd>{config.sample_guide_scale}</dd>
      <dt>GPU</dt><dd>{html.escape(config.gpu_id)}</dd>
      <dt>Output root</dt><dd>{html.escape(str(config.jobs_dir))}</dd>
    </dl>
  </aside>
</div>"""
    return page_shell("Text-to-Anime I2V", body)


def job_page(job: Job, include_log: bool = True) -> bytes:
    elapsed = None
    if job.started_at:
        elapsed = (job.finished_at or time.time()) - job.started_at
    refresh = job.status in {"queued", "waiting", "running"}
    status_class = "done" if job.status == "done" else "failed" if job.status == "failed" else ""

    result = ""
    if job.status == "done" and job.output_path:
        result = f"""
  <section class="panel">
    <h2>Generated Video</h2>
    <video controls src="/file/{html.escape(job.job_id)}/output"></video>
    <p><a class="button" href="/file/{html.escape(job.job_id)}/output" download>Download MP4</a></p>
  </section>"""
    elif job.status == "failed":
        result = f"""
  <section class="panel">
    <h2>Generation Failed</h2>
    <p>{html.escape(job.error or "Unknown error.")}</p>
  </section>"""

    log_text = ""
    log_path = job.work_dir / "anisora.log"
    if include_log and log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        log_text = f"<section class=\"panel\"><h2>Log</h2><pre>{html.escape(text[-12000:])}</pre></section>"

    body = f"""
<div class="grid">
  <section class="panel">
    <h2>Job</h2>
    <dl>
      <dt>Status</dt><dd><span class="status {status_class}">{html.escape(job.status)}</span></dd>
      <dt>Job ID</dt><dd>{html.escape(job.job_id)}</dd>
      <dt>Elapsed</dt><dd>{fmt_seconds(elapsed)}</dd>
      <dt>Return code</dt><dd>{job.returncode if job.returncode is not None else "-"}</dd>
    </dl>
    <p><a class="button secondary" href="/">New job</a></p>
  </section>
  <section class="panel">
    <h2>Input</h2>
    <img src="/file/{html.escape(job.job_id)}/input" alt="input image">
  </section>
</div>
<section class="panel" style="margin-top:20px">
  <h2>Prompt</h2>
  <p>{html.escape(job.prompt)}</p>
</section>
<div style="margin-top:20px">{result}</div>
<div style="margin-top:20px">{log_text}</div>"""
    return page_shell(f"Job {job.job_id}", body, refresh=refresh)


class AnimeHandler(BaseHTTPRequestHandler):
    server_version = "TextToAnimeI2V/0.1"
    config: ClassVar[ServerConfig]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html(form_page(self.config))
            return
        if parsed.path.startswith("/jobs/"):
            job_id = parsed.path.split("/", 2)[2]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Unknown job")
                return
            self.send_html(job_page(job))
            return
        if parsed.path.startswith("/file/"):
            self.send_job_file(parsed.path)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/generate":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        upload = form["image"] if "image" in form else None
        if upload is None or not getattr(upload, "filename", ""):
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing image")
            return

        prompt = normalize_prompt(form.getfirst("prompt", ""))
        job_id = uuid.uuid4().hex[:12]
        work_dir = self.config.jobs_dir / job_id
        image_path = work_dir / "input.png"

        try:
            save_uploaded_image(upload, image_path)
        except Exception as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, f"Invalid image: {exc}")
            return

        job = Job(job_id=job_id, prompt=prompt, image_path=image_path, work_dir=work_dir)
        with JOBS_LOCK:
            JOBS[job_id] = job

        thread = threading.Thread(target=run_job, args=(self.config, job_id), daemon=True)
        thread.start()
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", f"/jobs/{job_id}")
        self.end_headers()

    def send_job_file(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) != 3:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        _, job_id, kind = parts
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown job")
            return
        if kind == "input":
            file_path = job.image_path
            content_type = "image/png"
        elif kind == "output" and job.output_path:
            file_path = job.output_path
            content_type = "video/mp4"
        elif kind == "log":
            file_path = job.work_dir / "anisora.log"
            content_type = "text/plain; charset=utf-8"
        else:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.end_headers()
        with file_path.open("rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def send_html(self, payload: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--root", type=Path, default=Path("/data/shasegawa/t2a"))
    parser.add_argument("--gpu-id", default=os.environ.get("GPU_ID", "3"))
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--frame-num", type=int, default=17)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--sample-shift", type=float, default=5.0)
    parser.add_argument("--sample-guide-scale", type=float, default=1.0)
    parser.add_argument("--anisora-code", type=Path, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    config = ServerConfig(
        host=args.host,
        port=args.port,
        root=root,
        anisora_code=args.anisora_code or root / "code" / "index-anisora" / "anisoraV3.2",
        ckpt_dir=args.ckpt_dir or root / "models" / "Index-anisora" / "V3.2",
        gpu_id=args.gpu_id,
        size=args.size,
        frame_num=args.frame_num,
        sample_steps=args.sample_steps,
        sample_shift=args.sample_shift,
        sample_guide_scale=args.sample_guide_scale,
    )
    config.jobs_dir.mkdir(parents=True, exist_ok=True)
    AnimeHandler.config = config
    server = ThreadingHTTPServer((config.host, config.port), AnimeHandler)
    print(f"Serving AniSora I2V on http://{config.host}:{config.port}/")
    print(f"Jobs: {config.jobs_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()
