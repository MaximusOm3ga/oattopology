"""
app.py — FastAPI backend for oattopology UI
Run with: uvicorn app:app --reload --port 8000

Requires: fastapi uvicorn python-dotenv
"""

import os, io, uuid, base64, asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from makerrec import (
    parse_problem,
    refine_params,
    semantic_to_conditions,
    build_ae_batch,
    export_svg,
)
from OAT.DataUtils._utils import BatchDict
from OAT.Models import NFAE, CTOPUNet
from OAT.Pipelines.OATPipeline import OATPipeline, DDIMPipeline

print("Loading OAT models...")
_ae   = NFAE.from_pretrained("OpenTO/NFAE")
_ldm  = CTOPUNet.from_pretrained("OpenTO/LDM")
_ddim = DDIMPipeline()
_pipeline = OATPipeline(DDIM=_ddim, diffusion_model=_ldm, nfae=_ae)
_ae.eval()
_ldm.eval()
print("Models ready.")

_executor = ThreadPoolExecutor(max_workers=1)

_jobs: dict[str, dict] = {}


def _tensor_to_b64(tensor, W: int, H: int) -> str:
    arr = tensor.squeeze().float().numpy()

    if arr.shape == (W,H):
        arr = arr.T

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(arr, cmap="gray_r", vmin=0, vmax=1)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _run_inference(params: dict) -> tuple[list, np.ndarray, np.ndarray]:
    BCs_bc, BCs_load, W, H, vf = semantic_to_conditions(params)
    b = max(W, H)
    AR = np.array([W / b, H / b])
    Cs_t    = [torch.tensor(AR).float().unsqueeze(0), torch.tensor([vf]).float().unsqueeze(0)]
    BCs_t   = [torch.tensor(BCs_bc).float(), torch.tensor(BCs_load).float()]
    BC_batch = [
        torch.zeros(BCs_bc.shape[0], dtype=torch.long),
        torch.zeros(BCs_load.shape[0], dtype=torch.long),
    ]
    conditions = BatchDict({"Cs": Cs_t, "BCs": BCs_t, "BC_Batch": BC_batch, "unconditioned": False})
    ae_batch   = build_ae_batch(W, H)

    with torch.no_grad():
        gen = _pipeline.inference(
            neural_field_inputs=ae_batch,
            conditions=conditions,
            n_samples=8,
            num_sampling_steps=20,
            classifier_free_guidance=1.0,
            ddpm=False,
            clamp_latents=False,
            remap_latents=False,
            verbose=False,
        )
    return gen[0][0], BCs_bc, BCs_load


def _job_worker(job_id: str, params: dict):
    try:
        samples, BCs_bc, BCs_load = _run_inference(params)
        W, H = params["width"], params["height"]
        images = [_tensor_to_b64(s, W, H) for s in samples]
        _jobs[job_id]["samples"]  = samples
        _jobs[job_id]["BCs_bc"]   = BCs_bc
        _jobs[job_id]["BCs_load"] = BCs_load
        _jobs[job_id]["images"]   = images
        _jobs[job_id]["status"]   = "done"
    except Exception as e:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"]  = str(e)

app = FastAPI(title="oattopology")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    description: str

class RefineRequest(BaseModel):
    job_id: str
    instruction: str

class ExportRequest(BaseModel):
    job_id: str
    sample_index: int  # 0-based


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("ui.html") as f:
        return f.read()


@app.post("/generate")
async def generate(req: GenerateRequest):
    loop = asyncio.get_event_loop()
    try:
        params = await loop.run_in_executor(_executor, parse_problem, req.description, "groq")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM parse failed: {e}")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "params": params}
    loop.run_in_executor(_executor, _job_worker, job_id, params)
    return {"job_id": job_id, "params": params}


@app.post("/refine")
async def refine(req: RefineRequest):
    job = _jobs.get(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    loop = asyncio.get_event_loop()
    try:
        new_params = await loop.run_in_executor(
            _executor, refine_params, job["params"], req.instruction, "groq"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM refine failed: {e}")

    new_job_id = str(uuid.uuid4())
    _jobs[new_job_id] = {"status": "running", "params": new_params}
    loop.run_in_executor(_executor, _job_worker, new_job_id, new_params)
    return {"job_id": new_job_id, "params": new_params}


@app.get("/status/{job_id}")
async def status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "done":
        return {"status": "done", "images": job["images"], "params": job["params"]}
    if job["status"] == "error":
        return {"status": "error", "error": job["error"]}
    return {"status": "running"}


@app.post("/export")
async def export(req: ExportRequest):
    job = _jobs.get(req.job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not ready")
    idx = req.sample_index
    if not (0 <= idx < len(job["samples"])):
        raise HTTPException(status_code=400, detail="Invalid sample index")

    out_path = f"/tmp/oat_export_{req.job_id}_{idx}.svg"
    export_svg(job["samples"][idx], job["params"], filename=out_path)
    return FileResponse(out_path, media_type="image/svg+xml",
                        filename=f"topology_sample_{idx+1}.svg")
