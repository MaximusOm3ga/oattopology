import sys, os
sys.path.insert(0, os.path.abspath("."))

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from ollama import Client
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from OAT.DataUtils._utils import BatchDict
from OAT.Models import NFAE, CTOPUNet
from OAT.DataUtils._utils import make_coord_cell_grid
from OAT.Pipelines.OATPipeline import OATPipeline, DDIMPipeline

SYSTEM_PROMPT = """
You are a topology optimization problem parser.
Return ONLY valid JSON with this schema:

{
  "width": <int>,
  "height": <int>,
  "volume_fraction": <float 0.1-0.9>,
  "fixed_edges": <list of strings, each one of: "left", "right", "top", "bottom">,
  "fixed_corners": <list of strings, each one of: "top-left", "top-right", "bottom-left", "bottom-right">,
  "fixed_points": [{"loc": <"left"|"right"|"top"|"bottom"|"center">, "pos": <0.0-1.0 along that edge>}],
  "load_point": {"loc": <"left"|"right"|"top"|"bottom"|"center">, "pos": <0.0-1.0 along that edge>},
  "load_direction": <"down"|"up"|"left"|"right">
}

Return ONLY the JSON, no explanation, no markdown fences.
"""

REFINEMENT_PROMPT = """
You are a topology optimization problem parser handling a refinement request.
You will be given the current problem parameters as JSON and a user instruction to modify them.
Return ONLY the updated JSON with the same schema, incorporating the requested changes.
Do not change fields that were not mentioned. Return ONLY the JSON, no explanation, no markdown fences.

Current parameters:
{current_params}

User instruction: {instruction}
"""

LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")


def _clean_json(raw: str) -> dict:
    raw = raw.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def _parse_with_ollama(prompt: str, system: str) -> dict:
    client = Client(host=os.environ.get('OLLAMA_HOST', 'http://localhost:11434'))
    response = client.chat(
        model='llama3.2:1b',
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    )
    return _clean_json(response["message"]["content"])


def _parse_with_groq(prompt: str, system: str) -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment")
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    return _clean_json(response.choices[0].message.content)


def parse_problem(description: str, backend: str = None) -> dict:
    backend = (backend or LLM_BACKEND).lower()
    print("Making external call" if backend == "groq" else "Using local inference")
    if backend == "groq":
        return _parse_with_groq(description, SYSTEM_PROMPT)
    elif backend == "ollama":
        return _parse_with_ollama(description, SYSTEM_PROMPT)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")


def refine_params(current_params: dict, instruction: str, backend: str = None) -> dict:
    backend = (backend or LLM_BACKEND).lower()
    system = REFINEMENT_PROMPT.format(
        current_params=json.dumps(current_params, indent=2),
        instruction=instruction
    )
    print("Refining with external call" if backend == "groq" else "Refining with local inference")
    if backend == "groq":
        return _parse_with_groq(instruction, system)
    elif backend == "ollama":
        return _parse_with_ollama(instruction, system)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")


def semantic_to_conditions(params: dict):
    W, H = params["width"], params["height"]
    vf = params["volume_fraction"]

    bc_points = []

    edge_map = {
        "left":   lambda t: (0.0, t),
        "right":  lambda t: (1.0, t),
        "top":    lambda t: (t, 0.0),
        "bottom": lambda t: (t, 1.0),
    }

    corner_map = {
        "top-left":     (0.0, 0.0),
        "top-right":    (1.0, 0.0),
        "bottom-left":  (0.0, 1.0),
        "bottom-right": (1.0, 1.0),
    }

    loc_map = {
        "left":          lambda p: (0.0, p),
        "right":         lambda p: (1.0, p),
        "top":           lambda p: (p, 0.0),
        "bottom":        lambda p: (p, 1.0),
        "center":        lambda p: (0.5, 0.5),
        "top-center":    lambda p: (0.5, 0.0),
        "center-top":    lambda p: (0.5, 0.0),
        "bottom-center": lambda p: (0.5, 1.0),
        "center-bottom": lambda p: (0.5, 1.0),
        "left-center":   lambda p: (0.0, 0.5),
        "center-left":   lambda p: (0.0, 0.5),
        "right-center":  lambda p: (1.0, 0.5),
        "center-right":  lambda p: (1.0, 0.5),
    }

    def normalize(s):
        return s.lower().replace("_", "-").strip()

    for edge in params.get("fixed_edges", []):
        key = normalize(edge)
        if key in edge_map:
            for t in np.linspace(0, 1, 10):
                x, y = edge_map[key](t)
                bc_points.append([x, y, 1, 1])
        elif key in corner_map:
            x, y = corner_map[key]
            bc_points.append([x, y, 1, 1])

    for corner in params.get("fixed_corners", []):
        key = normalize(corner)
        if key in corner_map:
            x, y = corner_map[key]
            bc_points.append([x, y, 1, 1])

    for pt in params.get("fixed_points", []):
        key = normalize(pt["loc"])
        if key in loc_map:
            x, y = loc_map[key](pt["pos"])
            bc_points.append([x, y, 1, 1])

    load_dir_map = {"down": (0.0, -1.0), "up": (0.0, 1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}
    lp = params["load_point"]
    key = normalize(lp["loc"])
    lx, ly = loc_map.get(key, lambda p: (0.5, 0.5))(lp["pos"])
    fx, fy = load_dir_map.get(normalize(params["load_direction"]), (0.0, -1.0))
    load_points = [[lx, ly, fx, fy]]

    BCs_bc = np.array(bc_points, dtype=np.float32) if bc_points else np.zeros((1, 4), dtype=np.float32) - 1.0
    BCs_load = np.array(load_points, dtype=np.float32)

    return BCs_bc, BCs_load, W, H, vf


def build_ae_batch(W: int, H: int, encoder_res: int = 256):
    b = max(W, H)
    rel_w = W / b
    rel_h = H / b

    coord, cell = make_coord_cell_grid(
        (H, W),
        range=[[-rel_w, rel_w], [-rel_h, rel_h]]
    )
    cell[:] = torch.tensor([2 / encoder_res, 2 / encoder_res])

    ae_batch = BatchDict({
        'gt_coord': [coord.unsqueeze(0)],
        'gt_cell':  [cell.unsqueeze(0)],
    })
    return ae_batch


def run_inference(pipeline, params):
    BCs_bc, BCs_load, W, H, vf = semantic_to_conditions(params)

    b = max(W, H)
    AR = np.array([W / b, H / b])
    Cs_t = [torch.tensor(AR).float().unsqueeze(0), torch.tensor([vf]).float().unsqueeze(0)]
    BCs_t = [torch.tensor(BCs_bc).float(), torch.tensor(BCs_load).float()]
    BC_batch = [
        torch.zeros(BCs_bc.shape[0], dtype=torch.long),
        torch.zeros(BCs_load.shape[0], dtype=torch.long)
    ]

    conditions = BatchDict({
        'Cs': Cs_t,
        'BCs': BCs_t,
        'BC_Batch': BC_batch,
        'unconditioned': False
    })

    ae_batch = build_ae_batch(W, H)

    with torch.no_grad():
        gen = pipeline.inference(
            neural_field_inputs=ae_batch,
            conditions=conditions,
            n_samples=8,
            num_sampling_steps=20,
            classifier_free_guidance=1.0,
            ddpm=False,
            clamp_latents=False,
            remap_latents=False,
            verbose=True,
        )

    samples = gen[0][0]
    return samples, BCs_bc, BCs_load


def visualize(samples, params, bc_array, load_array, filename="result.png"):
    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4))
    if n == 1:
        axes = [axes]

    W, H = params["width"], params["height"]

    for i, sample in enumerate(samples):
        topology = sample.squeeze()
        if hasattr(topology, 'numpy'):
            topology = topology.float().numpy()

        ax = axes[i]
        ax.imshow(topology, cmap='gray_r', vmin=0, vmax=1, extent=[0, W, H, 0])

        for bc in bc_array:
            bx, by, dx, dy = bc
            px, py = bx * W, by * H
            color = 'limegreen' if (dx == 1 and dy == 1) else ('tomato' if dx == 1 else 'royalblue')
            ax.plot(px, py, 's', color=color, markersize=8, zorder=5)

        for load in load_array:
            lx, ly, fx, fy = load
            px, py = lx * W, ly * H
            scale = min(W, H) * 0.15
            ax.annotate('', xy=(px + fx * scale, py - fy * scale), xytext=(px, py),
                        arrowprops=dict(arrowstyle='->', color='orange', lw=2.5))

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_title(f"Sample {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"W={W} H={H} VF={params['volume_fraction']}", fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved to {filename}")
    plt.close()


def export_svg(sample, params, filename="result.svg"):
    try:
        from skimage.measure import find_contours
    except ImportError:
        print("scikit-image not installed. Run: pip install scikit-image")
        return

    topology = sample.squeeze()
    if hasattr(topology, 'numpy'):
        topology = topology.float().numpy()

    W, H = params["width"], params["height"]

    contours = find_contours(topology, level=0.5)

    def scale(contour):
        xs = contour[:, 1] / topology.shape[1] * W
        ys = contour[:, 0] / topology.shape[0] * H
        return xs, ys

    paths_svg = []
    for contour in contours:
        xs, ys = scale(contour)
        if len(xs) < 3:
            continue
        d = f"M {xs[0]:.3f},{ys[0]:.3f} "
        d += " ".join(f"L {x:.3f},{y:.3f}" for x, y in zip(xs[1:], ys[1:]))
        d += " Z"
        paths_svg.append(f'  <path d="{d}" fill="#333333" stroke="none"/>')

    svg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{W}mm" height="{H}mm"
     viewBox="0 0 {W} {H}">
  <!-- oattopology export | W={W}mm H={H}mm VF={params['volume_fraction']} -->
{chr(10).join(paths_svg)}
</svg>"""

    with open(filename, "w") as f:
        f.write(svg_content)
    print(f"SVG exported to {filename} ({len(contours)} contour(s), units = mm)")


def pick_sample(samples):
    while True:
        try:
            idx = int(input(f"\nWhich sample to export as SVG? (1-{len(samples)}): ")) - 1
            if 0 <= idx < len(samples):
                return samples[idx]
        except ValueError:
            pass
        print("Invalid choice, try again.")


if __name__ == "__main__":
    print("Loading models...")
    ae   = NFAE.from_pretrained("OpenTO/NFAE")
    ldm  = CTOPUNet.from_pretrained("OpenTO/LDM")
    ddim = DDIMPipeline()
    pipeline = OATPipeline(DDIM=ddim, diffusion_model=ldm, nfae=ae)
    ae.eval()
    ldm.eval()

    description = input("\nDescribe your topology problem:\n> ")
    print("\nExtracting parameters via LLM...")
    params = parse_problem(description)
    print("Parsed:", json.dumps(params, indent=2))

    iteration = 1
    samples = None
    BCs_bc = BCs_load = None

    while True:
        print(f"\n[Iteration {iteration}] Generating topologies")
        samples, BCs_bc, BCs_load = run_inference(pipeline, params)
        visualize(samples, params, BCs_bc, BCs_load, filename=f"result_{iteration}.png")

        print("\nOptions:")
        print("  [r] Refine — describe what to change")
        print("  [e] Export a sample as SVG")
        print("  [q] Quit")
        choice = input("> ").strip().lower()

        if choice == "q":
            print("Done.")
            break

        elif choice == "e":
            selected = pick_sample(samples)
            export_svg(selected, params, filename=f"result_{iteration}.svg")
            again = input("\nContinue refining? (y/n): ").strip().lower()
            if again != "y":
                break

        elif choice == "r":
            instruction = input("What would you like to change?\n> ").strip()
            print("\nRefining parameters...")
            params = refine_params(params, instruction)
            print("Updated params:", json.dumps(params, indent=2))
            iteration += 1

        else:
            print("Unrecognised input, try r / e / q.")