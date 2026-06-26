import sys, os
sys.path.insert(0, os.path.abspath("."))

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from ollama import Client


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

def parse_problem(description: str) -> dict:
    client = Client(host=os.environ.get('OLLAMA_HOST', 'http://localhost:11434'))
    response = client.chat(
        model='llama3.2:1b',
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": description}
        ]
    )
    raw = response["message"]["content"].strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)



def build_conditions(params: dict):
    W = params["width"]
    H = params["height"]
    vf = params["volume_fraction"]

    b = max(W, H)
    AR = np.array([W / b, H / b])
    Cs = [AR, np.array([vf])]

    bc_list = [p for p in params["boundary_conditions"] if "dx" in p and "dy" in p]
    load_list = [p for p in params["loads"] if "fx" in p and "fy" in p]

    if bc_list:
        BCs_bc = np.array([[p["x"], p["y"], p["dx"], p["dy"]] for p in bc_list], dtype=np.float32)
    else:
        BCs_bc = np.zeros((1, 4)) - 1.0

    if load_list:
        BCs_load = np.array([[p["x"], p["y"], p["fx"], p["fy"]] for p in load_list], dtype=np.float32)
    else:
        BCs_load = np.zeros((1, 4)) - 1.0

    BCs = [BCs_bc, BCs_load]
    sizes = [BCs_bc.shape[0], BCs_load.shape[0]]

    Cs_t = [torch.tensor(c).float().unsqueeze(0) if c.ndim == 1
            else torch.tensor(c).float() for c in Cs]
    for i in range(len(Cs_t)):
        if Cs_t[i].dim() == 1:
            Cs_t[i] = Cs_t[i].unsqueeze(1)

    BCs_t = [torch.tensor(bc.astype(np.float32)).float() for bc in BCs]
    BC_batch = [
        torch.zeros(BCs_bc.shape[0], dtype=torch.long),
        torch.zeros(BCs_load.shape[0], dtype=torch.long)
    ]

    from OAT.DataUtils._utils import BatchDict
    conditions = BatchDict({
        'Cs': Cs_t,
        'BCs': BCs_t,
        'BC_Batch': BC_batch,
        'unconditioned': False
    })

    return conditions, W, H



def build_ae_batch(W: int, H: int, encoder_res: int = 256):
    b = max(W, H)
    rel_w = W / b
    rel_h = H / b

    coord, cell = make_coord_cell_grid(
        (H, W),
        range=[[-rel_w, rel_w], [-rel_h, rel_h]]
    )
    cell[:] = torch.tensor([2 / encoder_res, 2 / encoder_res])


    from OAT.DataUtils._utils import BatchDict
    ae_batch = BatchDict({
        'gt_coord': [coord.unsqueeze(0)],
        'gt_cell':  [cell.unsqueeze(0)],
    })
    return ae_batch



def visualize(samples, params: dict, bc_array: np.ndarray, load_array: np.ndarray):

    print(type(gen))
    print(type(gen[0]))
    print(type(gen[0][0]) if isinstance(gen[0], list) else gen[0].shape)

    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4))
    if n == 1:
        axes = [axes]

    W, H = params["width"], params["height"]

    for i, sample in enumerate(samples):
        if hasattr(sample, 'squeeze'):
            topology = sample.squeeze()
            if hasattr(topology, 'numpy'):
                topology = topology.float().numpy()
        else:
            topology = sample.squeeze()

        ax = axes[i]

        ax.imshow(topology, cmap='gray_r', vmin=0, vmax=1,
                  extent=[0, W, H, 0])

        for bc in bc_array:
            bx, by, dx, dy = bc
            px, py = bx * W, by * H

            if dx == 1 and dy == 1:
                color = 'limegreen'
                label = 'Fixed XY'
            elif dx == 1:
                color = 'tomato'
                label = 'Fixed X'
            else:
                color = 'royalblue'
                label = 'Fixed Y'

            ax.plot(px, py, 's', color=color, markersize=8, zorder=5)

        for load in load_array:
            lx, ly, fx, fy = load
            px, py = lx * W, ly * H
            scale = min(W, H) * 0.15
            ax.annotate('',
                xy=(px + fx * scale, py - fy * scale),
                xytext=(px, py),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2.5))

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_title(f"Sample {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])

    title = f"W={W} H={H} VF={params['volume_fraction']}"
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig("result.png", dpi=150)
    plt.show()
    print("Saved to result.png")

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

    # Fixed edges
    for edge in params.get("fixed_edges", []):
        key = normalize(edge)
        if key in edge_map:
            for t in np.linspace(0, 1, 10):
                x, y = edge_map[key](t)
                bc_points.append([x, y, 1, 1])
        elif key in corner_map:
            x, y = corner_map[key]
            bc_points.append([x, y, 1, 1])
        else:
            print(f"Warning: unknown edge '{edge}', skipping")

    for corner in params.get("fixed_corners", []):
        key = normalize(corner)
        if key in corner_map:
            x, y = corner_map[key]
            bc_points.append([x, y, 1, 1])
        elif key in edge_map:
            for t in np.linspace(0, 1, 10):
                x, y = edge_map[key](t)
                bc_points.append([x, y, 1, 1])
        else:
            print(f"Warning: unknown corner '{corner}', skipping")

    for pt in params.get("fixed_points", []):
        key = normalize(pt["loc"])
        if key in loc_map:
            x, y = loc_map[key](pt["pos"])
            bc_points.append([x, y, 1, 1])
        else:
            print(f"Warning: unknown loc '{pt['loc']}', skipping")

    load_dir_map = {"down": (0.0, -1.0), "up": (0.0, 1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}
    lp = params["load_point"]
    key = normalize(lp["loc"])
    lx, ly = loc_map.get(key, lambda p: (0.5, 0.5))(lp["pos"])
    fx, fy = load_dir_map.get(normalize(params["load_direction"]), (0.0, -1.0))
    load_points = [[lx, ly, fx, fy]]

    BCs_bc = np.array(bc_points, dtype=np.float32) if bc_points else np.zeros((1, 4), dtype=np.float32) - 1.0
    BCs_load = np.array(load_points, dtype=np.float32)

    return BCs_bc, BCs_load, W, H, vf

if __name__ == "__main__":
    print("Loading models")
    ae  = NFAE.from_pretrained("OpenTO/NFAE")
    ldm = CTOPUNet.from_pretrained("OpenTO/LDM")
    ddim = DDIMPipeline()
    pipeline = OATPipeline(DDIM=ddim, diffusion_model=ldm, nfae=ae)
    ae.eval()
    ldm.eval()

    description = input("Describe topology problem:\n> ")

    print("\nExtracting parameters via LLM")
    params = parse_problem(description)
    print("Parsed:", json.dumps(params, indent=2))

    BCs_bc, BCs_load, W, H, vf = semantic_to_conditions(params)
    params["width"] = W
    params["height"] = H
    params["volume_fraction"] = vf

    b = max(W, H)
    AR = np.array([W / b, H / b])
    Cs_t = [torch.tensor(AR).float().unsqueeze(0), torch.tensor([vf]).float().unsqueeze(0)]
    BCs_t = [torch.tensor(BCs_bc).float(), torch.tensor(BCs_load).float()]
    BC_batch = [torch.zeros(BCs_bc.shape[0], dtype=torch.long),
                torch.zeros(BCs_load.shape[0], dtype=torch.long)]

    conditions = BatchDict({
        'Cs': Cs_t,
        'BCs': BCs_t,
        'BC_Batch': BC_batch,
        'unconditioned': False
    })

    ae_batch = build_ae_batch(W, H)

    print("\nGenerating topologies")

    coord, cell = make_coord_cell_grid((100, 200))
    print("coord shape:", coord.shape)
    print("cell shape:", cell.shape)

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

    BCs_bc, BCs_load, W, H, vf = semantic_to_conditions(params)

    samples = gen[0][0]

    print(len(gen[0]))
    print(len(gen[0][0]))
    print(type(gen[0][0][0]))

    # print("AR:", Cs_t[0])
    # print("VF:", Cs_t[1])
    # print("BCs_bc:", BCs_bc)
    # print("BCs_load:", BCs_load)

    visualize(samples, params, BCs_bc, BCs_load)


