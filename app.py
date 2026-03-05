"""
DiffKilter Web App

Script launches local Gradio web interface for boulder problem generation.
Loads the trained KilterTransformer, handles user clicks via a KDTree spatial
mapping, and triggers absorbing diffusion process to generate custom routes.

Usage:
    $ python app.py
"""

import gradio as gr
import numpy as np
import torch
from scipy.spatial import KDTree
from PIL import Image, ImageDraw
import os

from dataset import KilterDataset
from model import KilterTransformer
from diffusion import get_noise_schedule, generate_constrained_batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Starting App on device: {device}")

dataset = KilterDataset("./data/kilter_board_climbs.npz")
mean_np = dataset.mean_coords.cpu().numpy()
std_np = dataset.std_coords.cpu().numpy()
xy_raw = (dataset.coords.cpu().numpy() * std_np) + mean_np

model = KilterTransformer(num_classes=6, hidden_dim=256, num_layers=6, nhead=8).to(device)

checkpoint_path = "./weights/kilter_checkpoint_epoch_100.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict']) 
    print("Model weights loaded successfully.")
else:
    print(f"Warning: Checkpoint '{checkpoint_path}' not found! The AI will generate random noise.")

model.eval()

alphas_cumprod = get_noise_schedule(num_timesteps=100, device=device)

x_min, x_max = 0.01, 1.00
y_min, y_max = 0.03, 0.96

try:
    bg_img = Image.open("./image/kilter_bg.png").convert("RGBA")
    W, H = bg_img.size
except FileNotFoundError:
    print("Warning: kilter_bg.png not found!")
    W, H = 800, 1200
    bg_img = Image.new("RGBA", (W, H), (20, 20, 20, 255))

mean_np = dataset.mean_coords.cpu().numpy()
std_np = dataset.std_coords.cpu().numpy()
xy_raw = (dataset.coords.cpu().numpy() * std_np) + mean_np

pixel_coords = np.zeros_like(xy_raw)
pixel_coords[:, 0] = ((xy_raw[:, 0] - x_min) / (x_max - x_min)) * W
pixel_coords[:, 1] = ((xy_raw[:, 1] - y_min) / (y_max - y_min)) * H

hold_tree = KDTree(pixel_coords)

def render_board(route_array=None, constraints=None):
    """
    Renders Kilterboard interface using PIL image compositing.

    Args:
        route_array (np.ndarray, optional): The 476-length array of generated holds.
        constrains (dict, optional) = The user's manually selected holds.

    Returns:
        PIL.Image: The fully composited UI image.
    """

    if constraints is None: constraints = {}

    overlay = Image.new("RGBA", bg_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    r = int(W * 0.025)

    colors = {
        1: (0, 255, 0, 255),    # Start (Green)
        2: (0, 255, 255, 255),  # Hand (Cyan)
        3: (255, 165, 0, 255),  # Foot (Orange)
        4: (255, 0, 255, 255)   # Finish (Pink)
    }

    if route_array is not None:
        for idx in range(len(route_array)):
            cls = route_array[idx]
            x, y = pixel_coords[idx]
            if cls in colors:
                draw.ellipse((x-r, y-r, x+r, y+r), outline=colors[cls], width=5)
            elif cls == 0:
                draw.ellipse((x-2, y-2, x+2, y+2), fill=(255, 255, 255, 100))
    else:
        for idx in range(len(pixel_coords)):
            x, y = pixel_coords[idx]
            draw.ellipse((x-2, y-2, x+2, y+2), fill=(255, 255, 255, 100))

    #  Draw the manual user constraints
    for node_idx, class_id in constraints.items():
        node_idx = int(node_idx)
        x, y = pixel_coords[node_idx]
        draw.ellipse((x-r-2, y-r-2, x+r+2, y+r+2), outline=colors[int(class_id)], width=5)

    return Image.alpha_composite(bg_img, overlay)

#  Format text for UI
def format_active_nodes(constraints):
    if not constraints:
        return "No holds selected."

    names = {1: "Start", 2: "Hand", 3: "Foot", 4: "Finish"}
    lines = []
    for k, v in constraints.items():
        lines.append(f"Node {k}: {names[int(v)]}")
    return " | ".join(lines)

#  Gradio app logic
tool_to_class = {
    "Start": 1,
    "Hand": 2,
    "Foot": 3,
    "Finish": 4
}

def handle_click(evt: gr.SelectData, current_tool, current_constraints):
    """
    Intercepts user clicks on the image and snaps them to nearest valid Kilterboard hold.
    Updates the Gradio state dictionary to store the user's chosen Start/Hand/Foot/End holds.
    
    Args:
        evt (gr.SelectData): The (x, y) pixel coordinates of the user's click.
        current_tool (str): The brush currently selected in the UI (e.g., "Start", "Eraser").
        current_constraints (dict): The active dictionary of forced holds.

    Returns:
        tuple: (Updated Image, Updated Constraints Dict, Formatted Text String)
    """

    click_px_x, click_px_y = evt.index
    distance, node_idx = hold_tree.query([click_px_x, click_px_y])
    node_str = str(node_idx)

    if current_tool == "Eraser":
        if node_str in current_constraints:
            del current_constraints[node_str]
    else:
        current_constraints[node_str] = tool_to_class[current_tool]

    new_image = render_board(route_array=None, constraints=current_constraints)
    active_text = format_active_nodes(current_constraints)

    return new_image, current_constraints, active_text

def run_generation(current_constraints):
    """
    Passes the user's hold constraints to the PyTorch diffusion model to generate a route.
    """

    model_device = next(model.parameters()).device

    #  Convert string keys back to PyTorch ints
    safe_constraints = {int(k): int(v) for k, v in current_constraints.items()}

    print(f"GRADIO IS REQUESTING NODES: {safe_constraints}")

    generated_batch = generate_constrained_batch(
        model,
        dataset.coords.to(model_device),
        alphas_cumprod.to(model_device),
        constraints=safe_constraints,
        batch_size=1
    )

    route_array = generated_batch[0].cpu().numpy()
    final_image = render_board(route_array=route_array, constraints=current_constraints)
    return final_image

def clear_board():
    return render_board(route_array=None, constraints={}), {}, "No holds selected."

#  Build UI
with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    gr.Markdown("# D3PM Kilterboard Setter")

    state_constraints = gr.State(value={})

    with gr.Row():
        with gr.Column(scale=3):
            board_image = gr.Image(
                value=render_board(),
                interactive=False,
                label="Interactive Board"
            )

        with gr.Column(scale=1):
            tool_selector = gr.Radio(
                choices=["Start", "Hand", "Foot", "Finish", "Eraser"],
                value="Start",
                label="Select Brush"
            )

            generate_btn = gr.Button("Generate Route", variant="primary", size="lg")
            clear_btn = gr.Button("Clear Board")

    board_image.select(
        fn=handle_click,
        inputs=[tool_selector, state_constraints],
        outputs=[board_image, state_constraints]
    )

    generate_btn.click(
        fn=run_generation,
        inputs=[state_constraints],
        outputs=[board_image]
    )

    clear_btn.click(
        fn=clear_board,
        inputs=[],
        outputs=[board_image, state_constraints]
    )

if __name__ == "__main__":
    app.launch(debug=True)