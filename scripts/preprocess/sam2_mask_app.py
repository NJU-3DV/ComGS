# Modified from https://github.com/YunxuanMao/SAM2-GUI.git

import colorsys
import html
import os

import cv2
import gradio as gr
import imageio.v2 as iio
import numpy as np
import torch
from loguru import logger as guru
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2_video_predictor_hf


APP_CSS = """
.gradio-container {
    background: #f3f5f7;
    color: #0f172a;
}

.app-hero {
    margin-bottom: 12px;
    padding: 0 0 12px;
    border-bottom: 1px solid #dbe3ea;
}

.app-hero h1 {
    margin: 0 0 6px;
    font-size: 28px;
    line-height: 1.1;
    letter-spacing: -0.03em;
}

.app-hero p {
    margin: 0;
    max-width: 760px;
    font-size: 14px;
    color: #475569;
}

.workflow-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 12px;
}

.workflow-step {
    display: flex;
    align-items: center;
    gap: 10px;
    min-height: 54px;
    padding: 10px 14px;
    border-radius: 16px;
    border: 1px solid #dbe3ea;
    background: #ffffff;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
}

.workflow-step.active {
    border-color: #93c5fd;
    background: #eff6ff;
    box-shadow: 0 10px 28px rgba(37, 99, 235, 0.08);
}

.workflow-step.done {
    border-color: #cbd5e1;
    background: #f8fafc;
}

.workflow-step.pending {
    opacity: 0.92;
}

.workflow-label {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex: 0 0 26px;
    width: 26px;
    height: 26px;
    border-radius: 999px;
    background: #e2e8f0;
    font-size: 12px;
    font-weight: 700;
    color: #334155;
}

.workflow-step.active .workflow-label {
    background: #2563eb;
    color: #ffffff;
}

.workflow-step.done .workflow-label {
    background: #cbd5e1;
    color: #0f172a;
}

.workflow-copy {
    min-width: 0;
}

.workflow-step h3 {
    margin: 0;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: -0.02em;
}

.workflow-step span.meta {
    display: block;
    margin-top: 2px;
    font-size: 12px;
    color: #64748b;
}

.status-card {
    margin-bottom: 14px;
    padding: 12px 16px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    background: #ffffff;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.03);
}

.status-card .eyebrow {
    margin-bottom: 4px;
    font-size: 12px;
    font-weight: 600;
    color: #2563eb;
}

.status-card .message {
    margin: 0;
    font-size: 15px;
    line-height: 1.5;
}

.status-card .hint {
    margin: 4px 0 0;
    font-size: 13px;
    color: #64748b;
}

.step-panel,
.preview-panel {
    border-radius: 16px;
    padding: 18px 18px !important;
}

.step-panel {
    border: none;
    background: transparent;
    box-shadow: none;
}

.preview-panel {
    border: 1px solid #e2e8f0;
    background: #ffffff;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.03);
}

.step-panel .gradio-markdown,
.preview-panel .gradio-markdown {
    margin: 0 0 16px !important;
}

.step-panel .gradio-markdown:first-child,
.preview-panel .gradio-markdown:first-child {
    margin-top: 10px !important;
}

.step-panel .gradio-markdown:last-child,
.preview-panel .gradio-markdown:last-child {
    margin-bottom: 0 !important;
}

.step-panel .prose,
.preview-panel .prose {
    margin: 0 auto;
}

.step-panel h3,
.preview-panel h3 {
    margin: 0 0 8px !important;
    line-height: 1.3;
    text-align: center;
}

.step-panel p,
.preview-panel p {
    margin: 0 !important;
    color: #64748b;
    line-height: 1.55;
    text-align: center;
}

.point-mode-radio {
    margin: 2px 0 10px;
}

.point-mode-radio label {
    background: transparent;
    border: none;
    border-radius: 0;
    box-shadow: none;
    padding: 0;
}

button.primary {
    background: #0f172a;
}

button.secondary {
    background: #ffffff;
    color: #0f172a;
    border: 1px solid #cbd5e1;
}
"""


def configure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("sam2_mask_app requires a CUDA-enabled PyTorch environment.")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def read_image_rgb(path: str) -> np.ndarray:
    image = iio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    return image


class PromptGUI(object):
    def __init__(self, checkpoint_dir, model_cfg, model_id):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir) if checkpoint_dir else None
        self.model_cfg = model_cfg
        self.model_id = model_id
        self.sam_model = None

        self.selected_points = []
        self.selected_labels = []
        self.cur_label_val = 1.0

        self.frame_index = 0
        self.image = None
        self.cur_mask_idx = 0
        # can store multiple object masks
        # saves the masks and logits for each mask index
        self.cur_masks = {}
        self.cur_logits = {}
        self.index_masks_all = []
        self.color_masks_all = []
        self.inference_state = None
        self.features_ready = False
        self.tracking_done = False
        self.output_video_path = None

        self.img_dir = ""
        self.img_paths = []
        self.init_sam_model()

    def init_sam_model(self):
        if self.sam_model is None:
            configure_cuda()
            if self.checkpoint_dir:
                self.sam_model = build_sam2_video_predictor(
                    self.model_cfg, self.checkpoint_dir
                )
                guru.info(f"loaded local model checkpoint {self.checkpoint_dir}")
            else:
                self.sam_model = build_sam2_video_predictor_hf(self.model_id)
                guru.info(f"loaded SAM2 model from Hugging Face: {self.model_id}")

    def invalidate_tracking(self):
        self.tracking_done = False
        self.output_video_path = None
        self.index_masks_all = []
        self.color_masks_all = []

    def clear_points(self) -> tuple[None, None, str]:
        self.selected_points.clear()
        self.selected_labels.clear()
        self.invalidate_tracking()
        message = "Point prompts cleared. Add new points on the current frame."
        return None, None, message

    def add_new_mask(self):
        self.cur_mask_idx += 1
        self.clear_points()
        message = f"Creating new mask with index {self.cur_mask_idx}"
        return None, message

    def make_index_mask(self, masks):
        # Only support one object: foreground=255, background=0
        assert len(masks) > 0
        first_mask = list(masks.values())[0]
        idx_mask = np.zeros_like(first_mask, dtype="uint8")
        idx_mask[first_mask > 0] = 255
        return idx_mask

    def _clear_image(self):
        """
        Clear the current image state, prompts, and propagated outputs.
        """
        self.image = None
        self.cur_mask_idx = 0
        self.frame_index = 0
        self.selected_points.clear()
        self.selected_labels.clear()
        self.cur_masks = {}
        self.cur_logits = {}
        self.inference_state = None
        self.features_ready = False
        self.invalidate_tracking()

    def reset(self):
        state = self.inference_state
        self._clear_image()
        if self.sam_model is not None and state is not None:
            self.sam_model.reset_state(state)

    def set_img_dir(self, img_dir: str) -> int:
        self._clear_image()
        self.img_dir = img_dir.strip().strip('"').strip("'")
        if not os.path.isdir(self.img_dir):
            self.img_paths = []
            return 0
        self.img_paths = [
            os.path.join(self.img_dir, p)
            for p in sorted(os.listdir(self.img_dir))
            if isimage(p)
        ]

        return len(self.img_paths)

    def set_input_image(self, i: int = 0) -> np.ndarray | None:
        guru.debug(f"Setting frame {i} / {len(self.img_paths)}")
        if i < 0 or i >= len(self.img_paths):
            return self.image
        self.clear_points()
        self.frame_index = i
        image = read_image_rgb(self.img_paths[i])
        self.image = image

        return image

    def get_sam_features(self) -> tuple[str, np.ndarray | None]:
        if not self.img_paths:
            return "Load a valid frame folder before extracting SAM features.", None
        self.selected_points.clear()
        self.selected_labels.clear()
        self.invalidate_tracking()
        self.inference_state = self.sam_model.init_state(video_path=self.img_dir)
        self.sam_model.reset_state(self.inference_state)
        self.features_ready = True
        msg = (
            "SAM features are ready. Click on the frame to build a mask preview."
        )
        return msg, self.image

    def set_positive(self) -> str:
        self.cur_label_val = 1.0
        return "Positive point mode is active."

    def set_negative(self) -> str:
        self.cur_label_val = 0.0
        return "Negative point mode is active."

    def add_point(self, frame_idx, i, j):
        """
        Update the current mask preview with a new point prompt.
        """
        self.invalidate_tracking()
        self.selected_points.append([j, i])
        self.selected_labels.append(self.cur_label_val)
        # masks, scores, logits if we want to update the mask
        masks = self.get_sam_mask(
            frame_idx,
            np.array(self.selected_points, dtype=np.float32),
            np.array(self.selected_labels, dtype=np.int32),
        )
        mask = self.make_index_mask(masks)

        return mask

    def get_sam_mask(self, frame_idx, input_points, input_labels):
        """
        :param frame_idx int
        :param input_points (np array) (N, 2)
        :param input_labels (np array) (N,)
        return (H, W) mask, (H, W) logits
        """
        assert self.sam_model is not None
        assert self.inference_state is not None

        configure_cuda()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, out_obj_ids, out_mask_logits = self.sam_model.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=self.cur_mask_idx,
                points=input_points,
                labels=input_labels,
            )

        return {
            out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    def run_tracker(self) -> tuple[str, str]:
        images = [read_image_rgb(p) for p in self.img_paths]
        video_segments = {}
        configure_cuda()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam_model.propagate_in_video(
                self.inference_state, start_frame_idx=0
            ):
                masks = {
                    out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                video_segments[out_frame_idx] = masks
        ordered_segments = [video_segments[idx] for idx in sorted(video_segments)]
        self.index_masks_all = [self.make_index_mask(masks) for masks in ordered_segments]
        out_frames, self.color_masks_all = colorize_masks(images, self.index_masks_all)
        out_vidpath = "tracked_colors.mp4"
        iio.mimwrite(out_vidpath, out_frames)
        self.output_video_path = out_vidpath
        self.tracking_done = True
        message = f"Wrote the tracked preview to {out_vidpath}."
        instruct = "Review the result, then save the masks if it looks correct."
        return out_vidpath, f"{message} {instruct}"

    def save_masks_to_dir(self, output_dir: str) -> str:
        if not self.index_masks_all:
            return "No propagated masks are available yet. Run propagation first."
        os.makedirs(output_dir, exist_ok=True)
        for img_path, id_mask in zip(self.img_paths, self.index_masks_all):
            name = os.path.basename(img_path)
            out_path = os.path.join(output_dir, os.path.splitext(name)[0] + ".png")
            # Save as 0-255 PNG mask
            iio.imwrite(out_path, id_mask.astype(np.uint8))
        message = f"Saved masks to {output_dir}!"
        guru.debug(message)
        return message


def isimage(p):
    ext = os.path.splitext(p.lower())[-1]
    return ext in [".png", ".jpg", ".jpeg"]


def draw_points(img, points, labels):
    out = img.copy()
    for p, label in zip(points, labels):
        x, y = int(p[0]), int(p[1])
        color = (0, 0, 255) if label == 1.0 else (255, 0, 0)
        out = cv2.circle(out, (x, y), 10, color, -1)
    return out


def get_hls_palette(
    n_colors: int,
    lightness: float = 0.5,
    saturation: float = 0.7,
) -> np.ndarray:
    """
    returns (n_colors, 3) tensor of colors,
        first is black and the rest are evenly spaced in HLS space
    """
    hues = np.linspace(0, 1, int(n_colors) + 1)[1:-1]  # (n_colors - 1)
    palette = [(0.0, 0.0, 0.0)] + [
        colorsys.hls_to_rgb(h_i, lightness, saturation) for h_i in hues
    ]
    return (255 * np.asarray(palette)).astype("uint8")


def colorize_masks(images, index_masks, fac: float = 0.5):
    color_masks = []
    out_frames = []
    for img, mask in zip(images, index_masks):
        # mask: 0 (bg), 255 (fg)
        clr_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        clr_mask[mask == 255] = [255, 0, 0]  # Red for object
        color_masks.append(clr_mask)
        out_u = compose_img_mask(img, clr_mask, fac)
        out_frames.append(out_u)
    return out_frames, color_masks


def compose_img_mask(img, color_mask, fac: float = 0.5):
    out_f = fac * img / 255 + (1 - fac) * color_mask / 255
    out_u = (255 * out_f).astype("uint8")
    return out_u


def build_status_html(title: str, message: str, hint: str = "") -> str:
    hint_html = f"<p class='hint'>{html.escape(hint)}</p>" if hint else ""
    return f"""
    <div class="status-card">
        <div class="eyebrow">{html.escape(title)}</div>
        <p class="message">{html.escape(message)}</p>
        {hint_html}
    </div>
    """


def build_workflow_html(
    folder_loaded: bool,
    features_ready: bool,
    has_points: bool,
    tracking_done: bool,
) -> str:
    steps = [
        ("1", "Load frames", "Folder"),
        ("2", "Extract features", "SAM2"),
        ("3", "Refine mask", "Points"),
        ("4", "Propagate", "Export"),
    ]
    completed = [folder_loaded, features_ready, has_points, tracking_done]
    if tracking_done:
        active_index = None
    elif not folder_loaded:
        active_index = 0
    elif not features_ready:
        active_index = 1
    elif not has_points:
        active_index = 2
    else:
        active_index = 3

    cards = []
    for idx, (label, title, meta) in enumerate(steps):
        if completed[idx]:
            state = "done"
        elif active_index == idx:
            state = "active"
        else:
            state = "pending"
        cards.append(
            f"""
            <div class="workflow-step {state}">
                <span class="workflow-label">{html.escape(label)}</span>
                <div class="workflow-copy">
                    <h3>{html.escape(title)}</h3>
                    <span class="meta">{html.escape(meta)}</span>
                </div>
            </div>
            """
        )
    return f"<div class='workflow-grid'>{''.join(cards)}</div>"


def make_demo(checkpoint_dir, model_cfg, model_id):
    prompts = PromptGUI(checkpoint_dir, model_cfg, model_id)
    start_message = "Enter the folder path containing ordered PNG or JPG frames."
    start_hint = "Switching folders clears the current session."

    def build_default_status() -> str:
        if not prompts.img_paths:
            return build_status_html("Step 1 of 4", start_message, start_hint)
        if not prompts.features_ready:
            return build_status_html(
                "Step 2 of 4",
                f"Loaded {len(prompts.img_paths)} frames. Pick a frame, then extract SAM features.",
                "",
            )
        if not prompts.selected_points:
            return build_status_html(
                "Step 3 of 4",
                "Choose a point type and click on the frame to build the first mask preview.",
                "",
            )
        if not prompts.tracking_done:
            return build_status_html(
                "Step 4 of 4",
                "Refine the preview until it looks right, then propagate through the sequence.",
                "",
            )
        return build_status_html(
            "Step 4 of 4",
            "Review the tracked preview and save the masks when you are satisfied.",
            "Masks are written to a `masks` folder next to the frame directory.",
        )

    def get_point_mode_value() -> str:
        return "Positive point" if prompts.cur_label_val == 1.0 else "Negative point"

    def build_ui_updates() -> dict[str, object]:
        folder_loaded = len(prompts.img_paths) > 0
        features_ready = prompts.features_ready
        has_points = len(prompts.selected_points) > 0
        tracking_done = prompts.tracking_done
        max_frame = max(len(prompts.img_paths) - 1, 0)
        return {
            "workflow": build_workflow_html(
                folder_loaded=folder_loaded,
                features_ready=features_ready,
                has_points=has_points,
                tracking_done=tracking_done,
            ),
            "point_mode": gr.update(
                value=get_point_mode_value(),
                interactive=features_ready,
            ),
            "frame_index": gr.update(
                minimum=0,
                maximum=max_frame,
                value=min(prompts.frame_index, max_frame),
                step=1,
                interactive=folder_loaded,
            ),
            "input_image": gr.update(value=prompts.image, interactive=features_ready),
            "sam_button": gr.update(interactive=folder_loaded),
            "reset_button": gr.update(interactive=folder_loaded),
            "clear_button": gr.update(interactive=features_ready),
            "submit_button": gr.update(interactive=features_ready and has_points),
            "save_button": gr.update(interactive=tracking_done),
        }

    with gr.Blocks(css=APP_CSS, title="SAM2 Mask Workflow") as demo:
        gr.HTML(
            """
            <div class="app-hero">
                <h1>SAM2 Mask Workflow</h1>
                <p>Refine one frame with clicks, propagate the result through the sequence, and export binary masks.</p>
            </div>
            """
        )
        workflow = gr.HTML(build_workflow_html(False, False, False, False))
        instruction = gr.HTML(build_status_html("Step 1 of 4", start_message, start_hint))

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=340):
                with gr.Group(elem_classes=["step-panel"]):
                    gr.Markdown(
                        "### Step 1 · Load frames\n"
                        "Paste the folder path that contains ordered frame images."
                    )
                    folder_path_display = gr.Textbox(
                        "",
                        label="Frame folder path",
                        placeholder=r"D:\data\sequence\frames",
                    )
                    frame_index = gr.Slider(
                        label="Frame index",
                        minimum=0,
                        maximum=0,
                        value=0,
                        step=1,
                        interactive=False,
                    )
                    reset_button = gr.Button("Reset session", interactive=False)

                with gr.Group(elem_classes=["step-panel"]):
                    gr.Markdown(
                        "### Step 2 · Extract features\n"
                        "Initialize SAM2 for the current frame sequence."
                    )
                    sam_button = gr.Button(
                        "Extract SAM features",
                        variant="primary",
                        interactive=False,
                    )

                with gr.Group(elem_classes=["step-panel"]):
                    gr.Markdown(
                        "### Step 3 · Refine mask\n"
                        "Set the point type, then click directly on the frame."
                    )
                    point_mode = gr.Radio(
                        choices=["Positive point", "Negative point"],
                        value="Positive point",
                        label="Point type",
                        interactive=False,
                        elem_classes=["point-mode-radio"],
                    )
                    clear_button = gr.Button("Clear points", interactive=False)

                with gr.Group(elem_classes=["step-panel"]):
                    gr.Markdown(
                        "### Step 4 · Propagate and save\n"
                        "Run tracking on the full sequence, then export the masks."
                    )
                    submit_button = gr.Button(
                        "Propagate through video",
                        variant="primary",
                        interactive=False,
                    )
                    save_button = gr.Button("Save masks", interactive=False)

            with gr.Column(scale=7, min_width=520):
                with gr.Group(elem_classes=["preview-panel"]):
                    gr.Markdown("### Frame workspace")
                    with gr.Row():
                        input_image = gr.Image(
                            None,
                            label="Current frame",
                            interactive=False,
                            type="numpy",
                        )
                        output_img = gr.Image(
                            label="Mask preview",
                            interactive=False,
                            type="numpy",
                        )

                with gr.Group(elem_classes=["preview-panel"]):
                    gr.Markdown("### Propagation result")
                    final_video = gr.Video(label="Tracked video preview")

        def on_select_folder(img_dir):
            num_imgs = prompts.set_img_dir(img_dir)
            if num_imgs > 0:
                prompts.set_input_image(0)
            ui = build_ui_updates()
            if num_imgs == 0:
                status_html = build_status_html(
                    "Step 1 of 4",
                    "No PNG or JPG frames were found in that folder.",
                    "Enter a valid folder path that contains ordered image files.",
                )
            else:
                status_html = build_status_html(
                    "Step 2 of 4",
                    f"Loaded {num_imgs} frames. Pick a frame, then extract SAM features.",
                    "",
                )
            return (
                prompts.img_dir,
                ui["frame_index"],
                ui["input_image"],
                gr.update(value=None),
                gr.update(value=None),
                status_html,
                ui["workflow"],
                ui["point_mode"],
                ui["sam_button"],
                ui["reset_button"],
                ui["clear_button"],
                ui["submit_button"],
                ui["save_button"],
            )

        def on_frame_change(frame_idx):
            prompts.set_input_image(frame_idx)
            ui = build_ui_updates()
            return (
                ui["input_image"],
                gr.update(value=None),
                gr.update(value=None),
                build_default_status(),
                ui["workflow"],
                ui["point_mode"],
                ui["submit_button"],
                ui["save_button"],
            )

        def on_extract_features():
            if not prompts.img_paths:
                ui = build_ui_updates()
                return (
                    ui["input_image"],
                    gr.update(value=None),
                    gr.update(value=None),
                    build_status_html(
                        "Step 1 of 4",
                        "Load a valid frame folder before extracting SAM features.",
                        "",
                    ),
                    ui["workflow"],
                    ui["point_mode"],
                    ui["sam_button"],
                    ui["reset_button"],
                    ui["clear_button"],
                    ui["submit_button"],
                    ui["save_button"],
                )
            prompts.get_sam_features()
            ui = build_ui_updates()
            return (
                ui["input_image"],
                gr.update(value=None),
                gr.update(value=None),
                build_status_html(
                    "Step 3 of 4",
                    "SAM features are ready. Choose a point type and click on the frame.",
                    "",
                ),
                ui["workflow"],
                ui["point_mode"],
                ui["sam_button"],
                ui["reset_button"],
                ui["clear_button"],
                ui["submit_button"],
                ui["save_button"],
            )

        def on_point_mode_change(mode):
            if mode == "Negative point":
                prompts.set_negative()
                if prompts.selected_points:
                    return build_status_html(
                        "Step 4 of 4",
                        "Negative point mode is active. Keep refining or propagate through the sequence.",
                        "",
                    )
                return build_status_html(
                    "Step 3 of 4",
                    "Negative point mode is active. Click on unwanted regions to trim the mask.",
                    "",
                )
            prompts.set_positive()
            if prompts.selected_points:
                return build_status_html(
                    "Step 4 of 4",
                    "Positive point mode is active. Keep refining or propagate through the sequence.",
                    "",
                )
            return build_status_html(
                "Step 3 of 4",
                "Positive point mode is active. Click on the object to strengthen the mask.",
                "",
            )

        def get_select_coords(frame_idx, img, evt: gr.SelectData):
            if img is None:
                ui = build_ui_updates()
                return (
                    gr.update(value=None),
                    gr.update(value=None),
                    build_default_status(),
                    ui["workflow"],
                    ui["submit_button"],
                    ui["save_button"],
                )
            i = evt.index[1]
            j = evt.index[0]
            index_mask = prompts.add_point(frame_idx, i, j)
            color_mask = np.zeros((*index_mask.shape, 3), dtype=np.uint8)
            color_mask[index_mask == 255] = [255, 0, 0]
            out_u = compose_img_mask(img, color_mask)
            out = draw_points(out_u, prompts.selected_points, prompts.selected_labels)
            ui = build_ui_updates()
            return (
                gr.update(value=out),
                gr.update(value=None),
                build_status_html(
                    "Step 4 of 4",
                    "Preview updated. Keep refining or propagate through the sequence.",
                    "",
                ),
                ui["workflow"],
                ui["submit_button"],
                ui["save_button"],
            )

        def on_clear_points():
            prompts.clear_points()
            ui = build_ui_updates()
            return (
                gr.update(value=None),
                gr.update(value=None),
                build_default_status(),
                ui["workflow"],
                ui["point_mode"],
                ui["submit_button"],
                ui["save_button"],
            )

        def on_propagate():
            if not prompts.selected_points:
                ui = build_ui_updates()
                return (
                    gr.update(value=None),
                    build_status_html(
                        "Step 3 of 4",
                        "Add at least one point on the current frame before propagation.",
                        "",
                    ),
                    ui["workflow"],
                    ui["save_button"],
                )
            out_vidpath, _ = prompts.run_tracker()
            ui = build_ui_updates()
            return (
                gr.update(value=out_vidpath),
                build_status_html(
                    "Step 4 of 4",
                    f"Propagation finished. Preview written to {out_vidpath}.",
                    "",
                ),
                ui["workflow"],
                ui["save_button"],
            )

        def on_save_masks():
            img_dir = prompts.img_dir
            parent_dir = os.path.dirname(img_dir.rstrip("/\\"))
            output_dir = os.path.join(parent_dir, "masks")
            message = prompts.save_masks_to_dir(output_dir)
            return build_status_html("Step 4 of 4", message, "")

        def on_reset_session():
            prompts.reset()
            if prompts.img_paths:
                prompts.set_input_image(0)
            ui = build_ui_updates()
            return (
                ui["frame_index"],
                ui["input_image"],
                gr.update(value=None),
                gr.update(value=None),
                build_status_html(
                    "Step 2 of 4",
                    "Session reset. Extract SAM features again to continue.",
                    "",
                ),
                ui["workflow"],
                ui["point_mode"],
                ui["sam_button"],
                ui["reset_button"],
                ui["clear_button"],
                ui["submit_button"],
                ui["save_button"],
            )

        folder_path_display.change(
            on_select_folder,
            [folder_path_display],
            [
                folder_path_display,
                frame_index,
                input_image,
                output_img,
                final_video,
                instruction,
                workflow,
                point_mode,
                sam_button,
                reset_button,
                clear_button,
                submit_button,
                save_button,
            ],
        )
        frame_index.change(
            on_frame_change,
            [frame_index],
            [
                input_image,
                output_img,
                final_video,
                instruction,
                workflow,
                point_mode,
                submit_button,
                save_button,
            ],
        )
        input_image.select(
            get_select_coords,
            [frame_index, input_image],
            [
                output_img,
                final_video,
                instruction,
                workflow,
                submit_button,
                save_button,
            ],
        )
        sam_button.click(
            on_extract_features,
            outputs=[
                input_image,
                output_img,
                final_video,
                instruction,
                workflow,
                point_mode,
                sam_button,
                reset_button,
                clear_button,
                submit_button,
                save_button,
            ],
        )
        point_mode.change(on_point_mode_change, [point_mode], [instruction])
        clear_button.click(
            on_clear_points,
            outputs=[
                output_img,
                final_video,
                instruction,
                workflow,
                point_mode,
                submit_button,
                save_button,
            ],
        )
        submit_button.click(
            on_propagate,
            outputs=[final_video, instruction, workflow, save_button],
        )
        save_button.click(on_save_masks, outputs=[instruction])
        reset_button.click(
            on_reset_session,
            outputs=[
                frame_index,
                input_image,
                output_img,
                final_video,
                instruction,
                workflow,
                point_mode,
                sam_button,
                reset_button,
                clear_button,
                submit_button,
                save_button,
            ],
        )
    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/sam2-hiera-large",
    )
    args = parser.parse_args()
    demo = make_demo(args.checkpoint_dir, args.model_cfg, args.model_id)
    demo.launch(server_port=args.port)
