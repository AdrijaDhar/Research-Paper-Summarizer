"""Caption extracted figures/charts with a local vision-language model via MLX.

API pattern verified against Blaizzy/mlx-vlm's own docs/usage.md (load →
apply_chat_template → generate). Model choice (Qwen2.5-VL-7B, 4-bit) follows
DESIGN.md §4's VLM recommendation for chart/figure understanding.
"""

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

MODEL_PATH = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

PROMPT = (
    "Describe this figure from a scientific paper factually and precisely. "
    "State the figure type (e.g. line chart, bar chart, diagram, table image), "
    "axis labels and units if present, and any specific numeric values or "
    "trends that are visually clear. Do not speculate about anything not "
    "directly visible in the image."
)


def load_vlm():
    model, processor = load(MODEL_PATH)
    config = load_config(MODEL_PATH)
    return model, processor, config


def caption_figures(figures: list[dict], vlm: tuple | None = None) -> list[dict]:
    """Pass a preloaded `vlm` (from `load_vlm()`) when processing many papers
    in one run, to avoid reloading the 7B model per paper."""
    if not figures:
        return figures

    model, processor, config = vlm or load_vlm()
    formatted_prompt = apply_chat_template(processor, config, PROMPT, num_images=1)

    for figure in figures:
        output = generate(model, processor, formatted_prompt, [figure["image_path"]], verbose=False)
        figure["caption"] = str(output)

    return figures
