# code/ai/gen_img2txt.py
"""
Image-to-text using Qwen2-VL-7B-Instruct (4-bit quantized, fits RTX 4060 8GB).

Usage:
    python gen_img2txt.py --input image.png
    python gen_img2txt.py --input image.png --prompt "List every object you can see"
    python gen_img2txt.py --input image.png --prompt "Describe the scene" --output out.txt
    python gen_img2txt.py --input image.png --no-4bit   # FP16, needs ~14 GB VRAM
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("gen_img2txt")

DEFAULT_PROMPT = (
    "List every distinct object visible in this image. "
    "For each object give: name, approximate location (left/center/right, foreground/background), "
    "and a one-sentence description. Use a numbered list."
)

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"


def load_model(use_4bit: bool):
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    log.info(f"Loading {MODEL_ID} ({'4-bit' if use_4bit else 'FP16'})...")

    kwargs = {"device_map": "auto"}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    log.info("Model ready.")
    return model, processor


def run(image_path: str, prompt: str, use_4bit: bool, max_new_tokens: int) -> str:
    from PIL import Image

    model, processor = load_model(use_4bit)

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    log.info("Generating...")
    import torch
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    # Trim input tokens from output
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    result = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    return result


def main():
    p = argparse.ArgumentParser(description="Image-to-text via Qwen2-VL-7B")
    p.add_argument("--input",          required=True, help="Input image path")
    p.add_argument("--prompt",         default=DEFAULT_PROMPT, help="Question / instruction for the model")
    p.add_argument("--output",         default=None, help="Save result to this text file (optional)")
    p.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=1024)
    p.add_argument("--no-4bit",        dest="no_4bit", action="store_true",
                   help="Disable 4-bit quantization (needs ~14 GB VRAM)")
    args = p.parse_args()

    if not Path(args.input).exists():
        log.error(f"Input not found: {args.input}")
        sys.exit(1)

    result = run(
        image_path=args.input,
        prompt=args.prompt,
        use_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n--- Result ---")
    print(result)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
        log.info(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
