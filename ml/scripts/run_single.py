import argparse
import sys
import os

# Add ml folder to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dip_core.config import DIPConfig
from dip_core.pipeline.trainer import train_dip
from dip_core.pipeline.postprocess import save_image


def main():
    parser = argparse.ArgumentParser(
        description="Run Deep Image Prior on a single thermal image (denoise or inpaint)."
    )
    parser.add_argument("--input", required=True, help="Path to input degraded thermal image")
    parser.add_argument("--mask", default=None, help="Path to optional binary mask for inpainting")
    parser.add_argument("--reference", default=None, help="Path to optional clean reference image")
    parser.add_argument("--task", default="denoise", choices=["denoise", "inpaint"], help="Restoration task")
    parser.add_argument("--model", default="unet", choices=["unet", "conv"], help="Model architecture (unet or conv)")
    parser.add_argument("--iters", type=int, default=1500, help="Number of DIP iterations")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--output", default="output_restored.png", help="Path to save restored image")
    parser.add_argument("--device", default="cuda", help="Target device (cuda or cpu)")

    args = parser.parse_args()

    model_arch = "conv_baseline" if args.model == "conv" else "unet"
    config = DIPConfig(
        iterations=args.iters,
        learning_rate=args.lr,
        model_type=model_arch,
        device=args.device,
    )

    print("=============================================================")
    print("      DEEP IMAGE PRIOR: SINGLE THERMAL RESTORATION           ")
    print("=============================================================")
    print(f"Input: {args.input}")
    print(f"Task: {args.task} | Model: {args.model} | Iterations: {args.iters}")

    def progress_callback(info):
        if info["iteration"] % 50 == 0 or info["iteration"] == 1:
            print(
                f"[{info['iteration']:>4}/{info['total_iterations']}] "
                f"Loss: {info['loss']:.6f} | "
                f"PSNR proxy: {info['psnr']:.2f} dB | "
                f"Elapsed: {info['elapsed_sec']:.1f}s"
            )

    result = train_dip(
        image_input=args.input,
        mask_input=args.mask,
        reference_input=args.reference,
        config=config,
        task=args.task,
        on_progress=progress_callback,
    )

    save_image(result["restored_image"], args.output)
    print(f"\n[DONE] Restored thermal image saved to: {args.output}")
    print(f"Metrics: {result['metrics']}")


if __name__ == "__main__":
    main()
