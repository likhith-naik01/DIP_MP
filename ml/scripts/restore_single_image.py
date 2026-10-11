"""CLI entry point for restoring a single thermal image using Deep Image Prior."""

import argparse
import sys
from pathlib import Path

# Add ml folder to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dip_core.pipeline.restoration_pipeline import run_restoration
from dip_core.pipeline.image_postprocessing import save_image


def main():
    parser = argparse.ArgumentParser(description="Restore a single thermal image using Deep Image Prior.")
    parser.add_argument("--input", required=True, help="Path to input thermal image")
    parser.add_argument("--mask", default=None, help="Path to optional binary defect mask")
    parser.add_argument("--task", default="auto", choices=["auto", "deblur", "sharpen", "denoise", "inpaint", "denoise_and_inpaint"])
    parser.add_argument("--model", default="unet", choices=["unet", "conv_baseline"])
    parser.add_argument("--iterations", type=int, default=1500, help="Number of DIP iterations")
    parser.add_argument("--output-dir", default="output", help="Directory to save restoration artifacts")
    parser.add_argument("--simulate-damage", action="store_true", help="Simulate synthetic damage on clean input")
    parser.add_argument("--device", default=None, help="Target device (cuda or cpu)")

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("      DEEP IMAGE PRIOR: SINGLE THERMAL IMAGE RESTORATION          ")
    print("==================================================================")
    print(f"Input image: {args.input}")
    print(f"Task: {args.task} | Model: {args.model} | Iterations: {args.iterations}")

    def progress_callback(it, total, loss, preview_np):
        if it % 50 == 0 or it == 1 or it == total:
            print(f"[{it:>4}/{total}] Loss: {loss:.6f}")

    result = run_restoration(
        image=args.input,
        mask=args.mask,
        task=args.task,
        model_name=args.model,
        iterations=args.iterations,
        simulate_damage_flag=args.simulate_damage,
        progress_callback=progress_callback,
        device=args.device,
    )

    # Save output artifacts
    save_image(result.degraded_image, out_dir / "1_degraded.png", colormap="ironbow")
    save_image(result.mask, out_dir / "2_mask.png")
    save_image(result.restored_raw, out_dir / "3_restored_raw.png", colormap="ironbow")
    save_image(result.restored_composited, out_dir / "4_restored_composited.png", colormap="ironbow")

    print(f"\n[DONE] Restoration completed in {result.duration_seconds}s!")
    print(f"Artifacts saved to: {out_dir.resolve()}")
    print("Metrics:")
    for k, v in result.metrics.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
