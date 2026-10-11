import argparse
import sys
import os
import csv
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add ml folder to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dip_core.config import DIPConfig
from dip_core.pipeline.preprocess import load_image, simulate_damage
from dip_core.pipeline.trainer import train_dip
from dip_core.baselines import apply_mean_filter, apply_median_filter
from dip_core.metrics import calculate_psnr, calculate_ssim
from dip_core.pipeline.postprocess import save_image, apply_thermal_colormap


def run_benchmark(
    samples_dir: str,
    output_dir: str = "benchmark_output",
    iterations: int = 1000,
    noise_std: float = 0.1,
    device: str = "cuda",
):
    """
    Benchmark DIP generators vs Classical Filters on thermal images.
    Restores with: U-Net DIP, Conv DIP, Mean 3x3, Mean 7x7, Median.
    Writes results.csv, saves side-by-side comparison PNGs, and plots PSNR vs iteration.
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_files = list(Path(samples_dir).glob("*.png")) + list(Path(samples_dir).glob("*.jpg"))

    if not sample_files:
        raise FileNotFoundError(f"No PNG or JPG images found in {samples_dir}")

    print("=================================================================")
    print("      DEEP IMAGE PRIOR THERMAL BENCHMARK EVALUATION              ")
    print("=================================================================")
    print(f"Sample folder: {samples_dir} ({len(sample_files)} images)")
    print(f"Iterations: {iterations} | Noise std: {noise_std} | Device: {device}")

    all_records = []
    convergence_data = {}

    for idx, img_path in enumerate(sample_files):
        img_name = img_path.stem
        print(f"\n[{idx+1}/{len(sample_files)}] Benchmarking: {img_name}...")

        clean = load_image(str(img_path), grayscale=True)

        # Degrade: Gaussian noise + missing rectangular holes
        degraded, mask = simulate_damage(
            clean,
            noise_std=noise_std,
            pixel_dropout_rate=0.03,
            add_holes=True,
            hole_type="rect",
        )

        # 1. Classical Baselines
        # Mean 3x3
        t0 = time.time()
        m3_res = apply_mean_filter(degraded, ksize=3)
        dt_m3 = time.time() - t0
        p_m3 = calculate_psnr(m3_res, clean)
        s_m3 = calculate_ssim(m3_res, clean)
        all_records.append({"image": img_name, "method": "Mean 3x3", "psnr": round(p_m3, 2), "ssim": round(s_m3, 4), "runtime_sec": round(dt_m3, 3)})

        # Mean 7x7
        t0 = time.time()
        m7_res = apply_mean_filter(degraded, ksize=7)
        dt_m7 = time.time() - t0
        p_m7 = calculate_psnr(m7_res, clean)
        s_m7 = calculate_ssim(m7_res, clean)
        all_records.append({"image": img_name, "method": "Mean 7x7", "psnr": round(p_m7, 2), "ssim": round(s_m7, 4), "runtime_sec": round(dt_m7, 3)})

        # Median
        t0 = time.time()
        med_res = apply_median_filter(degraded, ksize=5)
        dt_med = time.time() - t0
        p_med = calculate_psnr(med_res, clean)
        s_med = calculate_ssim(med_res, clean)
        all_records.append({"image": img_name, "method": "Median", "psnr": round(p_med, 2), "ssim": round(s_med, 4), "runtime_sec": round(dt_med, 3)})

        # 2. Conv-DIP Baseline
        print("  Training Conv-DIP (Baseline)...")
        conv_cfg = DIPConfig(
            iterations=iterations,
            model_type="conv_baseline",
            learning_rate=0.01,
            early_stopping=False,
            device=device,
            run_baselines=False,
        )
        t0 = time.time()
        conv_out = train_dip(
            image_input=degraded,
            mask_input=mask,
            reference_input=clean,
            config=conv_cfg,
            task="inpaint",
        )
        dt_conv = time.time() - t0
        conv_res = conv_out["restored_image"]
        p_conv = conv_out["metrics"]["psnr"]
        s_conv = conv_out["metrics"]["ssim"]
        all_records.append({"image": img_name, "method": "Conv DIP", "psnr": p_conv, "ssim": s_conv, "runtime_sec": round(dt_conv, 2)})

        # 3. U-Net DIP (Ours)
        print("  Training U-Net DIP (Main Contribution)...")
        unet_cfg = DIPConfig(
            iterations=iterations,
            model_type="unet",
            learning_rate=0.01,
            early_stopping=False,
            device=device,
            run_baselines=False,
        )
        t0 = time.time()
        unet_out = train_dip(
            image_input=degraded,
            mask_input=mask,
            reference_input=clean,
            config=unet_cfg,
            task="inpaint",
        )
        dt_unet = time.time() - t0
        unet_res = unet_out["restored_image"]
        p_unet = unet_out["metrics"]["psnr"]
        s_unet = unet_out["metrics"]["ssim"]
        all_records.append({"image": img_name, "method": "U-Net DIP", "psnr": p_unet, "ssim": s_unet, "runtime_sec": round(dt_unet, 2)})

        # Save convergence metrics for plotting
        if idx == 0:
            convergence_data["conv_psnr"] = [m["psnr"] for m in conv_out["metric_history"] if "psnr" in m]
            convergence_data["unet_psnr"] = [m["psnr"] for m in unet_out["metric_history"] if "psnr" in m]
            convergence_data["iters"] = [m["iteration"] for m in unet_out["metric_history"] if "psnr" in m]

        # 4. Save Side-by-Side Comparison PNG
        fig, axes = plt.subplots(1, 7, figsize=(28, 4))
        titles = [
            f"1. Clean Ground Truth",
            f"2. Degraded ({calculate_psnr(degraded, clean):.2f} dB)",
            f"3. Mean 3x3 ({p_m3:.2f} dB)",
            f"4. Mean 7x7 ({p_m7:.2f} dB)",
            f"5. Median ({p_med:.2f} dB)",
            f"6. Conv DIP ({p_conv:.2f} dB)",
            f"7. U-Net DIP ({p_unet:.2f} dB)",
        ]
        images = [clean, degraded, m3_res, m7_res, med_res, conv_res, unet_res]

        for ax, im, t in zip(axes, images, titles):
            # Display with Ironbow thermal colormap
            rgb = apply_thermal_colormap(im, "ironbow")
            ax.imshow(rgb)
            ax.set_title(t, fontsize=11, fontweight="bold")
            ax.axis("off")

        plt.tight_layout()
        comp_path = os.path.join(output_dir, f"{img_name}_comparison.png")
        plt.savefig(comp_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved comparison image: {comp_path}")

    # 5. Write results.csv
    csv_file = os.path.join(output_dir, "results.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "method", "psnr", "ssim", "runtime_sec"])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n[DONE] CSV saved to: {csv_file}")

    # 6. Plot PSNR vs Iteration for U-Net vs Conv
    if "iters" in convergence_data and convergence_data["iters"]:
        plt.figure(figsize=(9, 5))
        plt.plot(convergence_data["iters"], convergence_data["unet_psnr"], label="U-Net DIP (Ours)", color="#f97316", linewidth=2.5)
        plt.plot(convergence_data["iters"], convergence_data["conv_psnr"], label="Conv DIP (Baseline)", color="#3b82f6", linewidth=2.0, linestyle="--")
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("PSNR (dB)", fontsize=12)
        plt.title("Restoration Quality Progression: U-Net DIP vs Conv DIP", fontsize=14, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=11)
        plot_path = os.path.join(output_dir, "psnr_convergence_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[DONE] Convergence plot saved to: {plot_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Image':<22} | {'Method':<14} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Time (s)':<8}")
    print("-" * 70)
    for r in all_records:
        print(f"{r['image']:<22} | {r['method']:<14} | {r['psnr']:<10} | {r['ssim']:<8} | {r['runtime_sec']:<8}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run Thermal DIP benchmark across sample images")
    parser.add_argument("--samples", default="storage/sample_images", help="Folder of sample images")
    parser.add_argument("--output", default="benchmark_output", help="Directory for results")
    parser.add_argument("--iters", type=int, default=1000, help="DIP iterations")
    parser.add_argument("--noise-std", type=float, default=0.1, help="Noise standard deviation")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")

    args = parser.parse_args()
    run_benchmark(
        samples_dir=args.samples,
        output_dir=args.output,
        iterations=args.iters,
        noise_std=args.noise_std,
        device=args.device,
    )


if __name__ == "__main__":
    main()
