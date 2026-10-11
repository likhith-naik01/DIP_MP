"""Benchmarking script: evaluates U-Net DIP vs Conv baseline vs classical filters."""

import argparse
import sys
import csv
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add ml folder to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dip_core.config import DamageConfig
from dip_core.pipeline.image_preprocessing import load_image
from dip_core.pipeline.damage_simulation import simulate_damage
from dip_core.pipeline.restoration_pipeline import run_restoration
from dip_core.pipeline.image_postprocessing import apply_thermal_colormap
from dip_core.baselines import apply_mean_filter, apply_median_filter, apply_inpaint_baseline
from dip_core.metrics import calculate_psnr, calculate_ssim


def run_benchmark(
    samples_dir: str = "sample_images",
    output_dir: str = "benchmark_output",
    iterations: int = 1000,
    device: str = "cuda",
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sample_path = Path(samples_dir)

    images = list(sample_path.glob("*.png")) + list(sample_path.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No PNG or JPG images found in {samples_dir}")

    print("==================================================================")
    print("      DEEP IMAGE PRIOR: COMPREHENSIVE BENCHMARK EVALUATION        ")
    print("==================================================================")
    print(f"Sample folder: {samples_dir} ({len(images)} images found)")
    print(f"Iterations: {iterations} | Target Device: {device}")

    all_records = []
    conv_progression = {}

    for idx, img_file in enumerate(images):
        img_name = img_file.stem
        print(f"\n[{idx+1}/{len(images)}] Benchmarking: {img_name}...")

        clean = load_image(str(img_file), grayscale=True)
        dmg_cfg = DamageConfig(gaussian_snr_db=20.0, dropout_ratio=0.03, num_line_holes=3, num_rect_holes=2, seed=42)
        degraded, mask, _ = simulate_damage(clean, dmg_cfg)

        # 1. Classical Baselines
        # Mean 3x3
        t0 = time.time()
        m3 = apply_mean_filter(degraded, 3)
        dt_m3 = time.time() - t0
        all_records.append({"image": img_name, "method": "Mean 3x3", "psnr": round(calculate_psnr(m3, clean), 2), "ssim": round(calculate_ssim(m3, clean), 4), "seconds": round(dt_m3, 3)})

        # Mean 7x7
        t0 = time.time()
        m7 = apply_mean_filter(degraded, 7)
        dt_m7 = time.time() - t0
        all_records.append({"image": img_name, "method": "Mean 7x7", "psnr": round(calculate_psnr(m7, clean), 2), "ssim": round(calculate_ssim(m7, clean), 4), "seconds": round(dt_m7, 3)})

        # Median 3x3
        t0 = time.time()
        med = apply_median_filter(degraded, 3)
        dt_med = time.time() - t0
        all_records.append({"image": img_name, "method": "Median 3x3", "psnr": round(calculate_psnr(med, clean), 2), "ssim": round(calculate_ssim(med, clean), 4), "seconds": round(dt_med, 3)})

        # OpenCV Inpaint Telea
        t0 = time.time()
        telea = apply_inpaint_baseline(degraded, mask, method="telea")
        dt_telea = time.time() - t0
        all_records.append({"image": img_name, "method": "OpenCV Telea", "psnr": round(calculate_psnr(telea, clean), 2), "ssim": round(calculate_ssim(telea, clean), 4), "seconds": round(dt_telea, 3)})

        # 2. Conv-Baseline DIP
        print("  Running Conv-Baseline DIP...")
        conv_res = run_restoration(
            image=degraded,
            mask=mask,
            task="inpaint",
            model_name="conv_baseline",
            iterations=iterations,
            reference_image=clean,
            device=device,
        )
        all_records.append({"image": img_name, "method": "Conv-DIP (Baseline)", "psnr": conv_res.metrics["psnr"], "ssim": conv_res.metrics["ssim"], "seconds": conv_res.duration_seconds})

        # 3. U-Net DIP (Ours)
        print("  Running U-Net DIP (Ours)...")
        unet_res = run_restoration(
            image=degraded,
            mask=mask,
            task="inpaint",
            model_name="unet",
            iterations=iterations,
            reference_image=clean,
            device=device,
        )
        all_records.append({"image": img_name, "method": "U-Net DIP (Ours)", "psnr": unet_res.metrics["psnr"], "ssim": unet_res.metrics["ssim"], "seconds": unet_res.duration_seconds})

        # Track progression for first image
        if idx == 0:
            conv_progression["unet"] = unet_res.psnr_history
            conv_progression["conv"] = conv_res.psnr_history
            conv_progression["steps"] = [i * 50 for i in range(1, len(unet_res.psnr_history) + 1)]

        # 4. Save side-by-side comparison PNG
        fig, axes = plt.subplots(1, 7, figsize=(24, 4))
        titles = [
            "Clean Reference",
            f"Degraded ({calculate_psnr(degraded, clean):.1f} dB)",
            f"Mean 3x3 ({calculate_psnr(m3, clean):.1f} dB)",
            f"Median ({calculate_psnr(med, clean):.1f} dB)",
            f"OpenCV ({calculate_psnr(telea, clean):.1f} dB)",
            f"Conv DIP ({conv_res.metrics['psnr']:.1f} dB)",
            f"U-Net DIP ({unet_res.metrics['psnr']:.1f} dB)",
        ]
        imgs = [clean, degraded, m3, med, telea, conv_res.restored_composited, unet_res.restored_composited]

        for ax, im, t in zip(axes, imgs, titles):
            ax.imshow(apply_thermal_colormap(im, "ironbow"))
            ax.set_title(t, fontsize=10, fontweight="bold")
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(out_path / f"{img_name}_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 5. Write benchmark_results.csv
    csv_file = out_path / "benchmark_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "method", "psnr", "ssim", "seconds"])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n[DONE] CSV saved to: {csv_file}")

    # 6. Generate Summary Table (Mean per method)
    methods = ["Mean 3x3", "Mean 7x7", "Median 3x3", "OpenCV Telea", "Conv-DIP (Baseline)", "U-Net DIP (Ours)"]
    summary = []
    for m in methods:
        m_recs = [r for r in all_records if r["method"] == m]
        if m_recs:
            avg_p = np.mean([r["psnr"] for r in m_recs])
            avg_s = np.mean([r["ssim"] for r in m_recs])
            avg_t = np.mean([r["seconds"] for r in m_recs])
            summary.append({"method": m, "avg_psnr": round(float(avg_p), 2), "avg_ssim": round(float(avg_s), 4), "avg_seconds": round(float(avg_t), 2)})

    summary_file = out_path / "benchmark_summary.csv"
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "avg_psnr", "avg_ssim", "avg_seconds"])
        writer.writeheader()
        writer.writerows(summary)

    # 7. Plots: PSNR vs Iteration & Bar Chart
    if conv_progression.get("steps"):
        plt.figure(figsize=(8, 4.5))
        plt.plot(conv_progression["steps"], conv_progression["unet"], label="U-Net DIP (Ours)", color="#f97316", linewidth=2.2)
        plt.plot(conv_progression["steps"], conv_progression["conv"], label="Conv DIP (Baseline)", color="#3b82f6", linewidth=1.8, linestyle="--")
        plt.xlabel("Iteration", fontsize=11)
        plt.ylabel("PSNR (dB)", fontsize=11)
        plt.title("Restoration Fidelity: U-Net vs Conv-Baseline", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(out_path / "psnr_convergence_comparison.png", dpi=150)
        plt.close()

    # Bar chart of mean PSNR
    plt.figure(figsize=(9, 4.5))
    bar_methods = [s["method"] for s in summary]
    bar_psnrs = [s["avg_psnr"] for s in summary]
    colors = ["#9ca3af", "#9ca3af", "#9ca3af", "#60a5fa", "#3b82f6", "#f97316"]
    plt.bar(bar_methods, bar_psnrs, color=colors)
    plt.ylabel("Average PSNR (dB)", fontsize=11)
    plt.title("Overall Restoration Quality by Method", fontsize=13, fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path / "method_comparison_barchart.png", dpi=150)
    plt.close()

    print("\n==================================================================")
    print("                     BENCHMARK SUMMARY                            ")
    print("==================================================================")
    print(f"{'Method':<24} | {'Avg PSNR (dB)':<14} | {'Avg SSIM':<10} | {'Avg Time (s)':<12}")
    print("-" * 68)
    for s in summary:
        print(f"{s['method']:<24} | {s['avg_psnr']:<14} | {s['avg_ssim']:<10} | {s['avg_seconds']:<12}")
    print("==================================================================")


def main():
    parser = argparse.ArgumentParser(description="Run benchmark comparing U-Net DIP vs Conv and filters.")
    parser.add_argument("--samples", default="sample_images", help="Folder containing sample images")
    parser.add_argument("--output", default="benchmark_output", help="Output directory")
    parser.add_argument("--iters", type=int, default=1000, help="DIP iterations")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")

    args = parser.parse_args()
    run_benchmark(
        samples_dir=args.samples,
        output_dir=args.output,
        iterations=args.iters,
        device=args.device,
    )


if __name__ == "__main__":
    main()
