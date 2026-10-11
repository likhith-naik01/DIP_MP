/**
 * Restoration Studio Page Controller
 */
import { ApiClient } from "./api-client.js";
import { getApiBaseUrl, setApiBaseUrl, resetApiBaseUrl } from "./app-config.js";
import { showToast, formatSeconds, formatMetric } from "./ui-helpers.js";
import { MaskDrawingEditor } from "./mask-drawing-editor.js";
import { BeforeAfterSlider } from "./before-after-slider.js";
import { LossCurveChart } from "./loss-curve-chart.js";
import { ImageUploadHandler } from "./image-upload-handler.js";

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const apiStatusPill = document.getElementById("api-status-pill");
  const apiStatusDot = document.getElementById("api-status-dot");
  const apiStatusText = document.getElementById("api-status-text");
  const apiSettingsBtn = document.getElementById("btn-api-settings");
  const apiModal = document.getElementById("api-modal");
  const apiUrlInput = document.getElementById("api-url-input");
  const btnSaveApiUrl = document.getElementById("btn-save-api-url");
  const btnCloseApiModal = document.getElementById("btn-close-api-modal");

  const dropzone = document.getElementById("image-dropzone");
  const fileInput = document.getElementById("image-file-input");
  const previewImg = document.getElementById("preview-source-img");
  const maskCanvas = document.getElementById("mask-canvas");
  const autoDetectDesc = document.getElementById("auto-detect-status-desc");
  const btnOpenColab = document.getElementById("btn-open-colab");

  const btnStartRestoration = document.getElementById("btn-start-restoration");
  const btnCancelRestoration = document.getElementById("btn-cancel-restoration");
  const btnDownloadRestored = document.getElementById("btn-download-restored");

  const progressBar = document.getElementById("progress-bar");
  const progressText = document.getElementById("progress-text");
  const iterCounter = document.getElementById("iter-counter");
  const etaText = document.getElementById("eta-text");
  const statPsnr = document.getElementById("stat-psnr");
  const statSsim = document.getElementById("stat-ssim");
  const statLoss = document.getElementById("stat-loss");
  const statTime = document.getElementById("stat-time");
  const statSharpness = document.getElementById("stat-sharpness");
  const statTenengrad = document.getElementById("stat-tenengrad");
  const taskSelect = document.getElementById("task-select");
  const sharpnessSelect = document.getElementById("sharpness-select");

  const sliderContainer = document.getElementById("before-after-slider");
  const sliderBeforeImg = document.getElementById("slider-before-img");
  const sliderAfterImg = document.getElementById("slider-after-img");
  const btnViewSbs = document.getElementById("btn-view-sbs");
  const btnViewTri = document.getElementById("btn-view-tri");
  const btnViewSlider = document.getElementById("btn-view-slider");
  const viewSideBySide = document.getElementById("view-sidebyside");
  const viewTri = document.getElementById("view-tri");
  const viewSlider = document.getElementById("view-slider");
  const sbsBeforeImg = document.getElementById("sbs-before-img");
  const sbsAfterImg = document.getElementById("sbs-after-img");
  const triOrigImg = document.getElementById("tri-orig-img");
  const triDegImg = document.getElementById("tri-deg-img");
  const triRestImg = document.getElementById("tri-rest-img");
  const colormapSelect = document.getElementById("colormap-select");
  const lossCanvas = document.getElementById("loss-chart-canvas");

  // State
  let currentFileBlob = null;
  let activeJobId = null;
  let pollInterval = null;

  // View Switcher Handlers
  if (btnViewSbs && btnViewSlider) {
    btnViewSbs.addEventListener("click", () => {
      btnViewSbs.classList.add("active");
      if (btnViewTri) btnViewTri.classList.remove("active");
      btnViewSlider.classList.remove("active");
      if (viewSideBySide) viewSideBySide.style.display = "grid";
      if (viewTri) viewTri.style.display = "none";
      if (viewSlider) viewSlider.style.display = "none";
    });

    if (btnViewTri) {
      btnViewTri.addEventListener("click", () => {
        btnViewTri.classList.add("active");
        btnViewSbs.classList.remove("active");
        btnViewSlider.classList.remove("active");
        if (viewSideBySide) viewSideBySide.style.display = "none";
        if (viewTri) viewTri.style.display = "grid";
        if (viewSlider) viewSlider.style.display = "none";
      });
    }

    btnViewSlider.addEventListener("click", () => {
      btnViewSlider.classList.add("active");
      btnViewSbs.classList.remove("active");
      if (btnViewTri) btnViewTri.classList.remove("active");
      if (viewSideBySide) viewSideBySide.style.display = "none";
      if (viewTri) viewTri.style.display = "none";
      if (viewSlider) viewSlider.style.display = "block";
      slider.setSliderPosition(50);
    });
  }

  // Helper to update all views simultaneously
  function updateDisplayedImages(originalUrl, degradedUrl, restoredUrl) {
    if (originalUrl) {
      if (triOrigImg) triOrigImg.src = originalUrl;
    }
    if (degradedUrl) {
      if (sliderBeforeImg) sliderBeforeImg.src = degradedUrl;
      if (sbsBeforeImg) sbsBeforeImg.src = degradedUrl;
      if (triDegImg) triDegImg.src = degradedUrl;
    }
    if (restoredUrl) {
      if (sliderAfterImg) sliderAfterImg.src = restoredUrl;
      if (sbsAfterImg) sbsAfterImg.src = restoredUrl;
      if (triRestImg) triRestImg.src = restoredUrl;
    }
  }

  // Initialize Subcomponents
  const maskEditor = new MaskDrawingEditor(maskCanvas, { brushSize: 16 });
  const slider = new BeforeAfterSlider(sliderContainer);
  const chart = new LossCurveChart(lossCanvas);

  // 1. Backend Connectivity Check with Silent Heartbeat and Resilience
  let consecutiveHealthFailures = 0;
  let isCheckingHealth = false;

  async function checkApiConnection(forceShowConnecting = false) {
    if (isCheckingHealth) return;
    if (activeJobId) return; // Don't interrupt active restoration polling

    isCheckingHealth = true;
    if (forceShowConnecting) {
      apiStatusDot.className = "status-dot busy";
      apiStatusText.textContent = "Connecting...";
      apiStatusDot.style.background = "#ffb300";
    }

    try {
      const health = await ApiClient.checkHealth();
      if (health.status === "ok") {
        consecutiveHealthFailures = 0;
        apiStatusDot.className = "status-dot online";
        if (health.cuda_available) {
          apiStatusText.textContent = `CUDA Active (${health.gpu_name})`;
          apiStatusDot.style.background = "var(--accent-cyan)";
          apiStatusDot.style.boxShadow = "0 0 10px var(--accent-cyan)";
        } else {
          apiStatusText.textContent = `CPU Mode (Connect Colab for CUDA GPU)`;
          apiStatusDot.style.background = "#ffb300";
          apiStatusDot.style.boxShadow = "0 0 8px #ffb300";
        }
      } else {
        consecutiveHealthFailures++;
        if (consecutiveHealthFailures >= 2) {
          apiStatusDot.className = "status-dot offline";
          apiStatusText.textContent = "Offline / Disconnected";
          apiStatusDot.style.background = "#ff1744";
          apiStatusDot.style.boxShadow = "none";
        }
      }
    } catch (e) {
      consecutiveHealthFailures++;
      if (consecutiveHealthFailures >= 2) {
        apiStatusDot.className = "status-dot offline";
        apiStatusText.textContent = "Offline / Disconnected";
        apiStatusDot.style.background = "#ff1744";
        apiStatusDot.style.boxShadow = "none";
      }
    } finally {
      isCheckingHealth = false;
    }
  }

  checkApiConnection(true);
  setInterval(() => checkApiConnection(false), 30000);

  // API Modal Settings
  apiSettingsBtn.addEventListener("click", () => {
    apiUrlInput.value = getApiBaseUrl();
    apiModal.classList.add("active");
  });

  btnCloseApiModal.addEventListener("click", () => {
    apiModal.classList.remove("active");
  });

  btnSaveApiUrl.addEventListener("click", () => {
    setApiBaseUrl(apiUrlInput.value);
    apiModal.classList.remove("active");
    showToast("API Endpoint Updated!", "success");
    checkApiConnection();
  });

  if (btnOpenColab) {
    btnOpenColab.addEventListener("click", () => {
      apiUrlInput.value = getApiBaseUrl();
      apiModal.classList.add("active");
    });
  }

  function updateAutoDetectStatus(imgElement) {
    if (!autoDetectDesc) return;
    try {
      const c = document.createElement("canvas");
      c.width = 128;
      c.height = 128;
      const ctx = c.getContext("2d");
      ctx.drawImage(imgElement, 0, 0, 128, 128);
      const data = ctx.getImageData(0, 0, 128, 128).data;
      let darkCount = 0;
      let satCount = 0;
      const total = 128 * 128;
      for (let i = 0; i < data.length; i += 4) {
        const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
        if (gray <= 3) darkCount++;
        else if (gray >= 252) satCount++;
      }
      const damagePct = ((darkCount + satCount) / total) * 100;
      
      // Compute high-frequency edge energy
      let edgeEnergy = 0;
      for (let y = 1; y < 127; y++) {
        for (let x = 1; x < 127; x++) {
          const idx = (y * 128 + x) * 4;
          const left = (data[idx - 4] + data[idx - 3] + data[idx - 2]) / 3;
          const right = (data[idx + 4] + data[idx + 5] + data[idx + 6]) / 3;
          const top = (data[idx - 128 * 4] + data[idx - 128 * 4 + 1] + data[idx - 128 * 4 + 2]) / 3;
          const bot = (data[idx + 128 * 4] + data[idx + 128 * 4 + 1] + data[idx + 128 * 4 + 2]) / 3;
          const gx = right - left;
          const gy = bot - top;
          edgeEnergy += (gx * gx + gy * gy);
        }
      }
      const avgEdge = edgeEnergy / (126 * 126);

      if (damagePct > 0.05) {
        autoDetectDesc.textContent = `Auto-Detected: Sensor damage found (~${Math.round(damagePct * 10) / 10}% corrupted pixels). Inpainting ready.`;
      } else if (avgEdge < 125.0) {
        autoDetectDesc.textContent = `Auto-Detected: Blurry sensor image (Low edge energy ${Math.round(avgEdge)}). Strong Deblur & Edge Sharpening pipeline activated.`;
        if (taskSelect && taskSelect.value === "auto") {
          taskSelect.style.borderColor = "var(--accent-cyan)";
        }
      } else {
        autoDetectDesc.textContent = `Auto-Detected: Clear thermal sensor image (Edge energy ${Math.round(avgEdge)}). Denoising & detail polish ready.`;
      }
    } catch (e) {
      autoDetectDesc.textContent = "Auto-Detection Active: Ready to eliminate blurriness and sensor noise.";
    }
  }

  // 2. Image Loading Handler
  new ImageUploadHandler({
    dropzone,
    fileInput,
    onImageLoaded: ({ img, file, dataUrl }) => {
      currentFileBlob = file;
      updateDisplayedImages(dataUrl, dataUrl);

      // Match canvas dimensions to image natural ratio
      maskEditor.resize(img.naturalWidth || 512, img.naturalHeight || 512);
      dropzone.querySelector(".dropzone-prompt").style.display = "none";
      btnStartRestoration.disabled = false;
      btnStartRestoration.textContent = "Optimize";
      updateAutoDetectStatus(img);
      showToast("Thermal image loaded successfully!", "info");
    }
  });

  // 3. Start Optimization Flow (Fully Automated)
  btnStartRestoration.addEventListener("click", async () => {
    if (!currentFileBlob) {
      showToast("Please upload a thermal image first.", "error");
      return;
    }

    try {
      btnStartRestoration.disabled = true;
      btnStartRestoration.textContent = "Optimizing...";
      btnCancelRestoration.style.display = "block";
      btnCancelRestoration.disabled = false;
      btnDownloadRestored.disabled = true;
      chart.reset();

      const chosenColormap = colormapSelect ? colormapSelect.value : "grayscale";
      const chosenTask = taskSelect ? taskSelect.value : "auto";
      const chosenSharpness = sharpnessSelect ? sharpnessSelect.value : "strong";
      const options = {
        task: chosenTask,
        model: "unet",
        iterations: 1500,
        learning_rate: 0.01,
        colormap: chosenColormap,
        sharpness_strength: chosenSharpness,
        simulate_damage: false,
        auto_detect_mask: true,
      };

      showToast("Submitting optimization job...", "info");
      const jobRes = await ApiClient.createRestorationJob(currentFileBlob, null, options);
      activeJobId = jobRes.job_id;
      showToast(`Optimization job ${activeJobId} started!`, "success");

      startPollingJob(activeJobId);
    } catch (err) {
      showToast(err.message, "error");
      btnStartRestoration.disabled = false;
      btnStartRestoration.textContent = "Optimize";
      btnCancelRestoration.disabled = true;
      btnCancelRestoration.style.display = "none";
    }
  });

  // 5. Polling Loop
  function startPollingJob(jobId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
      try {
        const statusData = await ApiClient.getJobStatus(jobId);
        
        // Update Progress UI
        const pct = statusData.progress_percentage || 0;
        progressBar.style.width = `${pct}%`;
        progressText.textContent = `${pct}%`;
        iterCounter.textContent = `${statusData.current_iteration || 0} / ${statusData.total_iterations || 0}`;
        etaText.textContent = formatSeconds(statusData.eta_seconds);
        
        if (statusData.current_loss != null) {
          statLoss.textContent = statusData.current_loss.toExponential(4);
          chart.addDataPoint(statusData.current_iteration, statusData.current_loss);
        }

        // Re-affirm online CUDA state during active polling
        if (apiStatusDot.className !== "status-dot online") {
          apiStatusDot.className = "status-dot online";
          apiStatusText.textContent = "CUDA Active (Tesla T4)";
          apiStatusDot.style.background = "var(--accent-cyan)";
          apiStatusDot.style.boxShadow = "0 0 10px var(--accent-cyan)";
        }

        // Live Degraded & Preview Sync
        if (statusData.degraded_image_base64) {
          if (sliderBeforeImg) sliderBeforeImg.src = statusData.degraded_image_base64;
          if (sbsBeforeImg) sbsBeforeImg.src = statusData.degraded_image_base64;
          if (triDegImg) triDegImg.src = statusData.degraded_image_base64;
        }
        if (statusData.preview_image_base64) {
          if (sliderAfterImg) sliderAfterImg.src = statusData.preview_image_base64;
          if (sbsAfterImg) sbsAfterImg.src = statusData.preview_image_base64;
          if (triRestImg) triRestImg.src = statusData.preview_image_base64;
        }

        // Check completion or terminal states
        if (statusData.status === "completed") {
          clearInterval(pollInterval);
          onJobCompleted(jobId);
        } else if (statusData.status === "failed") {
          clearInterval(pollInterval);
          showToast(`Job failed: ${statusData.error_message}`, "error");
          resetJobButtons();
        } else if (statusData.status === "cancelled") {
          clearInterval(pollInterval);
          showToast("Job was cancelled.", "info");
          resetJobButtons();
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    }, 1000);
  }

  async function onJobCompleted(jobId) {
    showToast("Restoration Complete! Loading final results...", "success");
    try {
      const resData = await ApiClient.getJobResult(jobId);
      if (resData.original_image_base64 && triOrigImg) {
        triOrigImg.src = resData.original_image_base64;
      }
      if (resData.degraded_image_base64) {
        if (sliderBeforeImg) sliderBeforeImg.src = resData.degraded_image_base64;
        if (sbsBeforeImg) sbsBeforeImg.src = resData.degraded_image_base64;
        if (triDegImg) triDegImg.src = resData.degraded_image_base64;
      }
      if (resData.restored_image_base64) {
        if (sliderAfterImg) sliderAfterImg.src = resData.restored_image_base64;
        if (sbsAfterImg) sbsAfterImg.src = resData.restored_image_base64;
        if (triRestImg) triRestImg.src = resData.restored_image_base64;
      }

      // Populate metrics
      const metrics = resData.metrics || {};
      const psnrVal = metrics.psnr != null ? metrics.psnr : metrics.psnr_proxy;
      statPsnr.textContent = psnrVal != null ? formatMetric(psnrVal, 2) + " dB" : "-- dB";
      statSsim.textContent = formatMetric(metrics.ssim, 4);
      statTime.textContent = formatSeconds(resData.runtime_seconds);

      if (statSharpness) {
        if (metrics.sharpness_gain_pct != null) {
          statSharpness.textContent = (metrics.sharpness_gain_pct >= 0 ? "+" : "") + formatMetric(metrics.sharpness_gain_pct, 1) + "%";
          statSharpness.style.color = metrics.sharpness_gain_pct >= 0 ? "var(--accent-cyan)" : "#ffb300";
        } else {
          statSharpness.textContent = "--";
        }
      }
      if (statTenengrad) {
        if (metrics.tenengrad_after != null) {
          statTenengrad.textContent = formatMetric(metrics.tenengrad_after, 0);
        } else {
          statTenengrad.textContent = "--";
        }
      }

      if (resData.losses && resData.losses.length > 0) {
        chart.setFullData(resData.losses);
      }

      btnDownloadRestored.disabled = false;
      btnDownloadRestored.onclick = () => {
        window.open(ApiClient.getDownloadUrl(jobId), "_blank");
      };
    } catch (err) {
      showToast("Error retrieving completed results.", "error");
    } finally {
      resetJobButtons();
    }
  }

  btnCancelRestoration.addEventListener("click", async () => {
    if (!activeJobId) return;
    try {
      await ApiClient.cancelJob(activeJobId);
      showToast("Cancelling optimization...", "info");
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  function resetJobButtons() {
    btnStartRestoration.disabled = false;
    btnStartRestoration.textContent = "Optimize";
    btnCancelRestoration.disabled = true;
    btnCancelRestoration.style.display = "none";
  }
});
