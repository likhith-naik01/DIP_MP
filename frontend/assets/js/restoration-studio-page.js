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
  const brushSizeInput = document.getElementById("brush-size");
  const btnClearMask = document.getElementById("btn-clear-mask");
  const btnToolBrush = document.getElementById("btn-tool-brush");
  const btnToolEraser = document.getElementById("btn-tool-eraser");

  const formTask = document.getElementById("opt-task");
  const formModel = document.getElementById("opt-model");
  const formIterations = document.getElementById("opt-iterations");
  const formLr = document.getElementById("opt-lr");
  const formColormap = document.getElementById("opt-colormap");
  const chkSimDamage = document.getElementById("chk-simulate-damage");
  const chkAutoMask = document.getElementById("chk-auto-mask");

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

  const sliderContainer = document.getElementById("before-after-slider");
  const sliderBeforeImg = document.getElementById("slider-before-img");
  const sliderAfterImg = document.getElementById("slider-after-img");
  const lossCanvas = document.getElementById("loss-chart-canvas");

  // State
  let currentFileBlob = null;
  let activeJobId = null;
  let pollInterval = null;

  // Initialize Subcomponents
  const maskEditor = new MaskDrawingEditor(maskCanvas, { brushSize: 16 });
  const slider = new BeforeAfterSlider(sliderContainer);
  const chart = new LossCurveChart(lossCanvas);

  // 1. Backend Connectivity Check
  async function checkApiConnection() {
    apiStatusDot.className = "status-dot busy";
    apiStatusText.textContent = "Connecting...";
    const health = await ApiClient.checkHealth();
    if (health.status === "ok") {
      apiStatusDot.className = "status-dot online";
      apiStatusText.textContent = `Online (${health.gpu_name || health.device})`;
    } else {
      apiStatusDot.className = "status-dot offline";
      apiStatusText.textContent = "Offline / Disconnected";
    }
  }

  checkApiConnection();
  setInterval(checkApiConnection, 15000);

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

  // 2. Image Loading Handler
  new ImageUploadHandler({
    dropzone,
    fileInput,
    onImageLoaded: ({ img, file, dataUrl }) => {
      currentFileBlob = file;
      previewImg.src = dataUrl;
      previewImg.style.display = "block";
      sliderBeforeImg.src = dataUrl;
      sliderAfterImg.src = dataUrl;

      // Match canvas dimensions to image natural ratio
      maskEditor.resize(img.naturalWidth || 512, img.naturalHeight || 512);
      dropzone.querySelector(".dropzone-prompt").style.display = "none";
      btnStartRestoration.disabled = false;
      showToast("Thermal image loaded successfully!", "info");
    }
  });

  // Sample Images Buttons
  document.querySelectorAll(".btn-sample-image").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const sampleUrl = btn.dataset.url;
      try {
        const resp = await fetch(sampleUrl);
        const blob = await resp.blob();
        currentFileBlob = blob;
        const dataUrl = URL.createObjectURL(blob);
        previewImg.src = dataUrl;
        previewImg.style.display = "block";
        sliderBeforeImg.src = dataUrl;
        sliderAfterImg.src = dataUrl;
        maskEditor.resize(512, 512);
        dropzone.querySelector(".dropzone-prompt").style.display = "none";
        btnStartRestoration.disabled = false;
        showToast(`Loaded sample: ${btn.textContent}`, "info");
      } catch (err) {
        showToast("Failed to load sample image.", "error");
      }
    });
  });

  // 3. Mask Tools
  brushSizeInput.addEventListener("input", (e) => {
    maskEditor.setBrushSize(e.target.value);
  });

  btnToolBrush.addEventListener("click", () => {
    maskEditor.setTool("brush");
    btnToolBrush.classList.add("btn-primary");
    btnToolBrush.classList.remove("btn-secondary");
    btnToolEraser.classList.remove("btn-primary");
    btnToolEraser.classList.add("btn-secondary");
  });

  btnToolEraser.addEventListener("click", () => {
    maskEditor.setTool("eraser");
    btnToolEraser.classList.add("btn-primary");
    btnToolEraser.classList.remove("btn-secondary");
    btnToolBrush.classList.remove("btn-primary");
    btnToolBrush.classList.add("btn-secondary");
  });

  btnClearMask.addEventListener("click", () => {
    maskEditor.clear();
    showToast("Mask cleared", "info");
  });

  // 4. Start Restoration Flow
  btnStartRestoration.addEventListener("click", async () => {
    if (!currentFileBlob) {
      showToast("Please upload a thermal image first.", "error");
      return;
    }

    try {
      btnStartRestoration.disabled = true;
      btnCancelRestoration.disabled = false;
      btnDownloadRestored.disabled = true;
      chart.reset();

      const maskBlob = await maskEditor.getMaskBlob();

      const options = {
        task: formTask.value,
        model: formModel.value,
        iterations: Number(formIterations.value),
        learning_rate: Number(formLr.value),
        colormap: formColormap.value,
        simulate_damage: chkSimDamage.checked,
        auto_detect_mask: chkAutoMask.checked,
      };

      showToast("Submitting restoration job to GPU...", "info");
      const jobRes = await ApiClient.createRestorationJob(currentFileBlob, maskBlob, options);
      activeJobId = jobRes.job_id;
      showToast(`Job ${activeJobId} started!`, "success");

      startPollingJob(activeJobId);
    } catch (err) {
      showToast(err.message, "error");
      btnStartRestoration.disabled = false;
      btnCancelRestoration.disabled = true;
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

        // Live Preview Thumbnail
        if (statusData.preview_image_base64) {
          sliderAfterImg.src = statusData.preview_image_base64;
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
      if (resData.degraded_image_base64) {
        sliderBeforeImg.src = resData.degraded_image_base64;
      }
      if (resData.restored_image_base64) {
        sliderAfterImg.src = resData.restored_image_base64;
      }

      // Populate metrics
      const metrics = resData.metrics || {};
      statPsnr.textContent = formatMetric(metrics.psnr, 2) + " dB";
      statSsim.textContent = formatMetric(metrics.ssim, 4);
      statTime.textContent = formatSeconds(resData.runtime_seconds);

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
    btnCancelRestoration.disabled = true;
  }
});
