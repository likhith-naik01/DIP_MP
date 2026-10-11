/**
 * UI Notification and Formatting Utilities
 */

export function showToast(message, type = "info") {
  let toastContainer = document.getElementById("toast-container");
  if (!toastContainer) {
    toastContainer = document.createElement("div");
    toastContainer.id = "toast-container";
    toastContainer.style.position = "fixed";
    toastContainer.style.bottom = "20px";
    toastContainer.style.right = "20px";
    toastContainer.style.zIndex = "9999";
    toastContainer.style.display = "flex";
    toastContainer.style.flexDirection = "column";
    toastContainer.style.gap = "10px";
    document.body.appendChild(toastContainer);
  }

  const toast = document.createElement("div");
  toast.style.padding = "12px 18px";
  toast.style.borderRadius = "8px";
  toast.style.fontFamily = "var(--font-sans)";
  toast.style.fontSize = "0.9rem";
  toast.style.fontWeight = "500";
  toast.style.backdropFilter = "blur(10px)";
  toast.style.boxShadow = "0 8px 24px rgba(0,0,0,0.5)";
  toast.style.transition = "all 0.3s ease";
  toast.style.opacity = "0";
  toast.style.transform = "translateY(10px)";

  if (type === "success") {
    toast.style.background = "rgba(0, 230, 118, 0.2)";
    toast.style.color = "#00e676";
    toast.style.border = "1px solid rgba(0, 230, 118, 0.4)";
  } else if (type === "error") {
    toast.style.background = "rgba(255, 23, 68, 0.2)";
    toast.style.color = "#ff5252";
    toast.style.border = "1px solid rgba(255, 23, 68, 0.4)";
  } else {
    toast.style.background = "rgba(0, 229, 255, 0.2)";
    toast.style.color = "#00e5ff";
    toast.style.border = "1px solid rgba(0, 229, 255, 0.4)";
  }

  toast.textContent = message;
  toastContainer.appendChild(toast);

  requestAnimationFrame(() => {
    toast.style.opacity = "1";
    toast.style.transform = "translateY(0)";
  });

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(10px)";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

export function formatSeconds(sec) {
  if (sec == null || isNaN(sec)) return "--";
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}m ${s}s`;
}

export function formatMetric(val, decimals = 2) {
  if (val == null || isNaN(val)) return "--";
  return Number(val).toFixed(decimals);
}
