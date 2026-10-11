/**
 * Landing Page Controller
 */
import { ApiClient } from "./api-client.js";

document.addEventListener("DOMContentLoaded", async () => {
  const heroDot = document.getElementById("hero-status-dot");
  const heroStatusText = document.getElementById("hero-status-text");

  if (heroDot && heroStatusText) {
    try {
      const health = await ApiClient.checkHealth();
      if (health.status === "ok") {
        heroDot.className = "status-dot online";
        heroStatusText.textContent = `Backend Ready: ${health.gpu_name || health.device}`;
      } else {
        heroDot.className = "status-dot offline";
        heroStatusText.textContent = "Colab Server Offline";
      }
    } catch {
      heroDot.className = "status-dot offline";
      heroStatusText.textContent = "Colab Server Offline";
    }
  }
});
