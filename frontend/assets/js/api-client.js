/**
 * REST API Client for Deep Image Prior Thermal Backend
 */
import { getApiBaseUrl } from "./app-config.js";

export class ApiClient {
  static getBaseUrl() {
    return getApiBaseUrl();
  }

  static async checkHealth() {
    const url = `${this.getBaseUrl()}/api/health`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 7000);
    try {
      const resp = await fetch(url, { method: "GET", signal: controller.signal });
      clearTimeout(timer);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return await resp.json();
    } catch (err) {
      clearTimeout(timer);
      return { status: "error", error: err.message };
    }
  }

  static async createRestorationJob(imageBlob, maskBlob = null, options = {}) {
    const url = `${this.getBaseUrl()}/api/restorations`;
    const formData = new FormData();
    formData.append("image", imageBlob, "input_thermal.png");
    
    if (maskBlob) {
      formData.append("mask", maskBlob, "mask.png");
    }
    
    if (options && Object.keys(options).length > 0) {
      formData.append("options", JSON.stringify(options));
    }

    const resp = await fetch(url, {
      method: "POST",
      body: formData
    });

    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned HTTP ${resp.status}`);
    }

    return await resp.json();
  }

  static async getJobStatus(jobId) {
    const url = `${this.getBaseUrl()}/api/restorations/${jobId}/status`;
    const resp = await fetch(url, { method: "GET" });
    if (!resp.ok) {
      throw new Error(`Failed to fetch status for job ${jobId}`);
    }
    return await resp.json();
  }

  static async getJobResult(jobId) {
    const url = `${this.getBaseUrl()}/api/restorations/${jobId}/result`;
    const resp = await fetch(url, { method: "GET" });
    if (!resp.ok) {
      throw new Error(`Failed to fetch results for job ${jobId}`);
    }
    return await resp.json();
  }

  static async cancelJob(jobId) {
    const url = `${this.getBaseUrl()}/api/restorations/${jobId}/cancel`;
    const resp = await fetch(url, { method: "POST" });
    if (!resp.ok) {
      throw new Error(`Failed to cancel job ${jobId}`);
    }
    return await resp.json();
  }

  static getDownloadUrl(jobId) {
    return `${this.getBaseUrl()}/api/restorations/${jobId}/download`;
  }
}
