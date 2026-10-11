/**
 * Application Configuration and Backend URL Management
 */

const STORAGE_KEY = "dip_thermal_backend_url";
const DEFAULT_URL = "http://localhost:8000";

export function getApiBaseUrl() {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved && saved.trim()) {
    return saved.trim().replace(/\/+$/, "");
  }
  // When deployed and served directly from backend, use current origin
  if (typeof window !== "undefined" && window.location && window.location.origin && window.location.protocol.startsWith("http")) {
    return window.location.origin;
  }
  return DEFAULT_URL;
}

export function setApiBaseUrl(url) {
  if (!url) return;
  const cleanUrl = url.trim().replace(/\/+$/, "");
  localStorage.setItem(STORAGE_KEY, cleanUrl);
}

export function resetApiBaseUrl() {
  localStorage.removeItem(STORAGE_KEY);
  return DEFAULT_URL;
}
