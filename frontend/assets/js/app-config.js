/**
 * Application Configuration and Backend URL Management
 */

const STORAGE_KEY = "dip_thermal_backend_url";
const DEFAULT_URL = "http://localhost:8000";

export function getApiBaseUrl() {
  const saved = localStorage.getItem(STORAGE_KEY);
  return saved ? saved.trim().replace(/\/+$/, "") : DEFAULT_URL;
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
