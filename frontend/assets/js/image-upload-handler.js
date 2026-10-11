/**
 * Image Upload & Preset Handler
 */

export class ImageUploadHandler {
  constructor({ dropzone, fileInput, onImageLoaded }) {
    this.dropzone = dropzone;
    this.fileInput = fileInput;
    this.onImageLoaded = onImageLoaded;

    this._initEvents();
  }

  _initEvents() {
    if (this.dropzone && this.fileInput) {
      this.dropzone.addEventListener("click", () => this.fileInput.click());

      this.dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        this.dropzone.style.borderColor = "var(--accent-cyan)";
      });

      this.dropzone.addEventListener("dragleave", () => {
        this.dropzone.style.borderColor = "var(--border-card)";
      });

      this.dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        this.dropzone.style.borderColor = "var(--border-card)";
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
          this.loadFile(e.dataTransfer.files[0]);
        }
      });

      this.fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
          this.loadFile(e.target.files[0]);
        }
      });
    }
  }

  loadFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        if (this.onImageLoaded) {
          this.onImageLoaded({ img, file, dataUrl: e.target.result });
        }
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  async loadFromUrl(url) {
    const resp = await fetch(url);
    const blob = await resp.blob();
    const file = new File([blob], "sample_thermal.png", { type: "image/png" });
    this.loadFile(file);
  }
}
