/**
 * Interactive Mask Drawing Editor for Canvas
 */

export class MaskDrawingEditor {
  constructor(canvasElement, options = {}) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext("2d");
    this.brushSize = options.brushSize || 16;
    this.tool = options.tool || "brush"; // "brush" or "eraser"
    this.isDrawing = false;
    this.hasCustomMask = false;

    this._initEvents();
    this.clear();
  }

  _initEvents() {
    this.canvas.addEventListener("mousedown", (e) => this._startDrawing(e));
    this.canvas.addEventListener("mousemove", (e) => this._draw(e));
    window.addEventListener("mouseup", () => this._stopDrawing());

    // Touch events for tablet/mobile
    this.canvas.addEventListener("touchstart", (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      const mouseEvent = new MouseEvent("mousedown", {
        clientX: touch.clientX,
        clientY: touch.clientY
      });
      this.canvas.dispatchEvent(mouseEvent);
    }, { passive: false });

    this.canvas.addEventListener("touchmove", (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      const mouseEvent = new MouseEvent("mousemove", {
        clientX: touch.clientX,
        clientY: touch.clientY
      });
      this.canvas.dispatchEvent(mouseEvent);
    }, { passive: false });

    window.addEventListener("touchend", () => this._stopDrawing());
  }

  _getCoordinates(e) {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  }

  _startDrawing(e) {
    this.isDrawing = true;
    this.hasCustomMask = true;
    const { x, y } = this._getCoordinates(e);
    this.ctx.beginPath();
    this.ctx.moveTo(x, y);
    this._draw(e);
  }

  _draw(e) {
    if (!this.isDrawing) return;
    const { x, y } = this._getCoordinates(e);

    this.ctx.lineWidth = this.brushSize;
    this.ctx.lineCap = "round";
    this.ctx.lineJoin = "round";

    if (this.tool === "eraser") {
      this.ctx.globalCompositeOperation = "destination-out";
    } else {
      // Draw red highlight for UI visualization of damage
      this.ctx.globalCompositeOperation = "source-over";
      this.ctx.strokeStyle = "rgba(255, 23, 68, 0.85)";
    }

    this.ctx.lineTo(x, y);
    this.ctx.stroke();
  }

  _stopDrawing() {
    if (this.isDrawing) {
      this.isDrawing = false;
      this.ctx.closePath();
    }
  }

  setBrushSize(size) {
    this.brushSize = Number(size);
  }

  setTool(toolName) {
    this.tool = toolName;
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.hasCustomMask = false;
  }

  resize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
    this.clear();
  }

  /**
   * Generates a binary mask blob where drawn damaged regions are 0 (black)
   * and undamaged regions are 255 (white).
   */
  async getMaskBlob() {
    if (!this.hasCustomMask) return null;

    const offscreen = document.createElement("canvas");
    offscreen.width = this.canvas.width;
    offscreen.height = this.canvas.height;
    const offCtx = offscreen.getContext("2d");

    // 1. Fill entire background with white (1.0 = valid)
    offCtx.fillStyle = "#ffffff";
    offCtx.fillRect(0, 0, offscreen.width, offscreen.height);

    // 2. Read drawn damage mask from drawing canvas
    const drawData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height).data;
    const offData = offCtx.getImageData(0, 0, offscreen.width, offscreen.height);

    for (let i = 0; i < drawData.length; i += 4) {
      const alpha = drawData[i + 3];
      if (alpha > 30) {
        // Pixel is drawn as damaged -> mark as black (0.0 in DIP valid mask)
        offData.data[i] = 0;
        offData.data[i + 1] = 0;
        offData.data[i + 2] = 0;
      }
    }

    offCtx.putImageData(offData, 0, 0);

    return new Promise((resolve) => {
      offscreen.toBlob((blob) => resolve(blob), "image/png");
    });
  }
}
