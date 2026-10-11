/**
 * Interactive Draggable Before / After Split Slider
 */

export class BeforeAfterSlider {
  constructor(containerElement) {
    this.container = containerElement;
    this.overlay = containerElement.querySelector(".slider-overlay");
    this.handle = containerElement.querySelector(".slider-handle");
    this.isDragging = false;

    this._initEvents();
    this.setSliderPosition(50);
  }

  _initEvents() {
    const startDrag = (e) => {
      this.isDragging = true;
      this._updatePositionFromEvent(e);
    };

    const doDrag = (e) => {
      if (!this.isDragging) return;
      this._updatePositionFromEvent(e);
    };

    const stopDrag = () => {
      this.isDragging = false;
    };

    this.container.addEventListener("mousedown", startDrag);
    window.addEventListener("mousemove", doDrag);
    window.addEventListener("mouseup", stopDrag);

    this.container.addEventListener("touchstart", (e) => {
      this.isDragging = true;
      this._updatePositionFromEvent(e.touches[0]);
    }, { passive: true });

    window.addEventListener("touchmove", (e) => {
      if (!this.isDragging) return;
      this._updatePositionFromEvent(e.touches[0]);
    }, { passive: true });

    window.addEventListener("touchend", stopDrag);
  }

  _updatePositionFromEvent(e) {
    const rect = this.container.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const percent = (x / rect.width) * 100;
    this.setSliderPosition(percent);
  }

  setSliderPosition(percent) {
    const clamped = Math.max(0, Math.min(percent, 100));
    if (this.overlay) {
      this.overlay.style.width = `${clamped}%`;
    }
    if (this.handle) {
      this.handle.style.left = `${clamped}%`;
    }
  }
}
