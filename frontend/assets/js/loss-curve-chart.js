/**
 * Real-time Chart.js Loss Curve Visualizer
 */

export class LossCurveChart {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.chart = null;
    this._initChart();
  }

  _initChart() {
    if (!window.Chart) {
      console.warn("Chart.js not loaded.");
      return;
    }

    const ctx = this.canvas.getContext("2d");
    this.chart = new window.Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Masked MSE Loss",
            data: [],
            borderColor: "#00e5ff",
            backgroundColor: "rgba(0, 229, 255, 0.08)",
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            fill: true,
            tension: 0.1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#64748b", font: { size: 10 } }
          },
          y: {
            type: "logarithmic",
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#64748b", font: { size: 10 } }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#0d1322",
            borderColor: "rgba(255, 255, 255, 0.1)",
            borderWidth: 1,
            titleFont: { family: "JetBrains Mono" },
            bodyFont: { family: "JetBrains Mono" }
          }
        }
      }
    });
  }

  addDataPoint(step, loss) {
    if (!this.chart) return;
    this.chart.data.labels.push(step);
    this.chart.data.datasets[0].data.push(loss);
    this.chart.update("none");
  }

  setFullData(losses) {
    if (!this.chart) return;
    this.chart.data.labels = losses.map((_, i) => i + 1);
    this.chart.data.datasets[0].data = losses;
    this.chart.update();
  }

  reset() {
    if (!this.chart) return;
    this.chart.data.labels = [];
    this.chart.data.datasets[0].data = [];
    this.chart.update();
  }
}
