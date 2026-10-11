# Sample Thermal Images for Benchmarking and Evaluation

This directory contains representative 8-bit single-channel thermal infrared scenes used for evaluation and benchmarking:

1. **`thermal_circuit_board.png`** (256x256):
   - **Scene**: Electronic printed circuit board with active IC microprocessor, power regulator heat plume, and copper thermal dissipation traces.
   - **Characteristics**: Sharp edges, delicate high-frequency traces, and high dynamic range temperature differentials. Ideal for evaluating edge preservation and circuit inspection.

2. **`thermal_building_heat_leak.png`** (256x256):
   - **Scene**: Building facade showing insulation failure and window thermal emission.
   - **Characteristics**: Smooth temperature gradients across architectural surfaces combined with structural window boundaries. Evaluates structural inpainting over large rectangular sensor dropouts.

3. **`thermal_machinery_bearing.png`** (256x256):
   - **Scene**: Industrial rotating machinery bearing experiencing localized friction overheating.
   - **Characteristics**: Radial heat diffusion patterns and concentric temperature contours. Evaluates isotropic smoothing vs. directional thermal gradient restoration.
