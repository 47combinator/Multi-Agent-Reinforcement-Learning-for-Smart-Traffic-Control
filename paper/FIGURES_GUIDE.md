# Paper Figures Guide
# =====================
# This document maps each figure reference in the paper to its source file.
# Copy or symlink these into paper/figures/ before compiling the LaTeX.

## Figure Mapping

| Paper Reference | Source File | Description |
|:---|:---|:---|
| Fig. 1 | `Generate from reward_calculator.py` | Quadratic reward penalty curve (queue vs reward) |
| Fig. 2 | `benchmark/plots/` or generate | SUMO-TraCI-RL architecture diagram |
| Fig. 3 | `benchmark/plots/training_curves_comparison.png` | Training reward convergence |
| Fig. 4 | `benchmark/plots/algorithm_comparison.png` | Bar chart: wait time + queue comparison |
| Fig. 5 | `Generate from benchmark_results.json` | Per-episode waiting time box/scatter plot |
| Fig. 6 | `SUMO GUI screenshot` | Intersection snapshot at peak congestion |
| Fig. 7 | `Generate from benchmark data` | Phase selection frequency bar chart |

## How to Generate Missing Figures

### Fig. 1 — Reward Penalty Curve
```python
import numpy as np
import matplotlib.pyplot as plt

queue = np.arange(0, 40, 0.5)
threshold = 10
linear = -0.3 * queue
quadratic = np.where(queue > threshold, linear - 0.05 * (queue - threshold)**2, linear)

plt.figure(figsize=(8, 5))
plt.plot(queue, linear, '--', label='Linear penalty', alpha=0.7)
plt.plot(queue, quadratic, '-', label='Quadratic penalty (ours)', linewidth=2)
plt.axvline(x=threshold, color='red', linestyle=':', label=f'Threshold = {threshold}')
plt.xlabel('Total Queue Length')
plt.ylabel('Reward Component')
plt.title('Non-linear Queue Penalty Mechanism')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('paper/figures/fig1_reward_curve.png', dpi=300, bbox_inches='tight')
```

### Fig. 6 — SUMO Screenshot
Run SUMO with GUI:
```bash
sumo-gui -c sumo_env/single_intersection.sumocfg
```
Take screenshot during peak congestion (~step 400-500).

## LaTeX Include Example
```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/fig1_reward_curve.png}
    \caption{Reward function behaviour as a function of total queue length...}
    \label{fig:reward}
\end{figure}
```
