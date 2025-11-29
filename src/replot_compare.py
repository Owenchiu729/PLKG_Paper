import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

models = ["encoder", "fnn", "cnn", "cnntp"]
colors = ["blue", "orange", "green", "red"]

log_dir = "logs"

plt.figure(figsize=(8, 5), dpi=150)

# y-scale = กั10^-3
yexp = -3
scale = 10 ** (-yexp)

for name, color in zip(models, colors):
    path = os.path.join(log_dir, f"{name}.csv")
    
    if not os.path.exists(path):
        print(f"[WARN] {path} not found, skip")
        continue
    
    df = pd.read_csv(path)

    if "val_mse" not in df.columns:
        print(f"[WARN] val_mse not in {path}, skip")
        continue

    y = df["val_mse"].values * scale
    plt.plot(df["epoch"], y, label=name.upper(), color=color, linewidth=2)

plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("Model Comparison (MSE)")
plt.grid(alpha=0.3)
plt.legend()

# --------------------------------------------------
# Scientific notation (กั10^-3)
# --------------------------------------------------

# Use normal numeric ticks (we already scaled by 10^-3)
# but still enable scientific formatter so tick spacing is nice
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

# Hide the default offset text in the top-right corner
plt.gca().yaxis.get_offset_text().set_visible(False)

# Manually place "กั10^-3" near the Y-axis label (paper-style)
plt.text(
    -0.14, 1.02,
    r"$\times 10^{-3}$",
    transform=plt.gca().transAxes,
    fontsize=14,
)

plt.tight_layout()

out_file = "compare_mse_replot.png"
plt.savefig(out_file)
print(f"[DONE] Saved figure --> {out_file}")
