import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

history_path = os.path.join(RESULTS_DIR, "history.pkl")
loss_plot_path = os.path.join(RESULTS_DIR, "loss_curve.png")
acc_plot_path = os.path.join(RESULTS_DIR, "accuracy_curve.png")
summary_csv_path = os.path.join(RESULTS_DIR, "training_summary.csv")
summary_txt_path = os.path.join(RESULTS_DIR, "training_summary.txt")

# -----------------------------
# Load history
# -----------------------------
if not os.path.exists(history_path):
    raise FileNotFoundError(f"❌ History file not found at {history_path}. Run train_model.py first.")

with open(history_path, "rb") as f:
    history = pickle.load(f)

# Convert to numpy arrays for easy math
train_loss = np.array(history["train_loss"])
val_loss = np.array(history["val_loss"])
train_acc = np.array(history["train_acc"])
val_acc = np.array(history["val_acc"])

# Compute averages & best values
avg_train_loss = train_loss.mean()
avg_val_loss = val_loss.mean()
avg_train_acc = train_acc.mean()
avg_val_acc = val_acc.mean()

best_train_acc = train_acc.max()
best_val_acc = val_acc.max()
best_train_loss = train_loss.min()
best_val_loss = val_loss.min()

# -----------------------------
# Print Summary
# -----------------------------
print("📊 Summary of Training:")
print(f"   - Average Train Loss: {avg_train_loss:.4f}")
print(f"   - Average Validation Loss: {avg_val_loss:.4f}")
print(f"   - Average Train Accuracy: {avg_train_acc:.4f}")
print(f"   - Average Validation Accuracy: {avg_val_acc:.4f}")
print(f"   - Best Train Accuracy: {best_train_acc:.4f}")
print(f"   - Best Validation Accuracy: {best_val_acc:.4f}")
print(f"   - Best Train Loss: {best_train_loss:.4f}")
print(f"   - Best Validation Loss: {best_val_loss:.4f}")

# -----------------------------
# Export Summary (CSV + TXT)
# -----------------------------
summary_data = {
    "Metric": [
        "Average Train Loss", "Average Validation Loss",
        "Average Train Accuracy", "Average Validation Accuracy",
        "Best Train Loss", "Best Validation Loss",
        "Best Train Accuracy", "Best Validation Accuracy"
    ],
    "Value": [
        avg_train_loss, avg_val_loss,
        avg_train_acc, avg_val_acc,
        best_train_loss, best_val_loss,
        best_train_acc, best_val_acc
    ]
}

# Save CSV
df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(summary_csv_path, index=False)
print(f"📑 Summary CSV saved to {summary_csv_path}")

# Save TXT
with open(summary_txt_path, "w") as f:
    for metric, value in zip(summary_data["Metric"], summary_data["Value"]):
        f.write(f"{metric}: {value:.4f}\n")
print(f"📑 Summary TXT saved to {summary_txt_path}")

# -----------------------------
# Plot Loss
# -----------------------------
plt.figure(figsize=(8, 6))
plt.plot(train_loss, label="Train Loss", marker="o")
plt.plot(val_loss, label="Validation Loss", marker="o")
plt.axhline(avg_train_loss, color="blue", linestyle="--", alpha=0.6,
            label=f"Avg Train Loss = {avg_train_loss:.3f}")
plt.axhline(avg_val_loss, color="orange", linestyle="--", alpha=0.6,
            label=f"Avg Val Loss = {avg_val_loss:.3f}")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(loss_plot_path)
print(f"📉 Loss curve saved to {loss_plot_path}")
plt.show()

# -----------------------------
# Plot Accuracy
# -----------------------------
plt.figure(figsize=(8, 6))
plt.plot(train_acc, label="Train Accuracy", marker="o")
plt.plot(val_acc, label="Validation Accuracy", marker="o")
plt.axhline(avg_train_acc, color="blue", linestyle="--", alpha=0.6,
            label=f"Avg Train Acc = {avg_train_acc:.3f}")
plt.axhline(avg_val_acc, color="orange", linestyle="--", alpha=0.6,
            label=f"Avg Val Acc = {avg_val_acc:.3f}")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(acc_plot_path)
print(f"📈 Accuracy curve saved to {acc_plot_path}")
plt.show()
