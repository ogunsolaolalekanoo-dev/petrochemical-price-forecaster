# reports/model_comparison_chart.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("reports", exist_ok=True)

# ── Your actual results ───────────────────────────────────────────────
models = [
    "Prophet\n(Tuned)",
    "Auto\nARIMAX",
    "LightGBM\n(Tuned)",
    "Ensemble\n(Prophet +\nLightGBM)"
]

metrics = {
    "MAE":  [8.118,  1.525,  8.349,  4.208],
    "RMSE": [9.790,  2.008, 12.379,  6.238],
    "MAPE": [3.02,  None,   2.97,   1.50],   # ARIMAX excluded (artifact)
    "R²":   [0.4721, 0.0123, 0.1560, 0.7857],
}

colors = {
    "Prophet\n(Tuned)":              "#2E75B6",
    "Auto\nARIMAX":                  "#ED7D31",
    "LightGBM\n(Tuned)":             "#70AD47",
    "Ensemble\n(Prophet +\nLightGBM)": "#C00000",
}
bar_colors = list(colors.values())

# ── Pre-shock vs shock data ───────────────────────────────────────────
shock_models  = ["LightGBM", "Prophet"]
pre_shock_mape  = [2.50, 2.79]
shock_mape      = [5.80, 4.42]
pre_shock_r2    = [0.0061, -0.3693]
shock_r2        = [0.0490,  0.7561]

# ── Figure layout ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor("#0F1117")

title = fig.suptitle(
    "Petrochemical Resin Price Forecasting\nModel Comparison Dashboard",
    fontsize=22, fontweight="bold",
    color="white", y=0.98
)

subtitle_text = (
    "Ensemble (Prophet + LightGBM)  |  "
    "MAPE: 1.50%  |  R²: 0.786  |  "
    "Test Period: 2023–2026  |  26 Years Training Data"
)
fig.text(
    0.5, 0.955, subtitle_text,
    ha="center", fontsize=13,
    color="#AAAAAA", style="italic"
)


def style_ax(ax, title_text, xlabel="", ylabel=""):
    ax.set_facecolor("#1A1D2E")
    ax.set_title(title_text, fontsize=13,
                 fontweight="bold", color="white", pad=12)
    ax.set_xlabel(xlabel, fontsize=11, color="#AAAAAA")
    ax.set_ylabel(ylabel, fontsize=11, color="#AAAAAA")
    ax.tick_params(colors="#AAAAAA", labelsize=10)
    ax.spines["bottom"].set_color("#333355")
    ax.spines["left"].set_color("#333355")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#333355",
                  linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)


# ─────────────────────────────────────────────────────────────────────
# ROW 1: MAE and RMSE side by side
# ─────────────────────────────────────────────────────────────────────
ax1 = fig.add_axes([0.06, 0.76, 0.40, 0.14])
ax2 = fig.add_axes([0.55, 0.76, 0.40, 0.14])

x    = np.arange(len(models))
w    = 0.6

# MAE
bars = ax1.bar(x, metrics["MAE"], width=w,
               color=bar_colors, edgecolor="#0F1117",
               linewidth=1.2, zorder=3)
for bar, val in zip(bars, metrics["MAE"]):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        f"{val:.3f}", ha="center", va="bottom",
        fontsize=10, color="white", fontweight="bold"
    )
# Highlight best
bars[3].set_edgecolor("gold")
bars[3].set_linewidth(2.5)
ax1.set_xticks(x)
ax1.set_xticklabels(models, color="#AAAAAA", fontsize=9)
style_ax(ax1,
         "① Mean Absolute Error (MAE)  —  Lower is Better",
         ylabel="MAE (PPI Index Points)")
ax1.text(3, metrics["MAE"][3] / 2, "BEST",
         ha="center", va="center",
         fontsize=9, color="gold", fontweight="bold")

# RMSE
bars2 = ax2.bar(x, metrics["RMSE"], width=w,
                color=bar_colors, edgecolor="#0F1117",
                linewidth=1.2, zorder=3)
for bar, val in zip(bars2, metrics["RMSE"]):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        f"{val:.3f}", ha="center", va="bottom",
        fontsize=10, color="white", fontweight="bold"
    )
bars2[3].set_edgecolor("gold")
bars2[3].set_linewidth(2.5)
ax2.set_xticks(x)
ax2.set_xticklabels(models, color="#AAAAAA", fontsize=9)
style_ax(ax2,
         "② Root Mean Squared Error (RMSE)  —  Lower is Better",
         ylabel="RMSE (PPI Index Points)")
ax2.text(3, metrics["RMSE"][3] / 2, "BEST",
         ha="center", va="center",
         fontsize=9, color="gold", fontweight="bold")
ax2.text(1, metrics["RMSE"][1] + 0.3,
         "Different\nunits*",
         ha="center", va="bottom",
         fontsize=8, color="#FF9999", style="italic")


# ─────────────────────────────────────────────────────────────────────
# ROW 2: MAPE and R² side by side
# ─────────────────────────────────────────────────────────────────────
ax3 = fig.add_axes([0.06, 0.56, 0.40, 0.14])
ax4 = fig.add_axes([0.55, 0.56, 0.40, 0.14])

# MAPE — exclude ARIMAX (N/A)
mape_vals    = [3.02, 0, 2.97, 1.50]
mape_colors  = [bar_colors[0], "#444444",
                bar_colors[2], bar_colors[3]]

bars3 = ax3.bar(x, mape_vals, width=w,
                color=mape_colors, edgecolor="#0F1117",
                linewidth=1.2, zorder=3)
for i, (bar, val) in enumerate(zip(bars3, mape_vals)):
    if i == 1:
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            0.3, "N/A*",
            ha="center", va="bottom",
            fontsize=9, color="#FF9999",
            fontweight="bold"
        )
    else:
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.05,
            f"{val:.2f}%",
            ha="center", va="bottom",
            fontsize=10, color="white", fontweight="bold"
        )

# Threshold lines
ax3.axhline(5,  color="#FFD700", linestyle="--",
            linewidth=1.2, alpha=0.8, zorder=2)
ax3.axhline(10, color="#FF6B6B", linestyle="--",
            linewidth=1.2, alpha=0.8, zorder=2)
ax3.text(3.55, 5.1,  "5% Excellent",
         color="#FFD700", fontsize=8)
ax3.text(3.55, 10.1, "10% Good",
         color="#FF6B6B", fontsize=8)

bars3[3].set_edgecolor("gold")
bars3[3].set_linewidth(2.5)
ax3.set_xticks(x)
ax3.set_xticklabels(models, color="#AAAAAA", fontsize=9)
ax3.set_ylim(0, 14)
style_ax(ax3,
         "③ Mean Absolute % Error (MAPE)  —  Lower is Better",
         ylabel="MAPE (%)")
ax3.text(0.02, 0.92,
         "*ARIMAX MAPE excluded — mathematical artifact\n"
         " (136.89% due to near-zero % change denominators)",
         transform=ax3.transAxes,
         fontsize=7.5, color="#FF9999",
         style="italic", va="top")

# R²
r2_colors_adj = []
for val, c in zip(metrics["R²"], bar_colors):
    r2_colors_adj.append(c if val >= 0 else "#FF6B6B")

bars4 = ax4.bar(x, metrics["R²"], width=w,
                color=r2_colors_adj, edgecolor="#0F1117",
                linewidth=1.2, zorder=3)
for bar, val in zip(bars4, metrics["R²"]):
    ypos = val + 0.01 if val >= 0 else val - 0.03
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        ypos, f"{val:.4f}",
        ha="center",
        va="bottom" if val >= 0 else "top",
        fontsize=10, color="white", fontweight="bold"
    )

ax4.axhline(0, color="#AAAAAA", linewidth=1.0,
            linestyle="-", zorder=2)
bars4[3].set_edgecolor("gold")
bars4[3].set_linewidth(2.5)
ax4.set_xticks(x)
ax4.set_xticklabels(models, color="#AAAAAA", fontsize=9)
style_ax(ax4,
         "④ R² (Coefficient of Determination)  —  Higher is Better",
         ylabel="R²")
ax4.text(3, metrics["R²"][3] / 2, "BEST",
         ha="center", va="center",
         fontsize=9, color="gold", fontweight="bold")
ax4.text(0.02, 0.05,
         "Negative R² = worse than predicting the mean",
         transform=ax4.transAxes,
         fontsize=8, color="#FF9999", style="italic")


# ─────────────────────────────────────────────────────────────────────
# ROW 3: Pre-Shock vs Shock MAPE comparison
# ─────────────────────────────────────────────────────────────────────
ax5 = fig.add_axes([0.06, 0.36, 0.40, 0.14])

x2 = np.arange(len(shock_models))
w2 = 0.3

bars_pre   = ax5.bar(x2 - w2/2, pre_shock_mape,
                     width=w2, color="#2E75B6",
                     label="Pre-Shock (2023–2025)",
                     edgecolor="#0F1117", zorder=3)
bars_shock = ax5.bar(x2 + w2/2, shock_mape,
                     width=w2, color="#C00000",
                     label="Shock Period (2026+)",
                     edgecolor="#0F1117", zorder=3)

for bar, val in zip(bars_pre, pre_shock_mape):
    ax5.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.05, f"{val:.2f}%",
        ha="center", va="bottom",
        fontsize=10, color="white", fontweight="bold"
    )
for bar, val in zip(bars_shock, shock_mape):
    ax5.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.05, f"{val:.2f}%",
        ha="center", va="bottom",
        fontsize=10, color="white", fontweight="bold"
    )

ax5.axhline(5,  color="#FFD700", linestyle="--",
            linewidth=1.2, alpha=0.8)
ax5.axhline(10, color="#FF6B6B", linestyle="--",
            linewidth=1.2, alpha=0.8)
ax5.text(1.55, 5.1,  "5% threshold",
         color="#FFD700", fontsize=8)
ax5.text(1.55, 10.1, "10% threshold",
         color="#FF6B6B", fontsize=8)
ax5.set_xticks(x2)
ax5.set_xticklabels(shock_models,
                    color="#AAAAAA", fontsize=11)
ax5.set_ylim(0, 13)
ax5.legend(fontsize=9, facecolor="#1A1D2E",
           labelcolor="white", framealpha=0.8)
style_ax(ax5,
         "⑤ MAPE: Pre-Shock vs Shock Period  —  Lower is Better",
         ylabel="MAPE (%)")
ax5.text(0.02, 0.88,
         "Both models stayed under 6% MAPE even during\n"
         "an unprecedented geopolitical black swan event",
         transform=ax5.transAxes,
         fontsize=8.5, color="#7EC8E3", style="italic")


# ─────────────────────────────────────────────────────────────────────
# ROW 3 RIGHT: Radar / Spider chart — overall model profile
# ─────────────────────────────────────────────────────────────────────
ax6 = fig.add_axes([0.55, 0.36, 0.40, 0.14],
                   projection="polar")
ax6.set_facecolor("#1A1D2E")

categories  = ["Low MAE", "Low RMSE",
               "Low MAPE", "High R²"]
N           = len(categories)
angles      = [n / float(N) * 2 * np.pi for n in range(N)]
angles     += angles[:1]

# Normalize scores 0–1 (higher = better for all)
# MAE:  lower better → invert
# RMSE: lower better → invert
# MAPE: lower better → invert (ARIMAX excluded)
# R²:   higher better

max_mae  = max(metrics["MAE"])
max_rmse = max(metrics["RMSE"])
max_mape = max([v for v in metrics["MAPE"] if v])
max_r2   = max(metrics["R²"])
min_r2   = min(metrics["R²"])

def normalize_model(idx):
    mae_s  = 1 - metrics["MAE"][idx]  / max_mae
    rmse_s = 1 - metrics["RMSE"][idx] / max_rmse
    mape_v = metrics["MAPE"][idx]
    mape_s = (1 - mape_v / max_mape) if mape_v else 0.5
    r2_s   = ((metrics["R²"][idx] - min_r2) /
               (max_r2 - min_r2 + 1e-6))
    return [mae_s, rmse_s, mape_s, r2_s]

radar_colors = ["#2E75B6", "#ED7D31",
                "#70AD47", "#C00000"]
radar_labels = ["Prophet", "ARIMAX",
                "LightGBM", "Ensemble"]

for i, (rc, rl) in enumerate(
        zip(radar_colors, radar_labels)):
    vals  = normalize_model(i)
    vals += vals[:1]
    ax6.plot(angles, vals, color=rc,
             linewidth=2, linestyle="solid")
    ax6.fill(angles, vals, color=rc, alpha=0.08)

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(categories,
                    color="white", fontsize=9)
ax6.set_yticks([0.25, 0.5, 0.75, 1.0])
ax6.set_yticklabels(["0.25", "0.5",
                      "0.75", "1.0"],
                    color="#777777", fontsize=7)
ax6.grid(color="#333355", linewidth=0.8)
ax6.spines["polar"].set_color("#333355")
ax6.set_title("⑥ Model Profile (Normalized)",
              fontsize=12, fontweight="bold",
              color="white", pad=15)

legend_patches = [
    mpatches.Patch(color=c, label=l)
    for c, l in zip(radar_colors, radar_labels)
]
ax6.legend(handles=legend_patches,
           loc="upper right",
           bbox_to_anchor=(1.35, 1.15),
           fontsize=8, facecolor="#1A1D2E",
           labelcolor="white", framealpha=0.8)


# ─────────────────────────────────────────────────────────────────────
# ROW 4: Summary scorecard table
# ─────────────────────────────────────────────────────────────────────
ax7 = fig.add_axes([0.06, 0.15, 0.88, 0.16])
ax7.set_facecolor("#1A1D2E")
ax7.axis("off")
ax7.set_title("⑦ Complete Model Scorecard",
              fontsize=13, fontweight="bold",
              color="white", pad=12, loc="left")

col_labels = [
    "Model", "MAE↓", "RMSE↓", "MAPE↓",
    "R²↑", "Pre-Shock\nMAPE↓",
    "Shock\nMAPE↓", "Verdict"
]
row_data = [
    ["Prophet (Tuned)",
     "8.118", "9.790", "3.02%",
     "0.4721", "2.79%", "4.42%", "🏆 Excellent"],
    ["Auto ARIMAX",
     "1.525*", "2.008*", "N/A†",
     "0.0123", "—", "—", "📊 Momentum Model"],
    ["LightGBM (Tuned)",
     "8.349", "12.379", "2.97%",
     "0.1560", "2.50%", "5.80%", "🏆 Excellent"],
    ["Ensemble ⭐ BEST",
     "4.208", "6.238", "1.50%",
     "0.7857", "—", "—", "🥇 WINNER"],
]

table = ax7.table(
    cellText=row_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 0.88]
)
table.auto_set_font_size(False)
table.set_fontsize(10)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#2E3555")
    cell.set_text_props(color="white",
                        fontweight="bold")
    cell.set_edgecolor("#444466")

# Style rows
row_colors = [
    "#1E2A4A",  # Prophet
    "#2A1E1E",  # ARIMAX
    "#1E2A1E",  # LightGBM
    "#2A1E00",  # Ensemble
]
for i, rc in enumerate(row_colors, 1):
    for j in range(len(col_labels)):
        cell = table[i, j]
        cell.set_facecolor(rc)
        cell.set_text_props(color="white")
        cell.set_edgecolor("#444466")

# Highlight ensemble row
for j in range(len(col_labels)):
    cell = table[4, j]
    cell.set_facecolor("#3D2200")
    cell.set_text_props(color="gold",
                        fontweight="bold")
    cell.set_edgecolor("gold")

# Footnotes
ax7.text(
    0.0, -0.08,
    "* ARIMAX metrics are in % change units "
    "(not PPI index points) — different scale to other models\n"
    "† ARIMAX MAPE = 136.89% — mathematical artifact: "
    "near-zero denominators in monthly % change forecasting. "
    "MAE of 1.525 pp is the honest metric.",
    transform=ax7.transAxes,
    fontsize=8, color="#AAAAAA", style="italic"
)


# ─────────────────────────────────────────────────────────────────────
# ROW 5: Key findings text box
# ─────────────────────────────────────────────────────────────────────
ax8 = fig.add_axes([0.06, 0.03, 0.88, 0.10])
ax8.set_facecolor("#1A1D2E")
ax8.axis("off")
ax8.set_title("⑧ Key Findings & Business Implications",
              fontsize=12, fontweight="bold",
              color="white", pad=8, loc="left")

findings = [
    "🎯  Ensemble (Prophet + LightGBM) achieved 1.50% MAPE and R²=0.786 — explaining 78.6% of resin price variance over the 2023–2026 test period",
    "📈  3-month resin price moving average (resin_ma_3m) is the single strongest predictor — confirming strong short-term momentum in resin markets",
    "🛢️   RSI technical indicator ranked 3rd in feature importance — resin prices exhibit mean-reverting behavior with overbought/oversold signals",
    "🏗️   Crude × Housing interaction feature ranked 4th — compound demand + cost pressure amplifies resin price movements beyond individual effects",
    "⚡  Both models stayed under 6% MAPE during the 2026 Middle East crude oil shock — an unprecedented geopolitical black swan event",
    "💼  At resin PPI=280, 1.50% MAPE = ±4.2 index points average error — translates to highly defensible procurement budget planning and supplier negotiation support",
]

for i, finding in enumerate(findings):
    ax8.text(
        0.01, 0.88 - i * 0.155,
        finding,
        transform=ax8.transAxes,
        fontsize=8.8, color="white",
        va="top"
    )

# ─────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────
output_path = "reports/model_comparison_chart.png"
plt.savefig(output_path, dpi=180,
            bbox_inches="tight",
            facecolor="#0F1117")
plt.close()

print(f"✅ Model comparison chart saved → {output_path}")
print("   Open reports/model_comparison_chart.png to view")