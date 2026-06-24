# reports/model_selection_chart.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import os

os.makedirs("reports", exist_ok=True)

fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor("#0F1117")


def styled_box(ax, x, y, w, h, color,
               alpha=0.15, radius=0.03,
               border_color=None, border_width=2):
    fancy = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=color, alpha=alpha,
        edgecolor=border_color or color,
        linewidth=border_width,
        transform=ax.transAxes,
        clip_on=False
    )
    ax.add_patch(fancy)


def style_ax(ax):
    ax.set_facecolor("#0F1117")
    ax.axis("off")


# ─────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.975,
    "Model Selection Strategy",
    ha="center", fontsize=26,
    fontweight="bold", color="white"
)
fig.text(
    0.5, 0.957,
    "Why We Use An Ensemble of Prophet + LightGBM  "
    "|  Petrochemical Resin Price Forecasting System",
    ha="center", fontsize=13,
    color="#AAAAAA", style="italic"
)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1 — The Three Candidate Models
# ─────────────────────────────────────────────────────────────────────
ax1 = fig.add_axes([0.03, 0.75, 0.94, 0.17])
style_ax(ax1)

ax1.text(
    0.5, 0.97,
    "STEP 1 — Three Candidate Models Evaluated",
    ha="center", va="top", fontsize=15,
    fontweight="bold", color="white",
    transform=ax1.transAxes
)

# ── Model boxes ──────────────────────────────────────────────────────
models_info = [
    {
        "name":    "Prophet",
        "creator": "Built by Meta (Facebook) · 2017",
        "color":   "#2E75B6",
        "x":       0.01,
        "icon":    "🔮",
        "type":    "Decomposition Model",
        "how": [
            "Splits resin price into 3 components:",
            "  Trend → long-term price direction",
            "  Seasonality → annual demand cycles",
            "  Regressors → crude oil, housing, RSI",
            "Forecasts each component separately",
            "then recombines for final forecast",
        ],
        "best_at": [
            "✅ Long-term trend direction",
            "✅ Annual seasonality patterns",
            "✅ Confidence interval generation",
            "✅ Structural break handling",
            "✅ Non-technical explainability",
        ],
        "weak_at": [
            "⚠️  Short-term price momentum",
            "⚠️  Non-linear feature interactions",
            "⚠️  Flat market periods (overestimates)",
        ],
        "result": "MAPE: 3.02%  |  R²: 0.472",
        "verdict": "✅ SELECTED",
        "verdict_color": "#70AD47",
    },
    {
        "name":    "Auto ARIMAX",
        "creator": "Classical Statistics · Box-Jenkins",
        "color":   "#ED7D31",
        "x":       0.345,
        "icon":    "📈",
        "type":    "Statistical Momentum Model",
        "how": [
            "Models monthly % changes (stationary)",
            "AR(3) = 3-month momentum structure",
            "X = exogenous variables (crude oil,",
            "  gas, housing, commodity PPI)",
            "Auto-selected via AIC minimization",
            "order: ARIMA(3,0,0)",
        ],
        "best_at": [
            "✅ Short-term momentum quantification",
            "✅ Coefficient interpretability",
            "✅ Statistically rigorous framework",
            "✅ Detecting mean-reversion patterns",
            "✅ AIC-optimized complexity control",
        ],
        "weak_at": [
            "❌ NOT SELECTED for forecasting",
            "⚠️  Different output units (% change)",
            "⚠️  MAPE misleading near zero values",
            "⚠️  Linear — misses complex patterns",
        ],
        "result": "MAE: 1.525pp  |  AIC: 1087.82",
        "verdict": "📊 INSIGHTS ONLY",
        "verdict_color": "#ED7D31",
    },
    {
        "name":    "LightGBM",
        "creator": "Built by Microsoft · 2017",
        "color":   "#70AD47",
        "x":       0.68,
        "icon":    "🌲",
        "type":    "Gradient Boosting Model",
        "how": [
            "Builds 500 decision trees sequentially",
            "Each tree corrects errors of previous",
            "Regularization prevents overfitting:",
            "  L1 (reg_alpha=0.5) — sparsity",
            "  L2 (reg_lambda=0.5) — shrinkage",
            "48 engineered features as inputs",
        ],
        "best_at": [
            "✅ Non-linear feature relationships",
            "✅ Interaction feature capture",
            "✅ Short-term momentum patterns",
            "✅ SHAP explainability",
            "✅ Handles 48 features efficiently",
        ],
        "weak_at": [
            "⚠️  Long-term trend extrapolation",
            "⚠️  No built-in confidence intervals",
            "⚠️  Low R² in low-variance periods",
        ],
        "result": "MAPE: 2.97%  |  R²: 0.156",
        "verdict": "✅ SELECTED",
        "verdict_color": "#70AD47",
    },
]

for m in models_info:
    bx = m["x"]

    # Main card background
    styled_box(ax1, bx, 0.02, 0.30, 0.90,
               m["color"], alpha=0.12,
               border_color=m["color"],
               border_width=2)

    # Icon + name
    ax1.text(
        bx + 0.15, 0.90,
        f"{m['icon']}  {m['name']}",
        ha="center", va="top",
        fontsize=15, fontweight="bold",
        color=m["color"],
        transform=ax1.transAxes
    )
    ax1.text(
        bx + 0.15, 0.82,
        m["creator"],
        ha="center", va="top",
        fontsize=8.5, color="#AAAAAA",
        style="italic",
        transform=ax1.transAxes
    )
    ax1.text(
        bx + 0.15, 0.75,
        m["type"],
        ha="center", va="top",
        fontsize=9, color="white",
        fontweight="bold",
        transform=ax1.transAxes
    )

    # How it works
    ax1.text(
        bx + 0.01, 0.68,
        "HOW IT WORKS:",
        va="top", fontsize=8,
        color="#AAAAAA", fontweight="bold",
        transform=ax1.transAxes
    )
    for i, line in enumerate(m["how"]):
        ax1.text(
            bx + 0.01, 0.62 - i * 0.076,
            line, va="top", fontsize=7.8,
            color="#DDDDDD",
            transform=ax1.transAxes
        )

    # Result
    ax1.text(
        bx + 0.15, 0.10,
        m["result"],
        ha="center", va="top",
        fontsize=8.5, color="white",
        fontweight="bold",
        transform=ax1.transAxes
    )

    # Verdict badge
    styled_box(ax1, bx + 0.03, 0.02, 0.24, 0.075,
               m["verdict_color"], alpha=0.25,
               border_color=m["verdict_color"],
               border_width=2)
    ax1.text(
        bx + 0.15, 0.06,
        m["verdict"],
        ha="center", va="center",
        fontsize=10, color=m["verdict_color"],
        fontweight="bold",
        transform=ax1.transAxes
    )


# ─────────────────────────────────────────────────────────────────────
# SECTION 2 — Strengths & Weaknesses detail
# ─────────────────────────────────────────────────────────────────────
ax2 = fig.add_axes([0.03, 0.52, 0.94, 0.21])
style_ax(ax2)

ax2.text(
    0.5, 0.98,
    "STEP 2 — Why Each Model Was Accepted or Rejected",
    ha="center", va="top", fontsize=15,
    fontweight="bold", color="white",
    transform=ax2.transAxes
)

for m in models_info:
    bx = m["x"]

    # Best at
    styled_box(ax2, bx, 0.52, 0.30, 0.42,
               m["color"], alpha=0.08,
               border_color=m["color"],
               border_width=1.5)
    ax2.text(
        bx + 0.01, 0.92,
        "STRENGTHS:",
        va="top", fontsize=8.5,
        color=m["color"], fontweight="bold",
        transform=ax2.transAxes
    )
    for i, line in enumerate(m["best_at"]):
        ax2.text(
            bx + 0.01, 0.85 - i * 0.082,
            line, va="top", fontsize=8,
            color="#DDDDDD",
            transform=ax2.transAxes
        )

    # Weak at
    styled_box(ax2, bx, 0.04, 0.30, 0.44,
               "#FF6B6B", alpha=0.06,
               border_color="#FF6B6B",
               border_width=1.5)
    ax2.text(
        bx + 0.01, 0.46,
        "LIMITATIONS:",
        va="top", fontsize=8.5,
        color="#FF9999", fontweight="bold",
        transform=ax2.transAxes
    )
    for i, line in enumerate(m["weak_at"]):
        ax2.text(
            bx + 0.01, 0.39 - i * 0.09,
            line, va="top", fontsize=8,
            color="#FFAAAA",
            transform=ax2.transAxes
        )


# ─────────────────────────────────────────────────────────────────────
# SECTION 3 — Why ARIMAX is excluded from ensemble
# ─────────────────────────────────────────────────────────────────────
ax3 = fig.add_axes([0.03, 0.41, 0.94, 0.09])
style_ax(ax3)

styled_box(ax3, 0.0, 0.0, 1.0, 1.0,
           "#ED7D31", alpha=0.08,
           border_color="#ED7D31",
           border_width=1.5)

ax3.text(
    0.5, 0.88,
    "WHY ARIMAX IS EXCLUDED FROM THE ENSEMBLE FORECAST",
    ha="center", va="top", fontsize=13,
    fontweight="bold", color="#ED7D31",
    transform=ax3.transAxes
)

reasons = [
    (
        "📐  Different Output Units",
        "ARIMAX forecasts monthly % changes in resin PPI.\n"
        "Prophet and LightGBM forecast raw PPI index levels.\n"
        "Combining them directly is mathematically incorrect\n"
        "— like averaging kilometers and miles."
    ),
    (
        "📊  MAPE Metric Breakdown",
        "ARIMAX MAPE = 136.89% — not a real error.\n"
        "When actual % changes are near zero (0.1%),\n"
        "even tiny absolute errors produce huge % errors.\n"
        "MAE of 1.53pp is the honest performance metric."
    ),
    (
        "💡  How ARIMAX Is Still Used",
        "ARIMAX provides strategic insights:\n"
        "AR(1)=0.46 confirms 1-month momentum\n"
        "AR(3)=-0.14 confirms 3-month mean reversion\n"
        "These findings inform procurement timing rules."
    ),
    (
        "✅  The Right Role For ARIMAX",
        "ARIMAX = analytical intelligence tool,\n"
        "NOT a price level forecasting tool.\n"
        "Its coefficients validate the momentum patterns\n"
        "that LightGBM captures non-linearly."
    ),
]

for i, (title, body) in enumerate(reasons):
    x_pos = 0.01 + i * 0.248
    ax3.text(
        x_pos, 0.72, title,
        va="top", fontsize=9,
        color="#ED7D31", fontweight="bold",
        transform=ax3.transAxes
    )
    for j, line in enumerate(body.split("\n")):
        ax3.text(
            x_pos, 0.58 - j * 0.13,
            line, va="top", fontsize=8,
            color="#DDDDDD",
            transform=ax3.transAxes
        )


# ─────────────────────────────────────────────────────────────────────
# SECTION 4 — The Ensemble Architecture
# ─────────────────────────────────────────────────────────────────────
ax4 = fig.add_axes([0.03, 0.22, 0.94, 0.17])
style_ax(ax4)

ax4.text(
    0.5, 0.97,
    "STEP 3 — The Ensemble Architecture",
    ha="center", va="top", fontsize=15,
    fontweight="bold", color="white",
    transform=ax4.transAxes
)

# Prophet box
styled_box(ax4, 0.02, 0.20, 0.22, 0.65,
           "#2E75B6", alpha=0.15,
           border_color="#2E75B6", border_width=2)
ax4.text(
    0.13, 0.83,
    "🔮 Prophet",
    ha="center", va="top", fontsize=12,
    fontweight="bold", color="#2E75B6",
    transform=ax4.transAxes
)
ax4.text(
    0.13, 0.73,
    "Captures:\n"
    "• Long-term trend\n"
    "• Seasonal patterns\n"
    "• Confidence intervals\n"
    "• Structural breaks\n\n"
    "Weight: 49.6%\n"
    "(inverse MAPE)",
    ha="center", va="top", fontsize=8.5,
    color="#DDDDDD",
    transform=ax4.transAxes
)

# LightGBM box
styled_box(ax4, 0.27, 0.20, 0.22, 0.65,
           "#70AD47", alpha=0.15,
           border_color="#70AD47", border_width=2)
ax4.text(
    0.38, 0.83,
    "🌲 LightGBM",
    ha="center", va="top", fontsize=12,
    fontweight="bold", color="#70AD47",
    transform=ax4.transAxes
)
ax4.text(
    0.38, 0.73,
    "Captures:\n"
    "• Short-term momentum\n"
    "• Non-linear patterns\n"
    "• Interaction effects\n"
    "• 48-feature signals\n\n"
    "Weight: 50.4%\n"
    "(inverse MAPE)",
    ha="center", va="top", fontsize=8.5,
    color="#DDDDDD",
    transform=ax4.transAxes
)

# Arrow right
ax4.annotate(
    "", xy=(0.535, 0.52),
    xytext=(0.495, 0.52),
    xycoords="axes fraction",
    arrowprops=dict(
        arrowstyle="->",
        color="white",
        lw=2.5
    )
)

# Weighting box
styled_box(ax4, 0.54, 0.30, 0.18, 0.45,
           "#FFD700", alpha=0.10,
           border_color="#FFD700", border_width=2)
ax4.text(
    0.63, 0.78,
    "⚖️ Weighting",
    ha="center", va="top", fontsize=11,
    fontweight="bold", color="#FFD700",
    transform=ax4.transAxes
)
ax4.text(
    0.63, 0.67,
    "Inverse-MAPE\nWeighting\n\n"
    "Better model\ngets higher weight\nautomatically\n\n"
    "w = 1 / MAPE\nnormalized to\nsum to 100%",
    ha="center", va="top", fontsize=8,
    color="#DDDDDD",
    transform=ax4.transAxes
)

# Arrow right
ax4.annotate(
    "", xy=(0.755, 0.52),
    xytext=(0.725, 0.52),
    xycoords="axes fraction",
    arrowprops=dict(
        arrowstyle="->",
        color="white",
        lw=2.5
    )
)

# Ensemble result box
styled_box(ax4, 0.76, 0.10, 0.22, 0.80,
           "#C00000", alpha=0.20,
           border_color="gold", border_width=3)
ax4.text(
    0.87, 0.90,
    "🥇 ENSEMBLE",
    ha="center", va="top", fontsize=13,
    fontweight="bold", color="gold",
    transform=ax4.transAxes
)
ax4.text(
    0.87, 0.79,
    "Prophet + LightGBM\nCombined Forecast",
    ha="center", va="top", fontsize=9,
    color="#DDDDDD",
    transform=ax4.transAxes
)

results_text = [
    ("MAPE",  "1.50%",  "#70AD47"),
    ("RMSE",  "6.238",  "#70AD47"),
    ("MAE",   "4.208",  "#70AD47"),
    ("R²",    "0.786",  "#70AD47"),
]
for i, (label, val, col) in enumerate(results_text):
    ax4.text(
        0.80, 0.60 - i * 0.11,
        f"{label}:", va="top",
        fontsize=9, color="#AAAAAA",
        transform=ax4.transAxes
    )
    ax4.text(
        0.94, 0.60 - i * 0.11,
        val, va="top", ha="right",
        fontsize=9, color=col,
        fontweight="bold",
        transform=ax4.transAxes
    )

ax4.text(
    0.87, 0.17,
    "Explains 78.6% of\nresin price variance",
    ha="center", va="top",
    fontsize=8.5, color="gold",
    fontweight="bold",
    transform=ax4.transAxes
)

# Why ensemble beats individuals
styled_box(ax4, 0.02, 0.0, 0.70, 0.17,
           "#FFFFFF", alpha=0.04,
           border_color="#555577",
           border_width=1)
ax4.text(
    0.01, 0.155,
    "💡 Why The Ensemble Beats Both Individual Models:",
    va="top", fontsize=9,
    color="#AAAAAA", fontweight="bold",
    transform=ax4.transAxes
)
ax4.text(
    0.01, 0.095,
    "Prophet and LightGBM make errors in DIFFERENT months — "
    "when one overestimates, the other often underestimates. "
    "Averaging their\n"
    "predictions causes errors to partially cancel out, "
    "producing a smoother, more accurate combined forecast "
    "than either model achieves alone.",
    va="top", fontsize=8.5, color="#DDDDDD",
    transform=ax4.transAxes
)


# ─────────────────────────────────────────────────────────────────────
# SECTION 5 — When each model is most reliable
# ─────────────────────────────────────────────────────────────────────
ax5 = fig.add_axes([0.03, 0.10, 0.94, 0.10])
style_ax(ax5)

ax5.text(
    0.5, 0.97,
    "STEP 4 — When To Trust Each Model",
    ha="center", va="top", fontsize=15,
    fontweight="bold", color="white",
    transform=ax5.transAxes
)

use_cases = [
    {
        "title":   "🔮 Prophet — Use When:",
        "color":   "#2E75B6",
        "x":       0.01,
        "points": [
            "You need confidence intervals for budget planning",
            "Communicating forecasts to non-technical leadership",
            "Understanding seasonal demand patterns",
            "Market is behaving within historical norms",
        ],
    },
    {
        "title":   "📈 ARIMAX — Use When:",
        "color":   "#ED7D31",
        "x":       0.26,
        "points": [
            "Quantifying short-term price momentum",
            "Explaining price behavior to economists",
            "Validating whether a market move is sustainable",
            "Testing mean-reversion hypotheses",
        ],
    },
    {
        "title":   "🌲 LightGBM — Use When:",
        "color":   "#70AD47",
        "x":       0.51,
        "points": [
            "Multiple market signals are shifting simultaneously",
            "Crude oil and housing demand are both moving",
            "Short-term procurement decisions (1–4 weeks)",
            "Exploring which features drive a specific prediction",
        ],
    },
    {
        "title":   "🥇 Ensemble — Use When:",
        "color":   "gold",
        "x":       0.76,
        "points": [
            "Making final procurement recommendations",
            "Setting quarterly resin budget forecasts",
            "Evaluating supplier price increase validity",
            "Any high-stakes sourcing decision",
        ],
    },
]

for uc in use_cases:
    styled_box(ax5, uc["x"], 0.02, 0.23, 0.88,
               uc["color"], alpha=0.10,
               border_color=uc["color"],
               border_width=1.5)
    ax5.text(
        uc["x"] + 0.01, 0.88,
        uc["title"],
        va="top", fontsize=9,
        color=uc["color"], fontweight="bold",
        transform=ax5.transAxes
    )
    for i, pt in enumerate(uc["points"]):
        ax5.text(
            uc["x"] + 0.01, 0.74 - i * 0.175,
            f"• {pt}",
            va="top", fontsize=7.8,
            color="#DDDDDD",
            transform=ax5.transAxes
        )


# ─────────────────────────────────────────────────────────────────────
# SECTION 6 — Bottom conclusion banner
# ─────────────────────────────────────────────────────────────────────
ax6 = fig.add_axes([0.03, 0.02, 0.94, 0.065])
style_ax(ax6)

styled_box(ax6, 0.0, 0.0, 1.0, 1.0,
           "#C00000", alpha=0.15,
           border_color="gold", border_width=2)

ax6.text(
    0.5, 0.82,
    "🥇 FINAL DECISION: Ensemble (Prophet + LightGBM) "
    "as Primary Forecast  |  ARIMAX as Strategic Insight Tool",
    ha="center", va="top", fontsize=13,
    fontweight="bold", color="gold",
    transform=ax6.transAxes
)
ax6.text(
    0.5, 0.46,
    "The ensemble combines Prophet's strength in "
    "trend and seasonality decomposition with LightGBM's "
    "ability to capture non-linear feature interactions — "
    "producing\n"
    "1.50% MAPE and R²=0.786 on a 42-month test period "
    "that included an unprecedented geopolitical crude oil "
    "shock. ARIMAX's ARIMA(3,0,0) structure\n"
    "independently validates the 3-month momentum pattern "
    "identified by LightGBM, strengthening confidence "
    "in the ensemble's underlying market logic.",
    ha="center", va="top", fontsize=9,
    color="#DDDDDD",
    transform=ax6.transAxes
)


# ─────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────
output_path = "reports/model_selection_chart.png"
plt.savefig(output_path, dpi=180,
            bbox_inches="tight",
            facecolor="#0F1117")
plt.close()

print(f"✅ Model selection chart saved → {output_path}")
print("   Open reports/model_selection_chart.png to view")