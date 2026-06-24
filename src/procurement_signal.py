# src/procurement_signal.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("reports", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


# ────────────────────────────────────────────────────────────────────
# CORE PROCUREMENT SIGNAL CLASS
# ────────────────────────────────────────────────────────────────────
class ProcurementSignal:
    """
    Converts raw resin price forecasts into actionable
    procurement recommendations for supply chain leadership.

    Three threshold types:
    1. Direction threshold  — minimum % change to trigger signal
    2. Confidence threshold — maximum uncertainty to trust signal
    3. Momentum threshold   — RSI overbought/oversold confirmation

    All thresholds are tunable — the right values depend on
    ABG's procurement cycle, contract flexibility, and
    how much price movement actually affects their margins.
    """

    def __init__(
        self,
        direction_threshold=0.03,
        confidence_threshold=0.08,
        rsi_overbought=70,
        rsi_oversold=30,
        momentum_window=3
    ):
        """
        Parameters
        ----------
        direction_threshold : float
            Minimum forecast % change to trigger BUY/MONITOR.
            Default 0.03 = 3% price change required.
            Lower = more sensitive (signals more often)
            Higher = more conservative (signals rarely)

        confidence_threshold : float
            Maximum confidence interval width as % of forecast.
            If interval too wide → UNCERTAIN signal.
            Default 0.08 = suppress signal if interval > 8%.

        rsi_overbought : float
            RSI level above which resin prices are considered
            overbought → likely to pull back → good time to WAIT.
            Default 70 (standard commodity market level)

        rsi_oversold : float
            RSI level below which resin prices are considered
            oversold → likely to bounce → good time to BUY.
            Default 30 (standard commodity market level)

        momentum_window : int
            Number of recent months to assess price momentum.
            Default 3 = look at last 3 months of movement.
        """
        self.direction_threshold   = direction_threshold
        self.confidence_threshold  = confidence_threshold
        self.rsi_overbought        = rsi_overbought
        self.rsi_oversold          = rsi_oversold
        self.momentum_window       = momentum_window

    # ── SIGNAL GENERATION ────────────────────────────────────────────
    def generate_signal(
        self,
        current_price,
        forecast,
        lower_bound,
        upper_bound,
        current_rsi=None,
        recent_prices=None,
        forecast_date=None
    ):
        """
        Generates a procurement recommendation from forecast data.

        Returns a detailed signal dictionary with:
        - Primary signal (BUY NOW / HOLD / MONITOR / UNCERTAIN)
        - Confidence level (HIGH / MEDIUM / LOW)
        - Recommended action
        - Business rationale
        - Risk assessment
        """
        signal = {
            "date":          forecast_date,
            "current_price": current_price,
            "forecast":      forecast,
            "lower_bound":   lower_bound,
            "upper_bound":   upper_bound,
        }

        # ── Check 1: Confidence interval width ───────────────────────
        # If forecast uncertainty is too high → UNCERTAIN
        interval_width = (upper_bound - lower_bound) / forecast
        signal["interval_width_pct"] = interval_width * 100

        if interval_width > self.confidence_threshold:
            signal.update({
                "primary_signal":  "UNCERTAIN",
                "signal_color":    "#888888",
                "confidence":      "LOW",
                "pct_change":      None,
                "action":          "Wait for clearer market signal",
                "rationale":       (
                    f"Forecast confidence interval is "
                    f"{interval_width*100:.1f}% of forecast value — "
                    f"above {self.confidence_threshold*100:.0f}% "
                    f"uncertainty threshold. Insufficient confidence "
                    f"to justify procurement action."
                ),
                "risk":            "HIGH UNCERTAINTY",
            })
            return signal

        # ── Check 2: Forecast direction ───────────────────────────────
        pct_change = (forecast - current_price) / current_price
        signal["pct_change"] = pct_change

        # ── Check 3: RSI confirmation ─────────────────────────────────
        rsi_signal   = None
        rsi_context  = ""

        if current_rsi is not None:
            if current_rsi > self.rsi_overbought:
                rsi_signal  = "OVERBOUGHT"
                rsi_context = (
                    f" RSI={current_rsi:.1f} indicates overbought "
                    f"conditions — price pullback likely."
                )
            elif current_rsi < self.rsi_oversold:
                rsi_signal  = "OVERSOLD"
                rsi_context = (
                    f" RSI={current_rsi:.1f} indicates oversold "
                    f"conditions — price bounce likely."
                )
            else:
                rsi_context = (
                    f" RSI={current_rsi:.1f} — neutral momentum."
                )

        signal["rsi_signal"]  = rsi_signal
        signal["rsi_context"] = rsi_context

        # ── Check 4: Recent momentum ──────────────────────────────────
        momentum_context = ""
        if recent_prices is not None and \
                len(recent_prices) >= self.momentum_window:
            recent     = recent_prices[-self.momentum_window:]
            momentum   = (recent[-1] - recent[0]) / recent[0]
            signal["recent_momentum"] = momentum
            if momentum > 0.02:
                momentum_context = (
                    f" {self.momentum_window}-month momentum: "
                    f"+{momentum*100:.1f}% (rising)."
                )
            elif momentum < -0.02:
                momentum_context = (
                    f" {self.momentum_window}-month momentum: "
                    f"{momentum*100:.1f}% (falling)."
                )
            else:
                momentum_context = (
                    f" {self.momentum_window}-month momentum: "
                    f"flat ({momentum*100:.1f}%)."
                )

        # ── Generate primary signal ───────────────────────────────────
        if pct_change > self.direction_threshold:
            # Prices forecast to RISE significantly

            if rsi_signal == "OVERBOUGHT":
                # RSI says overbought — conflicting signal
                primary    = "HOLD"
                color      = "#FFD700"
                confidence = "MEDIUM"
                action     = (
                    "Consider partial forward purchase — "
                    "lock in 50% of Q needs now"
                )
                rationale  = (
                    f"Forecast suggests +{pct_change*100:.1f}% "
                    f"price increase but RSI={current_rsi:.0f} "
                    f"signals overbought conditions. Mixed signal — "
                    f"partial purchase recommended."
                    + momentum_context
                )
                risk       = "MEDIUM — conflicting signals"

            else:
                primary    = "BUY NOW"
                color      = "#C00000"
                confidence = "HIGH"
                action     = (
                    "Lock in forward contracts or "
                    "accelerate near-term purchases"
                )
                rationale  = (
                    f"Ensemble forecast projects "
                    f"+{pct_change*100:.1f}% price increase "
                    f"({current_price:.1f} → {forecast:.1f} PPI). "
                    f"Purchasing now avoids anticipated cost increase."
                    + rsi_context + momentum_context
                )
                risk       = "LOW — strong upward signal"

        elif pct_change < -self.direction_threshold:
            # Prices forecast to FALL significantly

            if rsi_signal == "OVERSOLD":
                # RSI says oversold — price may bounce before falling
                primary    = "MONITOR"
                color      = "#2E75B6"
                confidence = "MEDIUM"
                action     = (
                    "Delay non-urgent purchases by 2–4 weeks "
                    "and monitor for bounce"
                )
                rationale  = (
                    f"Forecast suggests "
                    f"{pct_change*100:.1f}% price decline but "
                    f"RSI={current_rsi:.0f} signals oversold — "
                    f"short-term bounce possible before decline. "
                    f"Wait for confirmation."
                    + momentum_context
                )
                risk       = "MEDIUM — possible short bounce first"

            else:
                primary    = "MONITOR"
                color      = "#2E75B6"
                confidence = "HIGH"
                action     = (
                    "Delay purchases where operationally possible — "
                    "buy at lower prices in coming weeks"
                )
                rationale  = (
                    f"Ensemble forecast projects "
                    f"{pct_change*100:.1f}% price decrease "
                    f"({current_price:.1f} → {forecast:.1f} PPI). "
                    f"Delaying non-critical purchases captures "
                    f"anticipated cost saving."
                    + rsi_context + momentum_context
                )
                risk       = "LOW — strong downward signal"

        else:
            # Prices forecast to be STABLE
            primary    = "HOLD"
            color      = "#FFD700"
            confidence = "HIGH"
            action     = (
                "Maintain current purchasing schedule — "
                "no urgent action required"
            )
            rationale  = (
                f"Ensemble forecast projects stable prices "
                f"({pct_change*100:+.1f}% change) within the "
                f"±{self.direction_threshold*100:.0f}% "
                f"action threshold. No procurement urgency."
                + rsi_context + momentum_context
            )
            risk       = "LOW — stable market conditions"

        signal.update({
            "primary_signal": primary,
            "signal_color":   color,
            "confidence":     confidence,
            "action":         action,
            "rationale":      rationale,
            "risk":           risk,
        })
        return signal

    # ── BATCH SIGNAL GENERATION ───────────────────────────────────────
    def generate_signals_from_forecast(
        self,
        forecast_df,
        historical_df
    ):
        """
        Generates signals for all forecast periods.

        Parameters
        ----------
        forecast_df : DataFrame
            Must have columns: date, forecast,
            lower_bound, upper_bound
        historical_df : DataFrame
            Must have columns: resin_ppi, resin_rsi (index=date)

        Returns
        -------
        List of signal dictionaries
        """
        signals    = []
        current_px = historical_df["resin_ppi"].dropna().iloc[-1]
        current_rsi = (
            historical_df["resin_rsi"].dropna().iloc[-1]
            if "resin_rsi" in historical_df.columns
            else None
        )
        recent_prices = (
            historical_df["resin_ppi"]
            .dropna()
            .iloc[-self.momentum_window:]
            .values
        )

        for _, row in forecast_df.iterrows():
            sig = self.generate_signal(
                current_price  = current_px,
                forecast       = row["forecast"],
                lower_bound    = row["lower_bound"],
                upper_bound    = row["upper_bound"],
                current_rsi    = current_rsi,
                recent_prices  = recent_prices,
                forecast_date  = row["date"]
            )
            signals.append(sig)

        return signals

    # ── THRESHOLD SENSITIVITY ANALYSIS ───────────────────────────────
    def threshold_sensitivity(
        self,
        forecast_df,
        historical_df,
        thresholds=[0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    ):
        """
        Tests multiple threshold values and shows how signals change.
        Helps procurement team choose the right sensitivity level.

        Lower threshold = more sensitive = more BUY/MONITOR signals
        Higher threshold = more conservative = more HOLD signals
        """
        print("\n📊 THRESHOLD SENSITIVITY ANALYSIS")
        print("=" * 65)
        print(
            "How procurement signals change as threshold varies:\n"
        )
        print(
            f"   {'Threshold':>10} "
            f"{'BUY NOW':>10} "
            f"{'HOLD':>8} "
            f"{'MONITOR':>10} "
            f"{'UNCERTAIN':>12} "
            f"{'Interpretation':>20}"
        )
        print("   " + "-" * 75)

        results = []
        for thresh in thresholds:
            test_signal = ProcurementSignal(
                direction_threshold=thresh,
                confidence_threshold=self.confidence_threshold,
                rsi_overbought=self.rsi_overbought,
                rsi_oversold=self.rsi_oversold
            )
            sigs = test_signal.generate_signals_from_forecast(
                forecast_df, historical_df
            )

            counts = {
                "BUY NOW":   0,
                "HOLD":      0,
                "MONITOR":   0,
                "UNCERTAIN": 0
            }
            for s in sigs:
                counts[s["primary_signal"]] = (
                    counts.get(s["primary_signal"], 0) + 1
                )

            total = len(sigs)
            if counts["BUY NOW"] > total * 0.5:
                interp = "Very Aggressive"
            elif counts["HOLD"] > total * 0.7:
                interp = "Very Conservative"
            elif counts["HOLD"] > total * 0.4:
                interp = "Balanced"
            else:
                interp = "Active"

            results.append({
                "threshold": thresh,
                **counts,
                "interpretation": interp
            })

            print(
                f"   {thresh*100:>9.0f}% "
                f"{counts['BUY NOW']:>10} "
                f"{counts['HOLD']:>8} "
                f"{counts['MONITOR']:>10} "
                f"{counts['UNCERTAIN']:>12} "
                f"{interp:>20}"
            )

        print(
            "\n   💡 Recommended threshold: 3% for standard "
            "procurement cycles"
        )
        print(
            "      Adjust based on ABG's contract flexibility "
            "and margin sensitivity"
        )
        return results

    # ── PRINT SIGNALS ─────────────────────────────────────────────────
    def print_signals(self, signals):
        print("\n" + "=" * 65)
        print("🏭 PROCUREMENT SIGNAL REPORT")
        print("=" * 65)
        print(
            f"   Direction threshold:  "
            f"±{self.direction_threshold*100:.0f}%"
        )
        print(
            f"   Confidence threshold: "
            f"{self.confidence_threshold*100:.0f}% max interval"
        )
        print(
            f"   RSI overbought:       {self.rsi_overbought}"
        )
        print(
            f"   RSI oversold:         {self.rsi_oversold}\n"
        )

        icons = {
            "BUY NOW":   "🔴",
            "HOLD":      "🟡",
            "MONITOR":   "🟢",
            "UNCERTAIN": "⚪"
        }

        for sig in signals:
            icon = icons.get(sig["primary_signal"], "⚪")
            print(
                f"   {icon} {sig['date']}  "
                f"{sig['primary_signal']:<12} "
                f"[{sig['confidence']} CONFIDENCE]"
            )
            print(
                f"      Forecast: {sig['forecast']:.1f} PPI  "
                f"(current: {sig['current_price']:.1f})  "
                f"Change: {(sig.get('pct_change', 0) or 0)*100:+.1f}%"
            )
            print(
                f"      Action:   {sig['action']}"
            )
            print(
                f"      Rationale: {sig['rationale'][:90]}..."
                if len(sig['rationale']) > 90
                else f"      Rationale: {sig['rationale']}"
            )
            print(f"      Risk: {sig['risk']}\n")


# ────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ────────────────────────────────────────────────────────────────────
def plot_procurement_dashboard(
    historical_df,
    forecast_df,
    signals,
    signal_obj,
    output_path="reports/procurement_signal_dashboard.png"
):
    """
    Creates a complete procurement signal visualization with:
    1. Historical resin PPI + forecast with signal overlay
    2. RSI indicator with overbought/oversold bands
    3. Forecast confidence interval detail
    4. Signal summary panel
    5. Threshold sensitivity chart
    6. Business action summary
    """
    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor("#0F1117")

    fig.suptitle(
        "Resin Price Procurement Signal Dashboard",
        fontsize=22, fontweight="bold",
        color="white", y=0.98
    )
    fig.text(
        0.5, 0.962,
        f"Direction Threshold: "
        f"±{signal_obj.direction_threshold*100:.0f}%  |  "
        f"Confidence Threshold: "
        f"{signal_obj.confidence_threshold*100:.0f}%  |  "
        f"RSI Bands: {signal_obj.rsi_oversold}/"
        f"{signal_obj.rsi_overbought}  |  "
        f"Ensemble Model: MAPE 1.50%  R²=0.786",
        ha="center", fontsize=11,
        color="#AAAAAA", style="italic"
    )

    def style_ax(ax, title="", ylabel=""):
        ax.set_facecolor("#1A1D2E")
        if title:
            ax.set_title(
                title, fontsize=12,
                fontweight="bold",
                color="white", pad=10
            )
        if ylabel:
            ax.set_ylabel(
                ylabel, fontsize=10,
                color="#AAAAAA"
            )
        ax.tick_params(colors="#AAAAAA", labelsize=9)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#333355")
        ax.yaxis.grid(
            True, color="#333355",
            linestyle="--", alpha=0.4
        )
        ax.set_axisbelow(True)

    # ── Panel 1: Historical PPI + Forecast + Signals ──────────────
    ax1 = fig.add_axes([0.06, 0.75, 0.88, 0.17])

    hist = historical_df["resin_ppi"].dropna()
    ax1.plot(
        hist.index, hist.values,
        color="#2E75B6", lw=1.8,
        label="Historical Resin PPI", zorder=3
    )

    # Forecast line
    fc_dates = pd.to_datetime(
        [str(s["date"]) for s in signals]
    )
    fc_vals  = [s["forecast"] for s in signals]
    lb_vals  = [s["lower_bound"] for s in signals]
    ub_vals  = [s["upper_bound"] for s in signals]

    ax1.plot(
        fc_dates, fc_vals,
        color="darkorange", lw=2,
        linestyle="--",
        label="Ensemble Forecast", zorder=4
    )
    ax1.fill_between(
        fc_dates, lb_vals, ub_vals,
        alpha=0.2, color="darkorange",
        label="95% Confidence Interval"
    )

    # Signal markers on forecast
    signal_icons = {
        "BUY NOW":   ("^", "#C00000", 14),
        "HOLD":      ("o", "#FFD700", 10),
        "MONITOR":   ("v", "#2E75B6", 14),
        "UNCERTAIN": ("D", "#888888", 10),
    }

    for sig, date, val in zip(
            signals, fc_dates, fc_vals):
        marker, color, size = signal_icons.get(
            sig["primary_signal"],
            ("●", "#888888", 10)
        )
        ax1.scatter(
            date, val,
            color=color, s=size**2,
            zorder=5, marker=marker
        )

    # Only show last 3 years of history for clarity
    cutoff_display = hist.index[-36]
    ax1.set_xlim(cutoff_display, fc_dates[-1])

    style_ax(
        ax1,
        "① Resin PPI History + Ensemble Forecast "
        "with Procurement Signals",
        "PPI Index"
    )

    # Custom legend
    legend_elements = [
        plt.Line2D([0], [0], color="#2E75B6",
                   lw=2, label="Historical PPI"),
        plt.Line2D([0], [0], color="darkorange",
                   lw=2, ls="--", label="Forecast"),
        plt.Line2D([0], [0], color="#C00000",
                   marker="^", lw=0, markersize=10,
                   label="BUY NOW signal"),
        plt.Line2D([0], [0], color="#FFD700",
                   marker="o", lw=0, markersize=8,
                   label="HOLD signal"),
        plt.Line2D([0], [0], color="#2E75B6",
                   marker="v", lw=0, markersize=10,
                   label="MONITOR signal"),
    ]
    ax1.legend(
        handles=legend_elements,
        fontsize=8, facecolor="#1A1D2E",
        labelcolor="white", framealpha=0.8,
        loc="upper left"
    )

    # ── Panel 2: RSI Indicator ─────────────────────────────────────
    ax2 = fig.add_axes([0.06, 0.59, 0.88, 0.12])

    if "resin_rsi" in historical_df.columns:
        rsi = historical_df["resin_rsi"].dropna()
        rsi = rsi[rsi.index >= cutoff_display]

        ax2.plot(
            rsi.index, rsi.values,
            color="#9B59B6", lw=1.5,
            label="Resin RSI"
        )
        ax2.axhline(
            signal_obj.rsi_overbought,
            color="#C00000", lw=1.2,
            linestyle="--",
            label=f"Overbought ({signal_obj.rsi_overbought})"
        )
        ax2.axhline(
            signal_obj.rsi_oversold,
            color="#2E75B6", lw=1.2,
            linestyle="--",
            label=f"Oversold ({signal_obj.rsi_oversold})"
        )
        ax2.axhline(
            50, color="#555577",
            lw=0.8, linestyle=":"
        )
        ax2.fill_between(
            rsi.index, signal_obj.rsi_overbought, 100,
            alpha=0.1, color="#C00000"
        )
        ax2.fill_between(
            rsi.index, 0, signal_obj.rsi_oversold,
            alpha=0.1, color="#2E75B6"
        )
        ax2.set_xlim(cutoff_display, fc_dates[-1])
        ax2.set_ylim(0, 100)

    style_ax(
        ax2,
        "② Resin PPI Relative Strength Index (RSI)  "
        "— Overbought/Oversold Signal",
        "RSI"
    )
    ax2.legend(
        fontsize=8, facecolor="#1A1D2E",
        labelcolor="white", framealpha=0.8,
        loc="upper left"
    )

    # ── Panel 3: Forecast detail with confidence ───────────────────
    ax3 = fig.add_axes([0.06, 0.44, 0.55, 0.12])

    bar_colors_map = {
        "BUY NOW":   "#C00000",
        "HOLD":      "#FFD700",
        "MONITOR":   "#2E75B6",
        "UNCERTAIN": "#888888"
    }
    bar_colors_list = [
        bar_colors_map.get(s["primary_signal"], "#888888")
        for s in signals
    ]

    bars = ax3.bar(
        range(len(signals)),
        fc_vals,
        color=bar_colors_list,
        edgecolor="#0F1117",
        linewidth=1, zorder=3
    )

    # Error bars for confidence interval
    errors_lo = [
        f - l for f, l in zip(fc_vals, lb_vals)
    ]
    errors_hi = [
        u - f for f, u in zip(fc_vals, ub_vals)
    ]
    ax3.errorbar(
        range(len(signals)),
        fc_vals,
        yerr=[errors_lo, errors_hi],
        fmt="none", color="white",
        capsize=4, linewidth=1.5,
        zorder=4
    )

    for i, (sig, val) in enumerate(
            zip(signals, fc_vals)):
        ax3.text(
            i, val + errors_hi[i] + 2,
            f"{val:.0f}",
            ha="center", va="bottom",
            fontsize=7.5, color="white",
            fontweight="bold"
        )

    ax3.axhline(
        historical_df["resin_ppi"].dropna().iloc[-1],
        color="white", lw=1.5,
        linestyle=":", alpha=0.7,
        label="Current Price"
    )

    month_labels = [
        str(s["date"])[:7] for s in signals
    ]
    ax3.set_xticks(range(len(signals)))
    ax3.set_xticklabels(
        month_labels, rotation=45,
        ha="right", fontsize=8, color="#AAAAAA"
    )

    style_ax(
        ax3,
        "③ 12-Month Forecast with Confidence Intervals",
        "Resin PPI"
    )
    ax3.legend(
        fontsize=8, facecolor="#1A1D2E",
        labelcolor="white", framealpha=0.8
    )

    # ── Panel 4: Signal summary donut chart ───────────────────────
    ax4 = fig.add_axes([0.65, 0.44, 0.30, 0.12])
    ax4.set_facecolor("#1A1D2E")

    signal_counts = {}
    for sig in signals:
        ps = sig["primary_signal"]
        signal_counts[ps] = signal_counts.get(ps, 0) + 1

    donut_labels = list(signal_counts.keys())
    donut_vals   = list(signal_counts.values())
    donut_colors = [
        bar_colors_map.get(l, "#888888")
        for l in donut_labels
    ]

    wedges, texts, autotexts = ax4.pie(
        donut_vals,
        labels=donut_labels,
        colors=donut_colors,
        autopct="%1.0f%%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(
            width=0.5, edgecolor="#0F1117",
            linewidth=2
        )
    )
    for text in texts:
        text.set_color("white")
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(8)
        autotext.set_fontweight("bold")

    ax4.set_title(
        "④ Signal Distribution\n(12-Month Horizon)",
        fontsize=11, fontweight="bold",
        color="white", pad=10
    )

    # ── Panel 5: Threshold sensitivity ────────────────────────────
    ax5 = fig.add_axes([0.06, 0.28, 0.55, 0.12])

    thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    buy_counts     = []
    hold_counts    = []
    monitor_counts = []

    for thresh in thresholds:
        test_s = ProcurementSignal(
            direction_threshold=thresh,
            confidence_threshold=signal_obj.confidence_threshold
        )
        sigs = test_s.generate_signals_from_forecast(
            forecast_df, historical_df
        )
        counts = {
            "BUY NOW": 0, "HOLD": 0,
            "MONITOR": 0, "UNCERTAIN": 0
        }
        for s in sigs:
            counts[s["primary_signal"]] = (
                counts.get(s["primary_signal"], 0) + 1
            )
        buy_counts.append(counts["BUY NOW"])
        hold_counts.append(counts["HOLD"])
        monitor_counts.append(counts["MONITOR"])

    x_thresh = np.arange(len(thresholds))
    w_bar    = 0.25

    ax5.bar(
        x_thresh - w_bar, buy_counts,
        width=w_bar, color="#C00000",
        label="BUY NOW", edgecolor="#0F1117"
    )
    ax5.bar(
        x_thresh, hold_counts,
        width=w_bar, color="#FFD700",
        label="HOLD", edgecolor="#0F1117"
    )
    ax5.bar(
        x_thresh + w_bar, monitor_counts,
        width=w_bar, color="#2E75B6",
        label="MONITOR", edgecolor="#0F1117"
    )

    # Highlight current threshold
    current_idx = thresholds.index(
        signal_obj.direction_threshold
    ) if signal_obj.direction_threshold in thresholds else 2
    ax5.axvline(
        current_idx, color="white",
        lw=2, linestyle=":",
        alpha=0.8, label="Current threshold"
    )

    ax5.set_xticks(x_thresh)
    ax5.set_xticklabels(
        [f"{t*100:.0f}%" for t in thresholds],
        color="#AAAAAA", fontsize=9
    )
    style_ax(
        ax5,
        "⑤ Threshold Sensitivity  —  "
        "How Signal Counts Change With Threshold",
        "Number of Signals"
    )
    ax5.legend(
        fontsize=8, facecolor="#1A1D2E",
        labelcolor="white", framealpha=0.8
    )
    ax5.text(
        current_idx, ax5.get_ylim()[1] * 0.85,
        " ← Current\n   threshold",
        color="white", fontsize=7.5,
        style="italic"
    )

    # ── Panel 6: Risk assessment gauge ────────────────────────────
    ax6 = fig.add_axes([0.65, 0.28, 0.30, 0.12])
    ax6.set_facecolor("#1A1D2E")
    ax6.axis("off")
    ax6.set_title(
        "⑥ Risk Assessment",
        fontsize=11, fontweight="bold",
        color="white", pad=10
    )

    risk_data = []
    for sig in signals[:6]:
        risk_level = (
            "HIGH"   if "HIGH" in sig["risk"]
            else "MEDIUM" if "MEDIUM" in sig["risk"]
            else "LOW"
        )
        risk_data.append(risk_level)

    risk_colors = {
        "HIGH": "#C00000",
        "MEDIUM": "#FFD700",
        "LOW": "#70AD47"
    }
    risk_counts = {
        "HIGH":   risk_data.count("HIGH"),
        "MEDIUM": risk_data.count("MEDIUM"),
        "LOW":    risk_data.count("LOW")
    }

    y_pos = 0.85
    for level, count in risk_counts.items():
        color = risk_colors[level]
        bar_w = count / len(risk_data)
        ax6.add_patch(mpatches.FancyBboxPatch(
            (0.05, y_pos - 0.12), bar_w * 0.9, 0.10,
            boxstyle="round,pad=0.01",
            facecolor=color, alpha=0.7,
            edgecolor=color, linewidth=1.5,
            transform=ax6.transAxes
        ))
        ax6.text(
            0.05 + bar_w * 0.45, y_pos - 0.07,
            f"{level}: {count} months",
            ha="center", va="center",
            fontsize=9, color="white",
            fontweight="bold",
            transform=ax6.transAxes
        )
        y_pos -= 0.20

    # ── Panel 7: Detailed signal cards ────────────────────────────
    ax7 = fig.add_axes([0.03, 0.10, 0.94, 0.155])
    ax7.set_facecolor("#0F1117")
    ax7.axis("off")
    ax7.set_title(
        "⑦ Detailed Procurement Recommendations  "
        "(Next 6 Months)",
        fontsize=12, fontweight="bold",
        color="white", pad=10, loc="left"
    )

    card_w = 0.155
    for i, sig in enumerate(signals[:6]):
        cx   = 0.01 + i * 0.163
        col  = bar_colors_map.get(
            sig["primary_signal"], "#888888"
        )

        # Card background
        ax7.add_patch(mpatches.FancyBboxPatch(
            (cx, 0.02), card_w, 0.88,
            boxstyle="round,pad=0.015",
            facecolor=col, alpha=0.12,
            edgecolor=col, linewidth=2,
            transform=ax7.transAxes
        ))

        # Date
        ax7.text(
            cx + card_w/2, 0.88,
            str(sig["date"])[:7],
            ha="center", va="top",
            fontsize=9, color="#AAAAAA",
            transform=ax7.transAxes
        )

        # Signal
        ax7.text(
            cx + card_w/2, 0.76,
            sig["primary_signal"],
            ha="center", va="top",
            fontsize=11, color=col,
            fontweight="bold",
            transform=ax7.transAxes
        )

        # Confidence badge
        conf_col = (
            "#70AD47" if sig["confidence"] == "HIGH"
            else "#FFD700" if sig["confidence"] == "MEDIUM"
            else "#888888"
        )
        ax7.text(
            cx + card_w/2, 0.63,
            f"{sig['confidence']} CONFIDENCE",
            ha="center", va="top",
            fontsize=7.5, color=conf_col,
            fontweight="bold",
            transform=ax7.transAxes
        )

        # Forecast vs current
        pct = sig.get("pct_change") or 0
        ax7.text(
            cx + card_w/2, 0.52,
            f"Forecast: {sig['forecast']:.0f}",
            ha="center", va="top",
            fontsize=8.5, color="white",
            transform=ax7.transAxes
        )
        ax7.text(
            cx + card_w/2, 0.43,
            f"Current:  {sig['current_price']:.0f}",
            ha="center", va="top",
            fontsize=8.5, color="#AAAAAA",
            transform=ax7.transAxes
        )
        ax7.text(
            cx + card_w/2, 0.34,
            f"Change:   {pct:+.1%}",
            ha="center", va="top",
            fontsize=8.5,
            color=("#C00000" if pct > 0
                   else "#70AD47" if pct < 0
                   else "white"),
            fontweight="bold",
            transform=ax7.transAxes
        )

        # Action (truncated)
        action = sig["action"]
        words  = action.split()
        line1  = " ".join(words[:5])
        line2  = " ".join(words[5:10])
        line3  = " ".join(words[10:])
        ax7.text(
            cx + card_w/2, 0.24,
            line1,
            ha="center", va="top",
            fontsize=7, color="#DDDDDD",
            transform=ax7.transAxes
        )
        if line2:
            ax7.text(
                cx + card_w/2, 0.17,
                line2,
                ha="center", va="top",
                fontsize=7, color="#DDDDDD",
                transform=ax7.transAxes
            )
        if line3:
            ax7.text(
                cx + card_w/2, 0.10,
                line3,
                ha="center", va="top",
                fontsize=7, color="#DDDDDD",
                transform=ax7.transAxes
            )

    # ── Panel 8: Bottom summary banner ────────────────────────────
    ax8 = fig.add_axes([0.03, 0.02, 0.94, 0.065])
    ax8.set_facecolor("#1A2A1A")
    ax8.axis("off")

    ax8.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.01",
        facecolor="#1A2A1A", alpha=1.0,
        edgecolor="#70AD47", linewidth=2,
        transform=ax8.transAxes
    ))

    ax8.text(
        0.5, 0.88,
        "📋 METHODOLOGY NOTE",
        ha="center", va="top",
        fontsize=11, color="#70AD47",
        fontweight="bold",
        transform=ax8.transAxes
    )
    ax8.text(
        0.5, 0.62,
        "Procurement signals are generated by the "
        "Ensemble (Prophet + LightGBM) model achieving "
        "MAPE=1.50% and R²=0.786 on the 2023–2026 test period.  "
        "Signals incorporate three layers of analysis: "
        "(1) forecast direction vs configurable threshold, "
        "(2) confidence interval width check, "
        "(3) RSI overbought/oversold confirmation.  "
        "Thresholds are tunable based on ABG's procurement "
        "cycle and contract flexibility.",
        ha="center", va="top",
        fontsize=8.5, color="#DDDDDD",
        transform=ax8.transAxes
    )

    plt.savefig(
        output_path, dpi=160,
        bbox_inches="tight",
        facecolor="#0F1117"
    )
    plt.close()
    print(f"   💾 Dashboard saved → {output_path}")


# ────────────────────────────────────────────────────────────────────
# RUN
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Load forecast output
    print("📂 Loading forecast and historical data...")
    forecast_df = pd.read_csv(
        "data/processed/forecast_output.csv"
    )
    historical_df = pd.read_csv(
        "data/processed/clean_master.csv",
        index_col="date", parse_dates=True
    )

    print(
        f"   ✅ Forecast: {len(forecast_df)} periods"
    )
    print(
        f"   ✅ Historical: "
        f"{len(historical_df)} months\n"
    )

    # ── Initialize signal generator ──────────────────────────────
    # These are the default thresholds
    # Change them to see how signals respond
    signal_generator = ProcurementSignal(
        direction_threshold  = 0.03,  # 3% change triggers signal
        confidence_threshold = 0.08,  # 8% interval width = uncertain
        rsi_overbought       = 70,    # RSI above 70 = overbought
        rsi_oversold         = 30,    # RSI below 30 = oversold
        momentum_window      = 3      # 3-month momentum lookback
    )

    # ── Generate signals for all forecast periods ─────────────────
    signals = signal_generator.generate_signals_from_forecast(
        forecast_df, historical_df
    )

    # ── Print signals ─────────────────────────────────────────────
    signal_generator.print_signals(signals)

    # ── Threshold sensitivity analysis ───────────────────────────
    signal_generator.threshold_sensitivity(
        forecast_df, historical_df,
        thresholds=[0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    )

    # ── Build full procurement dashboard ─────────────────────────
    print("\n📊 Building procurement signal dashboard...")
    plot_procurement_dashboard(
        historical_df  = historical_df,
        forecast_df    = forecast_df,
        signals        = signals,
        signal_obj     = signal_generator
    )

    # ── Flag unreliable forecast periods ─────────────────────────
    print("\n⚠️  FORECAST RELIABILITY FLAGS:")
    print("=" * 65)
    reliable_cutoff = pd.Timestamp("2026-09-01")

    for sig in signals:
        date = pd.Timestamp(str(sig["date"]))
        if date >= reliable_cutoff:
            print(
                f"   ⚠️  {sig['date']} — HIGH UNCERTAINTY: "
                f"Forecast driven by 2026 shock dummy "
                f"variable extrapolation. "
                f"Treat as scenario analysis only."
            )
        else:
            print(
                f"   ✅ {sig['date']} — RELIABLE: "
                f"Within model's confident forecast horizon"
            )

    # ── Save signals to CSV ───────────────────────────────────────
    signal_records = []
    for sig in signals:
        signal_records.append({
            "date":           sig["date"],
            "signal":         sig["primary_signal"],
            "confidence":     sig["confidence"],
            "current_price":  sig["current_price"],
            "forecast":       sig["forecast"],
            "lower_bound":    sig["lower_bound"],
            "upper_bound":    sig["upper_bound"],
            "pct_change":     sig.get("pct_change"),
            "action":         sig["action"],
            "risk":           sig["risk"],
        })

    signals_df = pd.DataFrame(signal_records)
    signals_df.to_csv(
        "data/processed/procurement_signals.csv",
        index=False
    )
    print(
        "   💾 Signals saved → "
        "data/processed/procurement_signals.csv"
    )

    print("\n✅ Procurement signal analysis complete!")
    print("   Outputs:")
    print(
        "   → reports/procurement_signal_dashboard.png"
    )
    print(
        "   → data/processed/procurement_signals.csv"
    )