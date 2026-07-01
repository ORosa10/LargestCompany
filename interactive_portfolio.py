"""Interactive per-leg option editor for Phase 5."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


DEFINE_MODES = ["Boundary", "Strike"]
BOUNDARY_TYPES = ["Win boundary", "Loss boundary"]
OPTION_TYPES = ["Call", "Put"]
POSITIONS = ["Long", "Short"]
CARDS_PER_ROW = 4


def monotone_probability_curve(curve: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ordered = curve.sort_values("market_cap_to_current")
    ratios = ordered["market_cap_to_current"].to_numpy(dtype=float)
    win_probabilities = np.maximum.accumulate(
        ordered["win_probability"].to_numpy(dtype=float)
    )
    return ratios, np.clip(win_probabilities, 0.0, 1.0)


def confidence_at_strike(
    curve: pd.DataFrame,
    strike: float,
    *,
    boundary_type: str,
    normalized_spot: float = 100.0,
) -> float:
    ratios, win_probabilities = monotone_probability_curve(curve)
    ratio = float(strike) / float(normalized_spot)
    win_probability = float(
        np.interp(
            ratio,
            ratios,
            win_probabilities,
            left=win_probabilities[0],
            right=win_probabilities[-1],
        )
    )
    return win_probability if boundary_type == "Win boundary" else 1.0 - win_probability


def strike_at_confidence(
    curve: pd.DataFrame,
    confidence: float,
    *,
    boundary_type: str,
    normalized_spot: float = 100.0,
) -> float:
    ordered = curve.sort_values("market_cap_to_current").copy()
    ordered["monotone_win_probability"] = np.maximum.accumulate(
        ordered["win_probability"].to_numpy(dtype=float)
    )
    if boundary_type == "Win boundary":
        candidates = ordered[ordered["monotone_win_probability"] >= confidence]
        row = ordered.iloc[-1] if candidates.empty else candidates.iloc[0]
    else:
        ordered["monotone_loss_probability"] = (
            1.0 - ordered["monotone_win_probability"]
        )
        candidates = ordered[ordered["monotone_loss_probability"] >= confidence]
        row = ordered.iloc[0] if candidates.empty else candidates.iloc[-1]
    return float(normalized_spot) * float(row["market_cap_to_current"])


def default_interactive_rows(ticker: str, pricing_iv: float) -> list[dict]:
    return [
        {
            "id": 1,
            "active": True,
            "ticker": ticker,
            "option_type": "Put",
            "position": "Long",
            "quantity": 0.10,
            "define_by": "Boundary",
            "boundary_type": "Loss boundary",
            "confidence_pct": 80.0,
            "strike": 80.0,
            "pricing_iv": pricing_iv,
        },
        {
            "id": 2,
            "active": True,
            "ticker": ticker,
            "option_type": "Call",
            "position": "Short",
            "quantity": 0.10,
            "define_by": "Boundary",
            "boundary_type": "Win boundary",
            "confidence_pct": 80.0,
            "strike": 120.0,
            "pricing_iv": pricing_iv,
        },
    ]


def optimized_legs_to_interactive_rows(
    legs: pd.DataFrame, fallback_iv: float
) -> list[dict]:
    rows = []
    for position, (_, leg) in enumerate(legs.iterrows(), start=1):
        rows.append(
            {
                "id": position,
                "active": True,
                "ticker": str(leg["Ticker"]),
                "option_type": str(leg["Option type"]),
                "position": str(leg["Position"]),
                "quantity": float(leg["Quantity"]),
                "define_by": "Strike",
                "boundary_type": (
                    "Win boundary"
                    if str(leg["Option type"]) == "Call"
                    else "Loss boundary"
                ),
                "confidence_pct": 80.0,
                "strike": float(leg["Strike"]),
                "pricing_iv": float(leg.get("Model IV", fallback_iv)),
            }
        )
    return rows


def _render_leg_card(
    *,
    row: dict,
    row_number: int,
    tickers: list[str],
    curves: dict[str, pd.DataFrame],
    iv_by_ticker: pd.Series,
    normalized_spot: float,
) -> tuple[dict, bool]:
    """Render one narrow card with controls stacked vertically."""
    row_id = int(row["id"])
    status = "Active" if row["active"] else "Inactive"
    st.markdown(f"#### Leg {row_number}")
    st.caption(
        f"{status}: {row['position']} {row['ticker']} {row['option_type']}"
    )

    active = st.checkbox(
        "Use this leg",
        value=bool(row["active"]),
        key=f"leg_active_{row_id}",
    )
    ticker = st.selectbox(
        "Ticker",
        tickers,
        index=tickers.index(row["ticker"]) if row["ticker"] in tickers else 0,
        key=f"leg_ticker_{row_id}",
    )
    option_type = st.selectbox(
        "Option type",
        OPTION_TYPES,
        index=OPTION_TYPES.index(row["option_type"]),
        key=f"leg_type_{row_id}",
    )
    position = st.selectbox(
        "Trade direction",
        POSITIONS,
        index=POSITIONS.index(row["position"]),
        key=f"leg_position_{row_id}",
    )
    quantity = st.number_input(
        "Contracts",
        min_value=0.0,
        value=float(row["quantity"]),
        step=0.025,
        format="%.3f",
        key=f"leg_quantity_{row_id}",
    )

    st.markdown("**Strike setup**")
    define_by = st.selectbox(
        "Editable input",
        DEFINE_MODES,
        index=DEFINE_MODES.index(row["define_by"]),
        key=f"leg_mode_{row_id}",
        help=(
            "Boundary calculates the strike. Strike calculates the implied "
            "boundary confidence."
        ),
    )
    boundary_type = st.selectbox(
        "Boundary type",
        BOUNDARY_TYPES,
        index=BOUNDARY_TYPES.index(row["boundary_type"]),
        key=f"leg_boundary_{row_id}",
    )
    curve = curves[ticker]

    if define_by == "Boundary":
        confidence_pct = st.number_input(
            "Confidence (%) - editable",
            min_value=0.1,
            max_value=99.9,
            value=float(row["confidence_pct"]),
            step=1.0,
            format="%.1f",
            key=f"leg_confidence_{row_id}",
        )
        strike = strike_at_confidence(
            curve,
            confidence_pct / 100.0,
            boundary_type=boundary_type,
            normalized_spot=normalized_spot,
        )
        st.number_input(
            "Strike - calculated",
            value=float(strike),
            disabled=True,
            format="%.2f",
            key=f"leg_strike_locked_{row_id}",
        )
    else:
        strike = st.number_input(
            "Strike - editable",
            min_value=0.01,
            value=float(row["strike"]),
            step=5.0,
            format="%.2f",
            key=f"leg_strike_{row_id}",
        )
        confidence_pct = 100.0 * confidence_at_strike(
            curve,
            strike,
            boundary_type=boundary_type,
            normalized_spot=normalized_spot,
        )
        st.number_input(
            "Confidence (%) - calculated",
            value=float(confidence_pct),
            disabled=True,
            format="%.1f",
            key=f"leg_confidence_locked_{row_id}",
        )

    previous_ticker = str(row.get("ticker", ticker))
    initial_iv = (
        float(row.get("pricing_iv", iv_by_ticker.loc[ticker]))
        if ticker == previous_ticker
        else float(iv_by_ticker.loc[ticker])
    )
    pricing_iv = st.number_input(
        "Pricing IV",
        min_value=0.0001,
        max_value=5.0,
        value=initial_iv,
        step=0.01,
        format="%.2f",
        key=f"leg_iv_{row_id}",
        help="Annualized decimal: 0.42 means 42%.",
    )

    st.caption(
        f"{position} {quantity:.3f} x {ticker} {option_type} @ {strike:.2f} | "
        f"{confidence_pct:.1f}% {boundary_type.lower()}"
    )
    remove = st.button(
        "Remove leg",
        key=f"leg_remove_{row_id}",
        use_container_width=True,
    )

    rendered = {
        "id": row_id,
        "active": active,
        "ticker": ticker,
        "option_type": option_type,
        "position": position,
        "quantity": quantity,
        "define_by": define_by,
        "boundary_type": boundary_type,
        "confidence_pct": confidence_pct,
        "strike": strike,
        "pricing_iv": pricing_iv,
    }
    return rendered, remove


def render_interactive_leg_editor(
    *,
    tickers: list[str],
    curves: dict[str, pd.DataFrame],
    default_ticker: str,
    default_iv: float,
    iv_by_ticker: pd.Series,
    normalized_spot: float = 100.0,
    state_key: str = "phase5_interactive_rows",
) -> pd.DataFrame:
    """Render up to four vertical option cards per row."""
    if state_key not in st.session_state:
        st.session_state[state_key] = default_interactive_rows(
            default_ticker, default_iv
        )
    rows = st.session_state[state_key]

    rendered_rows = []
    remove_id = None
    for start in range(0, len(rows), CARDS_PER_ROW):
        batch = rows[start : start + CARDS_PER_ROW]
        card_columns = st.columns(CARDS_PER_ROW)
        for offset, row in enumerate(batch):
            with card_columns[offset]:
                with st.container(border=True):
                    rendered, remove = _render_leg_card(
                        row=row,
                        row_number=start + offset + 1,
                        tickers=tickers,
                        curves=curves,
                        iv_by_ticker=iv_by_ticker,
                        normalized_spot=normalized_spot,
                    )
                    rendered_rows.append(rendered)
                    if remove:
                        remove_id = int(row["id"])

    if remove_id is not None:
        st.session_state[state_key] = [
            row for row in rendered_rows if row["id"] != remove_id
        ]
        st.rerun()

    if st.button("Add another option leg", type="secondary"):
        next_id = max([int(row["id"]) for row in rendered_rows], default=0) + 1
        rendered_rows.append(
            {
                "id": next_id,
                "active": True,
                "ticker": default_ticker,
                "option_type": "Call",
                "position": "Long",
                "quantity": 0.10,
                "define_by": "Boundary",
                "boundary_type": "Win boundary",
                "confidence_pct": 80.0,
                "strike": 100.0,
                "pricing_iv": float(iv_by_ticker.loc[default_ticker]),
            }
        )
        st.session_state[state_key] = rendered_rows
        st.rerun()

    st.session_state[state_key] = rendered_rows
    return pd.DataFrame(
        [
            {
                "Active": row["active"],
                "Ticker": row["ticker"],
                "Option type": row["option_type"],
                "Position": row["position"],
                "Quantity": row["quantity"],
                "Strike source": "Manual strike",
                "Boundary confidence (%)": row["confidence_pct"],
                "Manual strike": row["strike"],
                "Pricing IV": row["pricing_iv"],
                "Definition mode": row["define_by"],
                "Boundary type": row["boundary_type"],
                "Implied confidence (%)": row["confidence_pct"],
            }
            for row in rendered_rows
        ]
    )
