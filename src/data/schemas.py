SPOT_COLUMNS = ["timestamp", "open", "high", "low", "close"]

FUTURES_COLUMNS = [
    "timestamp",
    "contract",
    "expiry",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_interest",
]

FEATURE_COLUMNS = [
    "spot_ret_1m",
    "spot_ret_5m",
    "spot_ret_15m",
    "fut_ret_1m",
    "fut_ret_5m",
    "fut_ret_15m",
    "spot_range",
    "fut_range",
    "basis",
    "basis_pct",
    "basis_change",
    "basis_zscore_60m",
    "fut_volume_zscore_by_time_of_day",
    "oi_change_1m",
    "oi_change_15m",
    "realized_vol_15m",
    "realized_vol_30m",
    "minutes_from_open",
    "minutes_to_close",
    "day_of_week",
    "days_to_expiry",
    "is_expiry_day",
    "is_expiry_week",
]

LABEL_COLUMNS = [
    "future_ret_15m",
    "future_ret_30m",
    "future_realized_vol_30m",
    "future_range_30m",
    "future_basis_change_30m",
    "max_adverse_excursion_30m",
    "max_favorable_excursion_30m",
]

PREDICTION_TARGETS = {
    "pred_return_15m": "future_ret_15m",
    "pred_return_30m": "future_ret_30m",
    "pred_vol_30m": "future_realized_vol_30m",
    "pred_range_30m": "future_range_30m",
    "pred_basis_change_30m": "future_basis_change_30m",
}
