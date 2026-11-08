"""Alarm logic and SMS / MQTT-style alarm messages.

The module only builds the payloads (JSON-serialisable dicts); it does not talk to a broker or an SMS gateway.
Publish `payload["mqtt"]["payload"]` with any MQTT client and send `payload["sms"]` through your own gateway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from .cooling import CoolingProfile, clock_str
from .model import THRESHOLDS, frost_probability

SCHEMA = "frost-alarm/1"
PROTECTION_MARGIN_C = 1.0  # start protection when the 2 m air temperature falls to (alarm threshold + margin)


@dataclass
class Hysteresis:
    """Alarm latch: switch on at p >= p_on, switch off only after `off_updates` consecutive updates below p_off."""

    p_on: float = 0.30
    p_off: float = 0.15
    off_updates: int = 2
    active: bool = False
    _below: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_off <= self.p_on <= 1.0:
            raise ValueError("need 0 <= p_off <= p_on <= 1")
        if self.off_updates < 1:
            raise ValueError("off_updates must be >= 1")

    def update(self, p: float) -> bool:
        if not self.active:
            if p >= self.p_on:
                self.active, self._below = True, 0
        else:
            self._below = self._below + 1 if p < self.p_off else 0
            if self._below >= self.off_updates:
                self.active, self._below = False, 0
        return self.active


def confidence_label(tmin_hat: float, sigma: float, threshold: float) -> str:
    """How far the forecast sits from the threshold in units of its own spread."""
    z = abs(threshold - tmin_hat) / max(sigma, 1e-6)
    return "high" if z >= 1.5 else "medium" if z >= 0.75 else "low"


RECOMMENDATIONS = {
    "alarm": ("Frost expected. Prepare protection now (for example frost candles or heaters, wind machines, "
              "overhead sprinklers where available) and start when the 2 m air temperature at the plot reaches "
              "{trigger:.1f} C. Check the critical temperature of the current bud stage."),
    "alarm_no_start": ("Frost possible. The forecast minimum stays above the {trigger:.1f} C start trigger, so "
                       "keep protection ready and start only if the plot sensor falls to it; re-check at the next update."),
    "watch": "Frost possible. Keep protection ready and re-check at the next update.",
    "none": "No frost expected.",
}


def build_alarm(station: str, night: date, issued_local: datetime, tmin_hat: float, sigma: float, t_now: float,
                state: Hysteresis, profile: CoolingProfile, threshold: float = 0.0, update_index: int = 0,
                tmin_ci_level: float = 0.8) -> dict:
    """Alarm message for one forecast update. Advances the hysteresis `state`."""
    before = state.active
    p = {f"{th:g}": round(float(frost_probability(tmin_hat, sigma, th)), 3) for th in THRESHOLDS}
    p_main = float(frost_probability(tmin_hat, sigma, threshold))
    active = state.update(p_main)
    level = "alarm" if active else ("watch" if p_main >= state.p_off else "none")
    trigger = threshold + PROTECTION_MARGIN_C
    hour_issued = issued_local.hour + issued_local.minute / 60.0
    start_h = profile.time_to_reach(t_now, tmin_hat, trigger, start_hour=max(hour_issued, 18.0)) if active else None
    start_iso = clock_str(start_h, night) if start_h is not None else None
    z = 1.2816 if tmin_ci_level == 0.8 else 1.96
    payload = {
        "schema": SCHEMA,
        "station": station,
        "night": night.isoformat(),
        "issued_at": issued_local.strftime("%Y-%m-%dT%H:%M"),
        "update_index": update_index,
        "level": level,
        "alarm": bool(active),
        "threshold_c": threshold,
        "tmin_forecast_c": round(float(tmin_hat), 1),
        "tmin_interval_c": [round(float(tmin_hat - z * sigma), 1), round(float(tmin_hat + z * sigma), 1)],
        "interval_level": tmin_ci_level,
        "p_frost": p,
        "confidence": confidence_label(tmin_hat, sigma, threshold),
        "hysteresis": {"p_on": state.p_on, "p_off": state.p_off, "off_updates": state.off_updates,
                       "state_before": "on" if before else "off", "state_after": "on" if active else "off"},
        "start_protection_at": start_iso,
        "start_protection_trigger_c": trigger if active else None,
        "recommendation": RECOMMENDATIONS["alarm_no_start" if active and start_iso is None else level].format(
            trigger=trigger),
    }
    payload["sms"] = sms_text(payload)
    payload["mqtt"] = {"topic": f"frost/{station.lower()}/alarm", "qos": 1, "retain": True,
                       "payload": json.dumps({k: payload[k] for k in payload if k not in ("sms", "mqtt")})}
    return payload


def sms_text(a: dict) -> str:
    """Single-segment SMS text (<= 160 chars)."""
    if a["level"] == "none":
        txt = f"Frost {a['station']} {a['night']}: none expected, Tmin {a['tmin_forecast_c']:.1f}C."
    else:
        start = f" Start protection {a['start_protection_at'][11:]}." if a["start_protection_at"] else ""
        txt = (f"FROST {a['level'].upper()} {a['station']} {a['night']}: Tmin {a['tmin_forecast_c']:.1f}C, "
               f"P(<=0C) {a['p_frost']['0'] * 100:.0f}%, conf {a['confidence']}.{start}")
    return txt[:160]


def to_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
