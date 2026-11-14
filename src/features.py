"""Plausibility ("lie detector") features for V2X misbehaviour detection.

Every feature uses ONLY information a real receiving car would have:
the messages it received (identified by pseudonym, never the real sender ID)
and its own GPS position.
"""
import numpy as np
import pandas as pd

FEATURES = [
    # --- single-message checks ---
    "speed", "accel", "heading_vs_velocity_deg", "dist_to_receiver",
    # --- checks against the previous message from the same pseudonym ---
    "is_first", "dt", "jump", "implied_speed", "speed_mismatch",
    "pos_pred_error", "speed_change_rate", "accel_mismatch", "heading_vs_motion_deg",
    # --- behaviour over time ---
    "msgs_last_1s", "pseudos_nearby_1s",
]


def _angle_deg(ax, ay, bx, by):
    """Angle in degrees between two 2-D vectors (NaN if either is ~zero)."""
    dot = ax * bx + ay * by
    norm = np.hypot(ax, ay) * np.hypot(bx, by)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(norm > 1e-6, dot / norm, np.nan)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def add_features(msgs, ego):
    """Return a copy of `msgs` with the FEATURES columns added."""
    df = msgs.copy()

    # ---------- single-message checks ----------
    df["speed"] = np.hypot(df["spd_x"], df["spd_y"])
    df["accel"] = np.hypot(df["acl_x"], df["acl_y"])
    # a car normally drives in the direction it is facing
    df["heading_vs_velocity_deg"] = _angle_deg(df["hed_x"], df["hed_y"], df["spd_x"], df["spd_y"])
    # (rcv_time - send_time) is always 0 in VeReMi Extension, so it is not used as a feature

    # where was the RECEIVER when it got the message? (its own GPS)
    ego_pos = ego[["receiver", "time", "pos_x", "pos_y"]].rename(
        columns={"pos_x": "rx_x", "pos_y": "rx_y"}).sort_values("time")
    df = df.sort_values("rcv_time")
    df = pd.merge_asof(df, ego_pos, left_on="rcv_time", right_on="time",
                       by="receiver", direction="backward").drop(columns="time")
    # radio range is limited: a sender claiming to be very far away is suspicious
    df["dist_to_receiver"] = np.hypot(df["pos_x"] - df["rx_x"], df["pos_y"] - df["rx_y"])

    # ---------- compare with previous message from same pseudonym ----------
    df = df.sort_values(["receiver", "sender_pseudo", "rcv_time"]).reset_index(drop=True)
    g = df.groupby(["receiver", "sender_pseudo"], sort=False)
    prev = g[["rcv_time", "send_time", "pos_x", "pos_y", "spd_x", "spd_y", "acl_x", "acl_y", "speed"]].shift(1)

    df["is_first"] = prev["rcv_time"].isna().astype(int)
    dt_send = df["send_time"] - prev["send_time"]
    df["dt"] = df["rcv_time"] - prev["rcv_time"]
    dx, dy = df["pos_x"] - prev["pos_x"], df["pos_y"] - prev["pos_y"]
    df["jump"] = np.hypot(dx, dy)
    safe_dt = dt_send.where(dt_send > 1e-3)
    # speed needed to cover the jump vs. the speed the car claims
    df["implied_speed"] = df["jump"] / safe_dt
    df["speed_mismatch"] = (df["implied_speed"] - (df["speed"] + prev["speed"]) / 2).abs()
    # where SHOULD the car be now, given its last position, speed and acceleration?
    pred_x = prev["pos_x"] + prev["spd_x"] * dt_send + 0.5 * prev["acl_x"] * dt_send ** 2
    pred_y = prev["pos_y"] + prev["spd_y"] * dt_send + 0.5 * prev["acl_y"] * dt_send ** 2
    df["pos_pred_error"] = np.hypot(df["pos_x"] - pred_x, df["pos_y"] - pred_y)
    # does the change in speed match the claimed acceleration?
    df["speed_change_rate"] = (df["speed"] - prev["speed"]) / safe_dt
    df["accel_mismatch"] = (df["speed_change_rate"].abs() - df["accel"]).abs()
    # does the car move in the direction of its claimed velocity?
    df["heading_vs_motion_deg"] = np.where(df["jump"] > 0.5,
                                           _angle_deg(dx, dy, prev["spd_x"], prev["spd_y"]), np.nan)

    # ---------- behaviour over time ----------
    # how many messages did this pseudonym send in the last second? (DoS)
    df["one"] = 1
    df["msgs_last_1s"] = (df.set_index(pd.to_timedelta(df["rcv_time"], unit="s"))
                            .groupby(["receiver", "sender_pseudo"], sort=False)["one"]
                            .rolling("1s").sum().to_numpy())
    # how many different pseudonyms claim to be within 50 m of this message's position,
    # heard by this receiver in the same 1-second window? (Sybil: many fake cars in one spot)
    df["sec"] = np.floor(df["rcv_time"]).astype(int)
    df["cell_x"] = (df["pos_x"] // 50).astype(int)
    df["cell_y"] = (df["pos_y"] // 50).astype(int)
    df["pseudos_nearby_1s"] = df.groupby(["receiver", "sec", "cell_x", "cell_y"])["sender_pseudo"].transform("nunique")

    return df.drop(columns=["one", "sec", "cell_x", "cell_y"])
