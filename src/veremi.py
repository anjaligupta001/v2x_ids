"""Load VeReMi Extension simulation runs into clean pandas tables."""
from pathlib import Path
import json
import re

import pandas as pd

# Standard VeReMi Extension (F2MD) attack codes
ATTACK_NAMES = {
    0: "Genuine",
    1: "ConstPos", 2: "ConstPosOffset", 3: "RandomPos", 4: "RandomPosOffset",
    5: "ConstSpeed", 6: "ConstSpeedOffset", 7: "RandomSpeed", 8: "RandomSpeedOffset",
    9: "EventualStop", 10: "Disruptive", 11: "DataReplay", 12: "DelayedMessages",
    13: "DoS", 14: "DoSRandom", 15: "DoSDisruptive",
    16: "GridSybil", 17: "DataReplaySybil", 18: "DoSRandomSybil", 19: "DoSDisruptiveSybil",
}

FILE_RE = re.compile(r"traceJSON-(\d+)-(\d+)-A(\d+)-(\d+)-(\d+)\.json")


def _xy(df, col, prefix):
    """Split a [x, y, z] list column into two numeric columns."""
    df[f"{prefix}_x"] = df[col].str[0]
    df[f"{prefix}_y"] = df[col].str[1]


def load_run(run_dir):
    """Load one simulation run folder.

    Returns
    -------
    msgs : one row per received V2X message (type 3), labelled by the sender's attack code
    ego  : one row per receiver GPS reading (type 2), i.e. where each receiving car really is
    """
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("traceJSON-*.json"))
    if not files:
        raise FileNotFoundError(f"No traceJSON files found in {run_dir}")

    # 1) Labels: each vehicle's attack code is in its own file name
    vehicle_label = {}
    for f in files:
        m = FILE_RE.match(f.name)
        vehicle_label[int(m[1])] = int(m[3])

    # 2) Read every car's inbox
    msg_rows, ego_rows = [], []
    for f in files:
        receiver = int(FILE_RE.match(f.name)[1])
        with open(f) as fh:
            for line in fh:
                rec = json.loads(line)
                rec["receiver"] = receiver
                (msg_rows if rec["type"] == 3 else ego_rows).append(rec)

    msgs = pd.DataFrame(msg_rows)
    ego = pd.DataFrame(ego_rows)

    # 3) Tidy columns
    for df in (msgs, ego):
        for col, p in [("pos", "pos"), ("spd", "spd"), ("acl", "acl"), ("hed", "hed"),
                       ("pos_noise", "posn"), ("spd_noise", "spdn"),
                       ("acl_noise", "acln"), ("hed_noise", "hedn")]:
            _xy(df, col, p)
        df.drop(columns=["pos", "spd", "acl", "hed", "pos_noise", "spd_noise",
                         "acl_noise", "hed_noise", "type"], inplace=True)

    msgs = msgs.rename(columns={"rcvTime": "rcv_time", "sendTime": "send_time",
                                "senderPseudo": "sender_pseudo", "messageID": "message_id"})
    ego = ego.rename(columns={"rcvTime": "time"})

    # 4) Attach labels (sender's attack code)
    msgs["label"] = msgs["sender"].map(vehicle_label)
    unknown = msgs["label"].isna().sum()
    msgs = msgs.dropna(subset=["label"]).copy()      # senders with no log file of their own
    msgs["label"] = msgs["label"].astype(int)
    msgs["attack_name"] = msgs["label"].map(ATTACK_NAMES)
    msgs["is_attack"] = (msgs["label"] != 0).astype(int)

    print(f"Loaded {len(files)} vehicles | {len(msgs):,} received messages "
          f"| {len(ego):,} GPS readings | dropped {unknown:,} messages from unknown senders")
    return msgs, ego
