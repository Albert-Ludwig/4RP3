"""Inspect raw Intel records without cleaning, imputing, or changing them."""

from array import array
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EPOCH = datetime(2004, 1, 1)


def save(name, data):
    (ROOT / "inspection" / name).write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    locations = np.loadtxt(ROOT / "raw/mote_locs.txt")
    ids = set(locations[:, 0].astype(int))
    times = defaultdict(lambda: array("d"))
    temperatures = defaultdict(lambda: array("d"))
    columns, counts, unknown, missing = Counter(), Counter(), Counter(), Counter()
    daily = defaultdict(Counter)
    bins = defaultdict(set)
    examples = []
    first = last = None
    with (ROOT / "raw/data.txt").open() as stream:
        for line_number, line in enumerate(stream, 1):
            counts["raw_rows"] += 1
            # Empty space-delimited fields are missing values. Whitespace
            # splitting would shift humidity/light/voltage into wrong columns.
            fields = line.rstrip("\r\n").split(" ")
            columns[len(fields)] += 1
            if len(fields) != 8:
                counts["rows_without_eight_fields"] += 1
                if len(examples) < 5:
                    examples.append({"line": line_number, "text": line.rstrip()})
                continue
            for name, value in zip(["date", "time", "epoch", "sensor_id", "temperature", "humidity", "light", "voltage"], fields):
                if value == "":
                    missing[name] += 1
            try:
                date = datetime.fromisoformat(fields[0] + "T" + fields[1])
                int(fields[2])  # Epoch is a sequence number, not a Unix timestamp.
                first = min(first, date) if first else date
                last = max(last, date) if last else date
                if not fields[3]:
                    counts["rows_without_sensor_id"] += 1
                    continue
                mote = int(fields[3])
            except (ValueError, OverflowError):
                counts["unparseable_rows"] += 1
                continue
            counts["timestamp_and_sensor_parseable_rows"] += 1
            if mote not in ids:
                unknown[mote] += 1
                counts["rows_without_coordinates"] += 1
                continue
            if fields[4] == "":
                counts["known_location_missing_temperature"] += 1
                continue
            try:
                temp = float(fields[4])
            except ValueError:
                counts["known_location_unparseable_temperature"] += 1
                continue
            if not np.isfinite(temp):
                counts["known_location_nonfinite_temperature"] += 1
                continue
            counts["known_location_finite_temperature_rows"] += 1
            seconds = (date - EPOCH).total_seconds()
            times[mote].append(seconds)
            temperatures[mote].append(temp)
            day = date.date().isoformat()
            daily[day]["rows"] += 1
            daily[day][f"sensor_{mote}"] += 1
            bins[(day, mote)].add((date.hour * 60 + date.minute) // 15)
            # Diagnostic only. This is not an accepted cleaning threshold.
            if not 0 <= temp <= 50:
                counts["temperature_outside_0_50_c_diagnostic"] += 1
                daily[day]["outside_0_50_c_diagnostic"] += 1
    sensor_rows = []
    all_temps = []
    for mote in sorted(ids):
        t = np.frombuffer(times[mote], dtype=np.float64)
        y = np.frombuffer(temperatures[mote], dtype=np.float64)
        row = {"sensor_id": int(mote), "records": len(t)}
        if len(t):
            order = np.argsort(t)
            gaps = np.diff(t[order])
            positive = gaps[gaps > 0]
            row.update(
                first=str(EPOCH + timedelta(seconds=float(t.min()))),
                last=str(EPOCH + timedelta(seconds=float(t.max()))),
                temperature_min_c=float(y.min()), temperature_max_c=float(y.max()),
                temperature_outside_0_50_c_diagnostic=int(((y < 0) | (y > 50)).sum()),
                duplicate_timestamp_rows=int((gaps == 0).sum()),
                out_of_order_steps_in_file=int((np.diff(t) < 0).sum()),
                median_positive_gap_s=float(np.median(positive)) if len(positive) else None,
                maximum_gap_s=float(gaps.max()) if len(gaps) else None)
            all_temps.append(y)
        sensor_rows.append(row)
    days = []
    for day, values in sorted(daily.items()):
        active = [int(k[7:]) for k in values if k.startswith("sensor_")]
        days.append({"date": day, "rows": values["rows"], "sensors": len(active),
                     "outside_0_50_c_diagnostic": values["outside_0_50_c_diagnostic"],
                     "sensors_in_at_least_90_of_96_quarter_hours": sum(
                         len(bins[(day, mote)]) >= 90 for mote in active),
                     "sensor_rows": {str(m): values[f"sensor_{m}"] for m in sorted(active)}})
    y = np.concatenate(all_temps)
    report = {"counts": dict(counts), "field_counts": dict(columns),
              "missing_values_by_column": dict(missing),
              "malformed_examples": examples,
              "first_parseable_timestamp": str(first), "last_parseable_timestamp": str(last),
              "timestamp_timezone": "Not specified by source; timestamps remain naive.",
              "coordinate_sensor_count": len(ids),
              "coordinate_ids_without_finite_temperature_records": [r["sensor_id"] for r in sensor_rows if not r["records"]],
              "unknown_sensor_rows": dict(sorted(unknown.items())),
              "coordinate_bounds_m": {"x_min": float(locations[:, 1].min()),
                                      "x_max": float(locations[:, 1].max()),
                                      "y_min": float(locations[:, 2].min()),
                                      "y_max": float(locations[:, 2].max())},
              "temperature_percentiles_c": dict(zip(["0", "1", "50", "95", "99", "100"],
                                                     np.percentile(y, [0, 1, 50, 95, 99, 100]).tolist())),
              "diagnostic_range_note": "0 to 50 C only flags rows for review; no cleaning was applied.",
              "duplicate_sensor_timestamp_rows": sum(r.get("duplicate_timestamp_rows", 0) for r in sensor_rows),
              "out_of_order_steps_within_sensors": sum(r.get("out_of_order_steps_in_file", 0) for r in sensor_rows)}
    save("summary.json", report)
    save("per_sensor.json", sensor_rows)
    save("per_day.json", days)
    print(json.dumps(report, indent=2))
    print("Days with broad temporal coverage:")
    best = sorted(days, key=lambda d: (-d["sensors_in_at_least_90_of_96_quarter_hours"],
                                     d["outside_0_50_c_diagnostic"], d["date"]))[:5]
    print(json.dumps([{k: v for k, v in day.items() if k != "sensor_rows"} for day in best], indent=2))


if __name__ == "__main__":
    main()
