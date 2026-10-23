"""Round discovery and the performance-vs-round table (question_evaluate/eval_rounds.py)."""

import csv
import json

from question_evaluate.eval_rounds import discover_rounds, write_table


def adapter(path):
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}")
    return str(path)


def test_discover_raw_rounds_in_order(tmp_path):
    root = tmp_path / "models"
    paths = [adapter(root / f"Q_s_b{big}" / f"global_step_{step}" / "actor" / "lora_adapter")
             for big, step in [(1, 10), (1, 5), (2, 5), (1, 15)]]
    (root / "Q_s_b1" / "global_step_20" / "actor" / "lora_adapter").mkdir(parents=True)  # incomplete: skipped
    rounds = discover_rounds(str(root), "auto")
    assert rounds == [(1, paths[1]), (2, paths[0]), (3, paths[3]), (4, paths[2])]


def test_discover_prefers_served_round_adapters(tmp_path):
    root = tmp_path / "models"
    adapter(root / "Q_s_b1" / "global_step_5" / "actor" / "lora_adapter")
    served = [adapter(root / "round_adapters" / "solver" / f"round_{r:03d}") for r in (1, 2)]
    assert discover_rounds(str(root), "auto") == [(1, served[0]), (2, served[1])]
    assert len(discover_rounds(str(root), "raw")) == 1


def write_round(out_dir, r, adapter_path, records):
    d = out_dir / f"round_{r:03d}"
    d.mkdir(parents=True)
    (d / "records.jsonl").write_text("".join(json.dumps(rec) + "\n" for rec in records))
    stats = {"n": len(records), "accuracy": 0.5, "balanced_accuracy": 0.6}
    summary = {"overall": {"all_voted": stats, "in_band": {"n": 1, "accuracy": 1.0}, "kept_fraction": 0.25,
                           "voted_fraction": 1.0, "candidates": len(records)}, "round": r, "adapter": adapter_path}
    (d / "report.json").write_text(json.dumps(summary))


def test_table(tmp_path):
    records = [
        {"label": 1, "votes": [1, 1, 0]},  # TP
        {"label": 1, "votes": [0, 0, 1]},  # FN
        {"label": 0, "votes": [1, 1, 1]},  # FP
        {"label": 0, "votes": [0, 0, 0]},  # TN
    ]
    write_round(tmp_path, 0, None, records)
    write_round(tmp_path, 1, "/runs/round_001", records)
    write_table(str(tmp_path), "test")
    rows = list(csv.DictReader(open(tmp_path / "rounds.csv")))
    assert [row["round"] for row in rows] == ["0", "1"]
    assert rows[0]["adapter"] == "base" and rows[1]["adapter"] == "/runs/round_001"
    assert float(rows[1]["positive_recall"]) == 0.5 and float(rows[1]["positive_precision"]) == 0.5
    md = (tmp_path / "rounds.md").read_text()
    assert "| 1 | 50.0% | 60.0% | 50.0% | 50.0% | 100.0% | 25.0% | 100.0% | 4 | /runs/round_001 |" in md
