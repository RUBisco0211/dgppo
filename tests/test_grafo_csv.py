import csv
import json
import tempfile
from pathlib import Path

from scripts.export_grafo_cl_csv import export_run


def test_csv_records_interim_and_stage_end_metrics():
    with tempfile.TemporaryDirectory() as temporary:
        run = Path(temporary) / "full_ts_frog_ewcm_seed0_b512"
        run.mkdir()
        records = [
            {"counters/stage": 1, "counters/update": 1,
             "counters/total_frames": 4096},
            {"counters/stage": 1, "counters/update": 100,
             "counters/total_frames": 409600,
             "eval/GRAFO-5/success_rate": 0.4},
            {"counters/stage": 1, "counters/update": 1221,
             "counters/total_frames": 5001216,
             "eval/GRAFO-5/success_rate": 0.8, "continual/AP": 0.8},
            {"counters/stage": 2, "counters/update": 1222,
             "counters/total_frames": 5005312},
            {"counters/stage": 2, "counters/update": 2442,
             "counters/total_frames": 10002432,
             "eval/GRAFO-5/success_rate": 0.6,
             "eval/GRAFO-6(1)/success_rate": 0.9,
             "continual/AP": 0.85, "continual/AF": -0.2,
             "continual/AFP": 0.75},
        ]
        (run / "metrics.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n")
        assert export_run(run) == 3
        with (run / "evaluation.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows[0]["stage_complete"] == "0"
        assert rows[1]["trained_task"] == "GRAFO-5"
        assert rows[1]["update_in_stage"] == "1221"
        assert rows[1]["AP"] == "0.8"
        assert rows[2]["trained_task"] == "GRAFO-6(1)"
        assert rows[2]["GRAFO-5/success_rate"] == "0.6"
        assert rows[2]["GRAFO-6(1)/success_rate"] == "0.9"
        assert rows[2]["AF"] == "-0.2"
        assert rows[2]["AFP"] == "0.75"
