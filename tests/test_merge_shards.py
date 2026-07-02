import csv

from src.merge_shards import merge


def _write_shard(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doi", "sample_id"])
        writer.writeheader()
        writer.writerows(rows)


def test_merge_combines_shards_with_single_header(tmp_path):
    _write_shard(tmp_path / "demographics_table.shard0.csv", [{"doi": "a", "sample_id": 1}])
    _write_shard(tmp_path / "demographics_table.shard1.csv", [{"doi": "b", "sample_id": 1}, {"doi": "c", "sample_id": 2}])

    out_path = tmp_path / "demographics_table.csv"
    total = merge(str(tmp_path / "demographics_table.shard*.csv"), str(out_path))

    assert total == 3
    with open(out_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["doi"] for r in rows] == ["a", "b", "c"]
