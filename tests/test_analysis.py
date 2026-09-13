import json
from flygpt.analysis import paired_differences, apply_claim_rule
from flygpt.config import ClaimRule


def _write(root, name, val):
    d = root / name; d.mkdir()
    (d / "log.jsonl").write_text(json.dumps({"step": 10, "val_loss": val}) + "\n")


def test_claim_rule_pass_and_fail(tmp_path):
    for k, (r, c) in enumerate([(1.70, 1.80), (1.72, 1.79), (1.69, 1.77), (1.71, 1.81), (1.70, 1.78)], 1):
        _write(tmp_path, f"real_seed{k}", r); _write(tmp_path, f"dp_seed{k}", c)
    p = paired_differences(tmp_path, "real", "dp", [1, 2, 3, 4, 5])
    v = apply_claim_rule(p, ClaimRule(True, 0.05), 5)
    assert v["passed"] and v["direction"] == "real_better"
    assert not apply_claim_rule(p, ClaimRule(True, 0.5), 5)["passed"]        # effect too small
    assert not apply_claim_rule(p, ClaimRule(True, 0.05), 6)["passed"]       # not enough seeds


def test_mixed_signs_fail(tmp_path):
    for k, (r, c) in enumerate([(1.70, 1.90), (1.72, 1.60), (1.69, 1.90)], 1):
        _write(tmp_path, f"real_seed{k}", r); _write(tmp_path, f"dp_seed{k}", c)
    p = paired_differences(tmp_path, "real", "dp", [1, 2, 3])
    v = apply_claim_rule(p, ClaimRule(True, 0.05), 3)
    assert not v["passed"] and "no detectable" in v["verdict"]
