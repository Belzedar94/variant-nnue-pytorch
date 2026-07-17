from types import SimpleNamespace

import pytest

from atomic_v3 import production


SHA256 = "a" * 64
TRAINER_COMMIT = "b" * 40


def _entry_arguments(tmp_path):
    provider = tmp_path / "training_data_loader.dll"
    provider.write_bytes(b"provider")
    return {
        "receipt_path": tmp_path / "bootstrap.receipt.json",
        "receipt_sha256": SHA256,
        "provider_library": provider,
        "output_root": tmp_path / "output",
        "trainer_commit": TRAINER_COMMIT,
        "run_ids": (next(iter(production.RUN_CONFIGS)),),
        "device": "cuda",
    }


def test_production_entry_authenticates_semantic_evidence_before_provider_factory(
    tmp_path, monkeypatch
):
    arguments = _entry_arguments(tmp_path)
    evidence = SimpleNamespace(
        receipt_path=arguments["receipt_path"], receipt_sha256=SHA256
    )
    events = []

    def inspect(receipt_path, receipt_sha256):
        assert receipt_path == arguments["receipt_path"]
        assert receipt_sha256 == SHA256
        events.append("inspect_bootstrap_roles")
        return evidence

    def provider_factory(snapshot, role, **_kwargs):
        assert snapshot is evidence
        assert events[0] == "inspect_bootstrap_roles"
        events.append("provider_factory:" + role)
        return lambda: None

    class StopAfterProviderFactories(RuntimeError):
        pass

    def stop_prepare(*_args, **_kwargs):
        events.append("prepare_production_run")
        raise StopAfterProviderFactories

    monkeypatch.setattr(production, "_validate_cuda_device", lambda _value: "cuda:0")
    monkeypatch.setattr(production, "inspect_bootstrap_roles", inspect)
    monkeypatch.setattr(production, "_provider_factory", provider_factory)
    monkeypatch.setattr(production, "prepare_production_run", stop_prepare)
    monkeypatch.setattr(production, "cleanup_prepared", lambda _prepared: None)
    monkeypatch.setattr(
        production,
        "load_or_create_shared_initial_state",
        lambda _path: (object(), {"path": "shared.pt", "state_sha256": SHA256}),
    )
    monkeypatch.setattr(
        production,
        "_run_mandatory_preflight",
        lambda **_kwargs: {
            "status": "target",
            "gate_passed": True,
            "benchmark": {
                "gate": {"metrics": {"cold_authentication_seconds": 0.0}}
            },
        },
    )

    with pytest.raises(StopAfterProviderFactories):
        production.execute_runs(**arguments)

    assert events == [
        "inspect_bootstrap_roles",
        "provider_factory:train",
        "provider_factory:validation",
        "prepare_production_run",
    ]


def test_production_entry_never_reaches_provider_factory_when_evidence_fails(
    tmp_path, monkeypatch
):
    arguments = _entry_arguments(tmp_path)

    class EvidenceRejected(RuntimeError):
        pass

    def reject_evidence(*_args):
        raise EvidenceRejected

    def forbidden_factory(*_args, **_kwargs):
        pytest.fail("provider factory was reached without authenticated bootstrap evidence")

    monkeypatch.setattr(production, "_validate_cuda_device", lambda _value: "cuda:0")
    monkeypatch.setattr(production, "inspect_bootstrap_roles", reject_evidence)
    monkeypatch.setattr(production, "_provider_factory", forbidden_factory)

    with pytest.raises(EvidenceRejected):
        production.execute_runs(**arguments)
