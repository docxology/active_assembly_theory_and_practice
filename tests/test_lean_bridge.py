"""Direct failure-mode tests for the generated Python-to-Lean bridge."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.lean_bridge as lean_bridge


def test_generated_fixture_is_deterministic_and_elan_is_required(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = lean_bridge.write_generated_lean_fixture(tmp_path)
    source = tmp_path / lean_bridge.GENERATED_RELATIVE_PATH
    assert source.is_file()
    assert metadata["python_reference_final_rax"] == lean_bridge.BRIDGE_EXPECTED_RAX
    assert metadata["source_sha256"]
    assert metadata["machine_state_schema"] == (
        "explicit_16_gpr_rip_flags_memory_stack_step_status"
    )
    assert metadata["machine_state_register_names"] == list(
        lean_bridge.BRIDGE_REGISTER_NAMES
    )
    assert len(metadata["python_initial_machine_state"]["registers"]) == 16
    assert metadata["lean_machine_state_theorems"] == list(
        lean_bridge.BRIDGE_MACHINE_STATE_THEOREMS
    )
    monkeypatch.setattr(lean_bridge.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="elan is required"):
        lean_bridge._elan_executable()


def test_generated_bridge_reports_compiler_failure_and_value_mismatch(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "formal").mkdir()
    monkeypatch.setattr(lean_bridge, "_elan_executable", lambda: "elan")
    monkeypatch.setattr(
        lean_bridge.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="compiler failure"
        ),
    )
    with pytest.raises(RuntimeError, match="bridge failed"):
        lean_bridge.run_generated_lean_fixture(tmp_path)

    monkeypatch.setattr(
        lean_bridge.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        lean_bridge,
        "write_generated_lean_fixture",
        lambda root: {
            "python_reference_final_rax": 1,
            "lean_expected_final_rax": 2,
        },
    )
    with pytest.raises(RuntimeError, match="values disagree"):
        lean_bridge.run_generated_lean_fixture(tmp_path)


def test_bridge_fixture_catalog_covers_expansion_families() -> None:
    bridge_ids = [fixture["bridge_id"] for fixture in lean_bridge.BRIDGE_FIXTURES]
    assert bridge_ids == [
        "push_pop",
        "cmp_jcc_taken",
        "cmp_jcc_not_taken",
        "mov_width_32",
        "mov_width_64",
        "mov_width_16",
    ]
    assert all(fixture["theorems"] for fixture in lean_bridge.BRIDGE_FIXTURES)
    source = lean_bridge._lean_source()
    assert "do not hand edit" in source
    for theorem in lean_bridge.BRIDGE_FIXTURE_THEOREMS:
        assert f"theorem {theorem.removeprefix('ActiveAssembly.')} " in source


def test_bridge_fixture_traces_fail_closed_on_mismatch(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drift between the real Unicorn/reference run and the Lean expected
    state must abort the bridge with an explicit failure."""

    metadata = lean_bridge.write_generated_lean_fixture(tmp_path)
    assert len(metadata["fixtures"]) == len(lean_bridge.BRIDGE_FIXTURES)

    monkeypatch.setattr(lean_bridge, "_elan_executable", lambda: "elan")
    monkeypatch.setattr(
        lean_bridge.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )
    monkeypatch.setattr(
        lean_bridge,
        "_verify_bridge_fixture",
        lambda fixture: {
            "kernel_checked": True,
            "boundary_matches_lean_expected": False,
            "final_matches_lean_expected": False,
            "memory_matches_lean_expected": False,
            "reference_boundary_matches": False,
            "reference_final_matches": False,
            "trace_matches_lean_expected": False,
        },
    )
    with pytest.raises(RuntimeError, match="does not match"):
        lean_bridge.run_generated_lean_fixture(tmp_path)


def test_bridge_fails_closed_when_emulation_is_unavailable(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing Unicorn runtime surfaces as an explicit error, not silence."""

    lean_bridge.write_generated_lean_fixture(tmp_path)
    monkeypatch.setattr(lean_bridge, "_elan_executable", lambda: "elan")
    monkeypatch.setattr(
        lean_bridge.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )

    def broken_emulation(fixture: object) -> object:
        raise RuntimeError("unicorn runtime unavailable")

    monkeypatch.setattr(lean_bridge, "_unicorn_fixture_run", broken_emulation)
    with pytest.raises(RuntimeError, match="unicorn runtime unavailable"):
        lean_bridge.run_generated_lean_fixture(tmp_path)


def test_bridge_fixture_python_state_is_complete_and_hashed(tmp_path) -> None:
    metadata = lean_bridge.write_generated_lean_fixture(tmp_path)
    for row in metadata["fixtures"]:
        state = row["python_initial_machine_state"]
        assert set(state["registers"]) == set(lean_bridge.BRIDGE_REGISTER_NAMES)
        assert row["python_initial_machine_state_sha256"]
        assert row["instruction_count"] == len(row["lean_program"])
        assert row["theorems"] == [
            theorem
            for fixture in lean_bridge.BRIDGE_FIXTURES
            if fixture["bridge_id"] == row["bridge_id"]
            for theorem in fixture["theorems"]
        ]


def test_bridge_fixture_memory_rows_are_independent_copies() -> None:
    """Boundary/final memory must not alias the emulator readback."""

    fixture = next(
        f for f in lean_bridge.BRIDGE_FIXTURES if f["bridge_id"] == "push_pop"
    )
    row = lean_bridge._verify_bridge_fixture(fixture)
    boundary_memory = row["unicorn_boundary"]["memory"]
    final_memory = row["unicorn_final"]["memory"]
    readback = row["unicorn_memory_readback"]
    assert row["unicorn_boundary"] is not row["unicorn_final"]
    assert boundary_memory is not final_memory
    assert boundary_memory is not readback
    assert final_memory is not readback
    # The real push fixture leaves the stack bytes it wrote, so mutating one
    # recorded state must not silently rewrite the others.
    assert readback, "push_pop declares expected memory, readback must be filled"
    probe_address = next(iter(readback))
    boundary_memory[probe_address] = -1
    assert final_memory[probe_address] != -1
    assert readback[probe_address] != -1


def test_bridge_fixture_fails_closed_on_under_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An emulator that runs fewer instructions than the fixture declares
    must raise with the fixture id, never degrade to a short readback."""

    import unicorn

    real_emu_start = unicorn.Uc.emu_start

    def short_emu_start(uc, begin, until, timeout=0, count=0):
        # Halt after one instruction so the fixture under-executes.
        return real_emu_start(uc, begin, min(until, begin + 1), timeout, count)

    monkeypatch.setattr(unicorn.Uc, "emu_start", short_emu_start)
    fixture = next(
        f for f in lean_bridge.BRIDGE_FIXTURES if f["bridge_id"] == "push_pop"
    )
    with pytest.raises(RuntimeError, match="push_pop"):
        lean_bridge._unicorn_fixture_run(fixture)
