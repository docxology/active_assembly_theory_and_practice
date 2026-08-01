"""Availability probes for optional heavy assembly-analysis packages."""

from __future__ import annotations

from importlib import metadata, util
import shutil

from .schema import PackageProvenance


def _python_package_version(package_name: str) -> str | None:
    if util.find_spec(package_name) is None:
        return None
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _command_probe(
    name: str, adapter: str, command_names: tuple[str, ...]
) -> PackageProvenance:
    command = next(
        (candidate for candidate in command_names if shutil.which(candidate)), None
    )
    return PackageProvenance(
        name=name,
        adapter=adapter,
        version="unknown",
        optional=True,
        available=command is not None,
        command=command or " | ".join(command_names),
        details={"role": "optional baseline package probe"},
    )


def probe_optional_packages() -> list[PackageProvenance]:
    """Return provenance placeholders for optional packages without importing them eagerly."""

    angr_version = _python_package_version("angr")
    keystone_version = _python_package_version("keystone")
    gtirb_version = _python_package_version("gtirb")
    miasm_version = _python_package_version("miasm")
    qiling_version = _python_package_version("qiling")
    return [
        PackageProvenance(
            name="angr",
            adapter="angr_cfg_symbolic_facts",
            version=angr_version or "not-installed",
            optional=True,
            available=angr_version is not None,
            command="python-angr",
            details={"role": "optional CFG and symbolic facts adapter"},
        ),
        _command_probe("rizin", "rizin_static_baseline", ("rizin", "rz-bin", "rz-asm")),
        _command_probe("radare2", "radare2_static_baseline", ("r2", "radare2")),
        _command_probe(
            "llvm-objdump", "llvm_objdump_static_baseline", ("llvm-objdump",)
        ),
        _command_probe(
            "ghidra", "ghidra_headless_analysis", ("analyzeHeadless", "ghidra")
        ),
        _command_probe("bap", "bap_ir_analysis", ("bap",)),
        PackageProvenance(
            name="gtirb",
            adapter="gtirb_interchange",
            version=gtirb_version or "not-installed",
            optional=True,
            available=gtirb_version is not None,
            command="python-gtirb",
            details={"role": "optional binary IR interchange adapter"},
        ),
        PackageProvenance(
            name="qiling",
            adapter="qiling_os_emulation",
            version=qiling_version or "not-installed",
            optional=True,
            available=qiling_version is not None,
            command="python-qiling",
            details={"role": "optional OS-aware emulation adapter"},
        ),
        PackageProvenance(
            name="miasm",
            adapter="miasm_python_ir",
            version=miasm_version or "not-installed",
            optional=True,
            available=miasm_version is not None,
            command="python-miasm",
            details={"role": "optional Python-native IR and emulation adapter"},
        ),
        _command_probe("dyninst", "dyninst_instrumentation", ("dyninst", "parseThat")),
        _command_probe(
            "retdec", "retdec_decompiler", ("retdec-decompiler", "retdec-fileinfo")
        ),
        _command_probe("mcsema", "mcsema_lifting", ("mcsema-lift", "mcsema-disass")),
        PackageProvenance(
            name="keystone",
            adapter="keystone_fixture_assembly",
            version=keystone_version or "not-installed",
            optional=True,
            available=keystone_version is not None,
            command="python-keystone",
            details={"role": "optional fixture assembly adapter"},
        ),
        _command_probe("nasm", "nasm_fixture_assembly", ("nasm",)),
    ]
