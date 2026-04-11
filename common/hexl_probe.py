from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


def cpu_supports_avx512() -> bool:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            return "avx512" in handle.read().lower()
    except Exception:
        return False


def find_tenseal_binaries() -> list[Path]:
    """Find candidate native TenSEAL/SEAL binaries for linkage inspection."""
    paths = []
    spec = importlib.util.find_spec("tenseal")
    if spec and spec.origin:
        pkg_dir = Path(spec.origin).resolve().parent
        site_dir = pkg_dir.parent
        candidates = [
            site_dir / "libtenseal.so",
            site_dir / "_sealapi_cpp.cpython-313-x86_64-linux-gnu.so",
            site_dir / "_tenseal_cpp.cpython-313-x86_64-linux-gnu.so",
        ]
        paths.extend(path for path in candidates if path.exists())
        paths.extend(sorted(site_dir.glob("_sealapi_cpp*.so")))
        paths.extend(sorted(site_dir.glob("_tenseal_cpp*.so")))

    deduped = []
    seen = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def binary_mentions_hexl(binary_path: Path) -> bool:
    """Probe a native binary for signs that Intel HEXL is linked or embedded."""
    commands = [
        ["ldd", str(binary_path)],
        ["strings", str(binary_path)],
    ]
    for command in commands:
        try:
            proc = subprocess.run(command, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            continue
        haystack = (proc.stdout + "\n" + proc.stderr).lower()
        if "hexl" in haystack or "seal_use_intel_hexl" in haystack:
            return True
    return False


def probe_hexl_linkage() -> dict[str, object]:
    binaries = find_tenseal_binaries()
    linked = [str(path) for path in binaries if binary_mentions_hexl(path)]
    return {
        "avx512_detected": cpu_supports_avx512(),
        "probed_binaries": [str(path) for path in binaries],
        "linked_binaries": linked,
        "hexl_linked": bool(linked),
    }
