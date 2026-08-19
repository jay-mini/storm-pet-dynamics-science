from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "release" / "public_manifest.json"
FORBIDDEN_PARTS = {"web", "private", "incoming", "integration", "node_modules"}
FORBIDDEN_SUFFIXES = {".pickle", ".pkl", ".joblib", ".pt", ".pth"}
FORBIDDEN_IDENTIFIERS = {"PTID", "RID", "LONIUID", "DOB"}
ABSOLUTE_WINDOWS_PATH = re.compile(r"[A-Za-z]:[\\/](?:Users|Documents|Download)[\\/]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_output(value: Path) -> Path:
    output = value.resolve()
    generated_root = (ROOT / "release" / "generated").resolve()
    if output == generated_root or generated_root not in output.parents:
        raise ValueError("public release output must be below release/generated/")
    return output


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _scan(output: Path) -> list[str]:
    problems: list[str] = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output)
        lowered_parts = {part.casefold() for part in relative.parts}
        if lowered_parts.intersection(FORBIDDEN_PARTS):
            problems.append(f"forbidden path: {relative.as_posix()}")
        if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden artifact: {relative.as_posix()}")
        if path.suffix.casefold() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                header = set(next(csv.reader(stream), []))
            leaked = sorted(header.intersection(FORBIDDEN_IDENTIFIERS))
            if leaked:
                problems.append(f"participant identifiers in {relative.as_posix()}: {leaked}")
        if path.suffix.casefold() in {".py", ".md", ".toml", ".yaml", ".yml", ".json"}:
            text = path.read_text(encoding="utf-8")
            if ABSOLUTE_WINDOWS_PATH.search(text):
                problems.append(f"absolute local path in {relative.as_posix()}")
    return problems


def build(manifest_path: Path, output_override: Path | None = None) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported public manifest schema")
    configured = ROOT / str(manifest["output_directory"])
    output = _validated_output(output_override or configured)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    copied: set[Path] = set()
    for pattern in manifest["include"]:
        matches = sorted(path for path in ROOT.glob(pattern) if path.is_file())
        if not matches:
            raise FileNotFoundError(f"public allowlist pattern matched nothing: {pattern}")
        for source in matches:
            relative = source.relative_to(ROOT)
            _copy(source, output / relative)
            copied.add(relative)
    for source_name, destination_name in manifest["templates"].items():
        _copy(ROOT / source_name, output / destination_name)
        copied.add(Path(destination_name))
    for source_name, destination_name in manifest["renames"].items():
        _copy(ROOT / source_name, output / destination_name)
        copied.add(Path(destination_name))

    for config_name in ["abeta_paper.yaml", "tau_paper.yaml"]:
        path = output / "configs" / "ot_cfm" / config_name
        text = path.read_text(encoding="utf-8")
        text = text.replace("data/processed/private/", "data/authorized/")
        text = text.replace("artifacts/runs/", "outputs/")
        path.write_text(text, encoding="utf-8", newline="\n")

    problems = _scan(output)
    if problems:
        raise RuntimeError("public release safety scan failed:\n- " + "\n- ".join(problems))

    files = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        files[relative] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    release_manifest = {
        "schema_version": 1,
        "publication_ready": False,
        "publication_blocker": "Choose and add a source-code LICENSE.",
        "file_count": len(files),
        "files": files,
    }
    (output / "PUBLIC_RELEASE_MANIFEST.json").write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the allowlisted scientific release tree.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = build(args.manifest.resolve(), args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
