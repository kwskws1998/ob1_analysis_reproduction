"""Download and verify every external asset used by the standalone analysis."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "asset_manifest.json"
OB1_SENTINELS = (
    "README.md",
    "src/main.py",
    "src/parameters.py",
    "src/reading_components.py",
    "src/reading_helper_functions.py",
    "src/simulate_experiment.py",
    "src/utils.py",
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    """Load the checked-in asset manifest."""
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported asset manifest schema")
    return payload["assets"]


def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> Path:
    """Validate one asset by byte count and SHA-256."""
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    if actual_bytes != int(expected_bytes):
        raise ValueError(
            f"Size mismatch for {path}: expected {expected_bytes}, got {actual_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    return path


def download_file(asset: dict, destination_key: str) -> Path:
    """Resume, validate, and atomically install one downloaded file."""
    destination = ROOT / asset[destination_key]
    if destination.is_file():
        verify_file(destination, asset["bytes"], asset["sha256"])
        print(f"verified {destination.relative_to(ROOT)}", flush=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    if temporary_path.is_file() and temporary_path.stat().st_size == int(
        asset["bytes"]
    ):
        try:
            verify_file(temporary_path, asset["bytes"], asset["sha256"])
            temporary_path.replace(destination)
            print(f"recovered {destination.relative_to(ROOT)}", flush=True)
            return destination
        except ValueError:
            temporary_path.unlink()
    if temporary_path.is_file() and temporary_path.stat().st_size > int(asset["bytes"]):
        temporary_path.unlink()
    resume_at = temporary_path.stat().st_size if temporary_path.is_file() else 0
    headers = {"User-Agent": "ob1-analysis-reproduction/1.0"}
    if resume_at:
        headers["Range"] = f"bytes={resume_at}-"
    request = urllib.request.Request(asset["source"], headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            status = getattr(response, "status", None)
            append = bool(resume_at and status == 206)
            mode = "ab" if append else "wb"
            with temporary_path.open(mode) as temporary:
                shutil.copyfileobj(response, temporary)
        verify_file(temporary_path, asset["bytes"], asset["sha256"])
        temporary_path.replace(destination)
    except BaseException:
        if temporary_path.is_file() and temporary_path.stat().st_size >= int(
            asset["bytes"]
        ):
            temporary_path.unlink(missing_ok=True)
        raise
    print(f"downloaded {destination.relative_to(ROOT)}", flush=True)
    return destination


def _validated_tar_root(archive: tarfile.TarFile) -> str:
    """Reject unsafe members and return the archive's single root directory."""
    roots: set[str] = set()
    for member in archive.getmembers():
        member_path = PurePosixPath(member.name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe archive member: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"Archive links are not allowed: {member.name}")
        if member_path.parts:
            roots.add(member_path.parts[0])
    if len(roots) != 1:
        raise ValueError(f"Expected one archive root, found {sorted(roots)}")
    return roots.pop()


def _ob1_tree_digest(destination: Path) -> tuple[int, str]:
    """Hash every relative filename and byte payload in the OB1 tree."""
    digest = hashlib.sha256()
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(destination).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return len(files), digest.hexdigest()


def _verify_ob1_tree(destination: Path, asset: dict) -> None:
    """Require the complete byte-exact pinned OB1 source tree."""
    missing = [name for name in OB1_SENTINELS if not (destination / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete OB1 source tree: {missing}")
    file_count, tree_sha256 = _ob1_tree_digest(destination)
    if file_count != int(asset["tree_files"]) or tree_sha256 != asset["tree_sha256"]:
        raise ValueError(
            "Pinned OB1 source-tree checksum mismatch: "
            f"files={file_count}, sha256={tree_sha256}"
        )


def extract_ob1(archive_path: Path, asset: dict) -> Path:
    """Safely extract the pinned OB1 source tree once."""
    destination = ROOT / asset["extract_destination"]
    if destination.exists():
        _verify_ob1_tree(destination, asset)
        print(f"verified {destination.relative_to(ROOT)}", flush=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent,
        prefix=".ob1_extract_",
    ) as temporary_dir:
        temporary_root = Path(temporary_dir)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive_root = _validated_tar_root(archive)
            archive.extractall(temporary_root, filter="data")
        extracted = temporary_root / archive_root
        _verify_ob1_tree(extracted, asset)
        extracted.replace(destination)
    print(f"extracted {destination.relative_to(ROOT)}", flush=True)
    return destination


def extract_subtlex(archive_path: Path, asset: dict) -> Path:
    """Extract and verify the single SUBTLEX-UK text member atomically."""
    destination = ROOT / asset["extract_destination"]
    if destination.is_file():
        verify_file(
            destination,
            asset["extracted_bytes"],
            asset["extracted_sha256"],
        )
        print(f"verified {destination.relative_to(ROOT)}", flush=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_member = asset["archive_member"]
    temporary_path: Path | None = None
    try:
        with zipfile.ZipFile(archive_path) as archive:
            if archive.namelist() != [expected_member]:
                raise ValueError(
                    f"Unexpected SUBTLEX archive members: {archive.namelist()}"
                )
            member_path = PurePosixPath(expected_member)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe ZIP member: {expected_member}")
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".part",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                with archive.open(expected_member) as source:
                    shutil.copyfileobj(source, temporary)
        verify_file(
            temporary_path,
            asset["extracted_bytes"],
            asset["extracted_sha256"],
        )
        temporary_path.replace(destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    print(f"extracted {destination.relative_to(ROOT)}", flush=True)
    return destination


def ensure_tokenizer(asset: dict, allow_download: bool = True) -> Path:
    """Verify pinned tokenizer files and their combined behavior fingerprint."""
    from transformers import AutoTokenizer

    from .tokenizer_fingerprint import tokenizer_fingerprint

    destination = ROOT / "models/t5_tokenizer"
    if allow_download:
        destination.mkdir(parents=True, exist_ok=True)
    elif not destination.is_dir():
        raise FileNotFoundError(destination)
    for file_asset in asset["files"]:
        if allow_download:
            download_file(file_asset, "destination")
        else:
            verify_file(
                ROOT / file_asset["destination"],
                file_asset["bytes"],
                file_asset["sha256"],
            )
    digest = hashlib.sha256()
    total_bytes = 0
    for file_asset in sorted(asset["files"], key=lambda item: item["destination"]):
        path = ROOT / file_asset["destination"]
        digest.update(path.name.encode("utf-8") + b"\0")
        total_bytes += path.stat().st_size
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    if (
        total_bytes != int(asset["bytes"])
        or digest.hexdigest() != asset["files_sha256"]
    ):
        raise ValueError("Combined ET1 tokenizer checksum mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        str(destination),
        local_files_only=True,
        model_max_length=2048,
    )
    fingerprint = tokenizer_fingerprint(tokenizer)
    if fingerprint != asset["fingerprint"]:
        raise ValueError(
            "ET1 tokenizer fingerprint mismatch: "
            f"expected {asset['fingerprint']}, got {fingerprint}"
        )
    print(f"verified {destination.relative_to(ROOT)}", flush=True)
    return destination


def download_all() -> dict:
    """Acquire and validate the exact external inputs for the experiment."""
    assets = load_manifest()
    eye = download_file(assets["provo_eye_tracking"], "destination")
    predictability = download_file(assets["provo_predictability"], "destination")
    ob1_archive = download_file(assets["ob1_provo_2024"], "archive_destination")
    ob1_source = extract_ob1(ob1_archive, assets["ob1_provo_2024"])
    subtlex_archive = download_file(assets["subtlex_uk"], "archive_destination")
    subtlex = extract_subtlex(subtlex_archive, assets["subtlex_uk"])
    et1 = download_file(assets["et1_checkpoint"], "destination")
    tokenizer = ensure_tokenizer(assets["et1_tokenizer"])
    return {
        "provo_eye_tracking": str(eye),
        "provo_predictability": str(predictability),
        "ob1_source": str(ob1_source),
        "subtlex_uk": str(subtlex),
        "et1_checkpoint": str(et1),
        "et1_tokenizer": str(tokenizer),
    }


def verify_all() -> dict:
    """Verify all downloaded files and extracted inputs without network access."""
    assets = load_manifest()
    verified = {}
    for name in ("provo_eye_tracking", "provo_predictability", "et1_checkpoint"):
        asset = assets[name]
        path = verify_file(
            ROOT / asset["destination"],
            asset["bytes"],
            asset["sha256"],
        )
        verified[name] = str(path)
    ob1_asset = assets["ob1_provo_2024"]
    verify_file(
        ROOT / ob1_asset["archive_destination"],
        ob1_asset["bytes"],
        ob1_asset["sha256"],
    )
    ob1_source = ROOT / ob1_asset["extract_destination"]
    _verify_ob1_tree(ob1_source, ob1_asset)
    verified["ob1_source"] = str(ob1_source)
    subtlex_asset = assets["subtlex_uk"]
    verify_file(
        ROOT / subtlex_asset["archive_destination"],
        subtlex_asset["bytes"],
        subtlex_asset["sha256"],
    )
    subtlex = verify_file(
        ROOT / subtlex_asset["extract_destination"],
        subtlex_asset["extracted_bytes"],
        subtlex_asset["extracted_sha256"],
    )
    verified["subtlex_uk"] = str(subtlex)
    verified["et1_tokenizer"] = str(
        ensure_tokenizer(assets["et1_tokenizer"], allow_download=False)
    )
    return verified
