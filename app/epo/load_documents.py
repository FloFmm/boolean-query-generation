"""Bulk loader for DOCDB exchange archives."""

from __future__ import annotations

import argparse
import html
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Dict, Iterator, Optional

from xml.etree import ElementTree as StdET

try:
    from lxml import etree as LET  # type: ignore[assignment]
except ImportError:  # pragma: no cover - optional dependency
    LET = None

from dotenv import load_dotenv

from elasticsearch import Elasticsearch

from sample_loader import ingest_documents, parse_exchange_documents, get_client

load_dotenv()


ARCHIVE_PATTERN = "EPRTBJV*.zip"#docdb_xml_bck_*"
ENTITY_DTD_NAME = "ep-patent-document-v1-7.dtd"#"docdb-entities.dtd"
ENTITY_DECLARATION_PATTERN = re.compile(
    r"<!ENTITY\s+(?P<name>[A-Za-z0-9._-]+)\s+\"(?P<value>[^\"]*)\""
)


@dataclass
class ProcessingStats:
    root_archives: int = 0
    nested_archives: int = 0
    xml_files: int = 0
    documents: int = 0

    def merge(self, other: "ProcessingStats") -> None:
        self.root_archives += other.root_archives
        self.nested_archives += other.nested_archives
        self.xml_files += other.xml_files
        self.documents += other.documents


def discover_root_archives(root_dir: Path) -> Iterator[Path]:
    for path in sorted(root_dir.glob(ARCHIVE_PATTERN)):
        if path.is_file():
            yield path


def find_doc_directory(extracted_root: Path) -> Optional[Path]:
    direct = extracted_root / "DOC"
    if direct.is_dir():
        return direct
    for candidate in extracted_root.rglob("DOC"):
        if candidate.is_dir() and candidate.name.upper() == "DOC":
            return candidate
    return None


def find_dtd_directory(extracted_root: Path) -> Optional[Path]:
    direct = extracted_root / "DTDS"
    if direct.is_dir():
        return direct
    for candidate in extracted_root.rglob("DTDS"):
        if candidate.is_dir() and candidate.name.upper() == "DTDS":
            return candidate
    return None


def extract_xml_to_temp(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo, temp_dir: Path
) -> Path:
    temp_file = NamedTemporaryFile(delete=False, dir=temp_dir, suffix=".xml")
    try:
        with archive.open(member) as source:
            shutil.copyfileobj(source, temp_file)
    finally:
        temp_file.close()
    return Path(temp_file.name)


@dataclass
class DTDResources:
    entities: Dict[str, str]
    dtd_paths: Dict[str, Path]


if LET is not None:  # pragma: no cover - requires lxml runtime

    class LocalResolver(LET.Resolver):
        """Resolve DTD lookups from the extracted DTDS directory."""

        def __init__(self, base_dir: Path, dtd_paths: Dict[str, Path]):
            super().__init__()
            self.base_dir = base_dir
            self.dtd_paths = dtd_paths

        def resolve(self, system_url: str, public_id: str, context):  # type: ignore[override]
            candidate = (self.base_dir / system_url).resolve()
            if candidate.is_file():
                return self.resolve_filename(str(candidate), context)

            lookup = self.dtd_paths.get(Path(system_url).name)
            if lookup and lookup.is_file():
                return self.resolve_filename(str(lookup), context)

            return None


def create_parser(resources: DTDResources, base_dir: Path):
    if LET is not None:  # pragma: no cover - requires lxml runtime
        parser = LET.XMLParser(
            load_dtd=True,
            resolve_entities=True,
            no_network=True,
            huge_tree=True,
        )
        parser.resolvers.add(LocalResolver(base_dir, resources.dtd_paths))
        print(
            f"Creating lxml parser with {len(resources.dtd_paths)} DTDs and "
            f"{len(resources.entities)} entity overrides."
        )
        return parser

    if not resources.entities:
        print(
            "No entities available and lxml unavailable; falling back to default parser."
        )
        return None
    parser = StdET.XMLParser()
    for name, value in resources.entities.items():
        parser.entity[name] = value
    print(
        f"Configured stdlib XMLParser with {len(resources.entities)} entity overrides."
    )
    return parser


def process_xml_file(
    es: Elasticsearch,
    xml_path: Path,
    index: str,
    resources: DTDResources,
    base_dir: Path,
) -> int:
    parser = create_parser(resources, base_dir)
    documents = parse_exchange_documents(xml_path, parser=parser)
    if not documents:
        print(f"    No documents parsed from {xml_path.name}.")
        return 0
    ingest_documents(es, documents, index)
    print(
        f"    Indexed {len(documents)} document(s) from {xml_path.name} into '{index}'."
    )
    return len(documents)


def process_nested_archive(
    es: Elasticsearch,
    archive_path: Path,
    temp_dir: Path,
    index: str,
    resources: DTDResources,
) -> ProcessingStats:
    stats = ProcessingStats(nested_archives=1)
    print(f"  Processing nested archive {archive_path.name}...")
    try:
        with zipfile.ZipFile(archive_path) as nested_zip:
            xml_members = [
                member
                for member in nested_zip.infolist()
                if member.filename.lower().endswith(".xml") and not member.is_dir()
                and not member.filename.lower().endswith("toc.xml")  # skip TOC files
            ]
            if not xml_members:
                print(f"  No XML members found inside {archive_path.name}.")
                return stats

            for member in xml_members:
                xml_file = extract_xml_to_temp(nested_zip, member, temp_dir)
                try:
                    print(f"    Parsed XML member {member.filename}")
                    stats.xml_files += 1
                    stats.documents += process_xml_file(
                        es,
                        xml_file,
                        index,
                        resources,
                        base_dir=temp_dir,
                    )
                finally:
                    xml_file.unlink(missing_ok=True)
    except zipfile.BadZipFile:
        print(f"  Warning: {archive_path.name} is not a valid ZIP file.")
        pass
    return stats


def scan_dtd_resources(extracted_root: Path) -> DTDResources:
    dtd_dir = find_dtd_directory(extracted_root)
    if not dtd_dir or not dtd_dir.is_dir():
        return DTDResources(entities={}, dtd_paths={})

    entities: Dict[str, str] = {}
    dtd_paths: Dict[str, Path] = {}
    entity_source: Optional[Path] = None

    for path in dtd_dir.rglob("*.dtd"):
        dtd_paths[path.name] = path
        if path.name == ENTITY_DTD_NAME and entity_source is None:
            entity_source = path

    print(
        f"Loaded DTD directory '{dtd_dir}' with {len(dtd_paths)} DTD file(s); ",
        f"entity source: {entity_source}",
    )

    if entity_source is not None:
        try:
            content = entity_source.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            content = ""

        for match in ENTITY_DECLARATION_PATTERN.finditer(content):
            name = match.group("name")
            raw_value = match.group("value")
            if not name:
                continue
            entities[name] = html.unescape(raw_value)

    if entities:
        sample = ", ".join(list(entities.keys())[:5])
        print(f"Registered {len(entities)} XML entity mappings (sample: {sample})")
    else:
        print("No XML entities registered from DTDs.")

    return DTDResources(entities=entities, dtd_paths=dtd_paths)


def process_root_archive(
    es: Elasticsearch, archive_path: Path, index: str
) -> ProcessingStats:
    stats = ProcessingStats(root_archives=1)
    with TemporaryDirectory(prefix=f"{archive_path.stem}_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        try:
            with zipfile.ZipFile(archive_path) as root_zip:
                root_zip.extractall(temp_dir)
        except zipfile.BadZipFile:
            return stats

        doc_dir = find_doc_directory(temp_dir)
        if doc_dir is None:
            return stats

        dtd_resources = scan_dtd_resources(temp_dir)
        if not dtd_resources.dtd_paths:
            print(
                f"Warning: no DTD files discovered for archive {archive_path.name}; "
                "entity expansion may fail."
            )

        nested_archives = sorted(
            path for path in doc_dir.rglob("*.zip") if path.is_file()
        )

        for nested_archive in nested_archives:
            stats.merge(
                process_nested_archive(
                    es, nested_archive, temp_dir, index, dtd_resources
                )
            )

    return stats


def run_loader(
    root_dir: Path, index: str, es: Optional[Elasticsearch] = None
) -> ProcessingStats:
    client = es or get_client()
    aggregate = ProcessingStats()
    for archive_path in discover_root_archives(root_dir):
        print(f"Processing root archive {archive_path.name}...")
        archive_stats = process_root_archive(client, archive_path, index)
        print(
            f"Completed {archive_path.name}: {archive_stats.documents} document(s) "
            f"from {archive_stats.xml_files} XML file(s)."
        )
        aggregate.merge(archive_stats)
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load DOCDB exchange archives into Elasticsearch."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing docdb_xml_bck_* root archives.",
    )
    parser.add_argument(
        "--index",
        default="patents",
        help="Target Elasticsearch index (default: patents).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir: Path = args.root.expanduser()
    if not root_dir.exists():
        raise SystemExit(f"Root directory '{root_dir}' does not exist.")

    client = get_client()
    stats = run_loader(root_dir, args.index, es=client)
    print(
        f"Processed {stats.root_archives} archive(s), {stats.nested_archives} nested archive(s), "
        f"{stats.xml_files} XML file(s), {stats.documents} documents indexed into '{args.index}'."
    )


if __name__ == "__main__":
    main()
