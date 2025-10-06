"""Utilities for loading DOCDB exchange documents into Elasticsearch."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
try:
    from lxml import etree as ET  # type: ignore[assignment]
except ImportError:  # pragma: no cover - lxml optional
    from xml.etree import ElementTree as ET  # type: ignore[assignment]

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import ApiError

load_dotenv()

NS = {"exch": "http://www.epo.org/exchange"}


def get_client() -> Elasticsearch:
    url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    api_key = os.getenv("ELASTICSEARCH_API_KEY")
    api_key_id = os.getenv("ELASTICSEARCH_API_KEY_ID")

    if api_key_id and api_key:
        return Elasticsearch(url, api_key=(api_key_id, api_key))
    if api_key:
        return Elasticsearch(url, api_key=api_key)
    return Elasticsearch(url)


def text_or_none(element: Optional[ET.Element]) -> Optional[str]:
    if element is None or element.text is None:
        return None
    return element.text.strip() or None


def build_document_id(parts: Iterable[Optional[str]]) -> Optional[str]:
    values = [part for part in parts if part]
    if not values:
        return None
    return "-".join(values)


def parse_publication_reference(doc: ET.Element) -> Dict[str, Any]:
    node = doc.find(
        "exch:bibliographic-data/exch:publication-reference[@data-format='docdb']/document-id",
        NS,
    )
    if node is None:
        return {}

    country = text_or_none(node.find("exch:country", NS))
    doc_number = text_or_none(node.find("exch:doc-number", NS))
    kind = text_or_none(node.find("exch:kind", NS))
    date = text_or_none(node.find("exch:date", NS))

    return {
        "country": country,
        "doc_number": doc_number,
        "kind": kind,
        "date": date,
        "document_id": build_document_id((country, doc_number, kind)),
    }


def parse_application_reference(doc: ET.Element) -> Dict[str, Any]:
    node = doc.find(
        "exch:bibliographic-data/exch:application-reference[@data-format='docdb']/document-id",
        NS,
    )
    if node is None:
        return {}

    return {
        "country": text_or_none(node.find("exch:country", NS)),
        "doc_number": text_or_none(node.find("exch:doc-number", NS)),
        "kind": text_or_none(node.find("exch:kind", NS)),
        "date": text_or_none(node.find("exch:date", NS)),
    }


def parse_priority_claims(doc: ET.Element) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    for claim in doc.findall(
        "exch:bibliographic-data/exch:priority-claims/exch:priority-claim[@data-format='docdb']",
        NS,
    ):
        sequence = claim.attrib.get("sequence")
        doc_id = claim.find("document-id", NS)
        claims.append(
            {
                "sequence": int(sequence) if sequence else None,
                "country": text_or_none(doc_id.find("exch:country", NS))
                if doc_id
                else None,
                "doc_number": text_or_none(doc_id.find("exch:doc-number", NS))
                if doc_id
                else None,
                "kind": text_or_none(doc_id.find("exch:kind", NS)) if doc_id else None,
                "date": text_or_none(doc_id.find("exch:date", NS)) if doc_id else None,
            }
        )
    return [
        claim for claim in claims if any(value is not None for value in claim.values())
    ]


IPC_SYMBOL_PATTERN = re.compile(r"([A-H][0-9]{2}[A-Z]\s*\d+/?\d*)")


def parse_ipc_classifications(doc: ET.Element) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for classification in doc.findall(
        "exch:bibliographic-data/exch:classifications-ipcr/exch:classification-ipcr",
        NS,
    ):
        raw_text = text_or_none(classification.find("exch:text", NS))
        if not raw_text:
            continue
        sequence = classification.attrib.get("sequence")
        symbol_match = IPC_SYMBOL_PATTERN.search(raw_text)
        symbol = symbol_match.group(1).replace(" ", "") if symbol_match else raw_text
        version_match = re.search(r"(\d{8})", raw_text)
        results.append(
            {
                "level": "primary" if sequence == "1" else "additional",
                "symbol": symbol,
                "version": version_match.group(1) if version_match else None,
                "is_primary": sequence == "1",
            }
        )
    return results


def parse_cpc_classifications(doc: ET.Element) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for classification in doc.findall(
        "exch:bibliographic-data/exch:patent-classifications/exch:patent-classification",
        NS,
    ):
        symbol = text_or_none(classification.find("exch:classification-symbol", NS))
        if not symbol:
            continue
        version = text_or_none(
            classification.find("exch:classification-scheme/exch:date", NS)
        )
        is_inventive = (
            text_or_none(classification.find("exch:classification-value", NS)) == "I"
        )
        position = text_or_none(classification.find("exch:symbol-position", NS))
        results.append(
            {
                "position": position,
                "symbol": symbol.strip(),
                "version": version,
                "is_inventive": is_inventive,
            }
        )
    return results


def parse_party_entries(parent: ET.Element, tag: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for party in parent.findall(
        f"exch:{tag}/exch:{tag[:-1]}[@data-format='docdb']", NS
    ):
        sequence_raw = party.attrib.get("sequence")
        name = (
            text_or_none(party.find("exch:applicant-name/exch:name", NS))
            or text_or_none(party.find("exch:inventor-name/exch:name", NS))
            or text_or_none(party.find("exch:assignee-name/exch:name", NS))
            or text_or_none(party.find("exch:agent-name/exch:name", NS))
        )
        if not name:
            continue
        residence_country = text_or_none(party.find("exch:residence/exch:country", NS))
        address = None
        address_node = party.find("exch:address", NS)
        if address_node is not None:
            address_parts = [text_or_none(child) for child in address_node]
            address_values = [part for part in address_parts if part]
            if address_values:
                address = ", ".join(address_values)
        sequence = (
            int(sequence_raw) if sequence_raw and sequence_raw.isdigit() else None
        )
        entries.append(
            {
                "sequence": sequence,
                "name": name,
                "address": address,
                "country": residence_country,
            }
        )
    return entries


def parse_parties(doc: ET.Element) -> Dict[str, Any]:
    parties_node = doc.find("exch:bibliographic-data/exch:parties", NS)
    if parties_node is None:
        return {}
    return {
        "applicants": parse_party_entries(parties_node, "applicants"),
        "inventors": parse_party_entries(parties_node, "inventors"),
        "assignees": parse_party_entries(parties_node, "assignees"),
        "agents": parse_party_entries(parties_node, "agents"),
    }


def parse_titles(doc: ET.Element) -> List[Dict[str, Any]]:
    titles: List[Dict[str, Any]] = []
    for title in doc.findall("exch:bibliographic-data/exch:invention-title", NS):
        text = text_or_none(title)
        if not text:
            continue
        titles.append(
            {
                "lang": title.attrib.get("lang")
                or title.attrib.get("{http://www.w3.org/XML/1998/namespace}lang"),
                "text": text,
            }
        )
    return titles


def parse_abstracts(doc: ET.Element) -> List[Dict[str, Any]]:
    abstracts: List[Dict[str, Any]] = []
    for abstract in doc.findall("exch:abstract", NS):
        paragraphs = [text_or_none(p) for p in abstract.findall("exch:p", NS)]
        values = [p for p in paragraphs if p]
        if not values:
            continue
        combined_text = " ".join(values)
        abstracts.append(
            {
                "lang": abstract.attrib.get("lang")
                or abstract.attrib.get("{http://www.w3.org/XML/1998/namespace}lang"),
                "text": combined_text,
            }
        )
    return abstracts


def build_fulltext(document: Dict[str, Any]) -> Optional[str]:
    parts: List[str] = []

    def add(value: Optional[str]) -> None:
        if not value:
            return
        stripped = value.strip()
        if stripped:
            parts.append(stripped)

    publication_reference = document.get("publication_reference") or {}
    add(publication_reference.get("document_id"))

    for classification in document.get("ipc_classifications", []) or []:
        if isinstance(classification, dict):
            add(classification.get("symbol"))
        elif isinstance(classification, str):
            add(classification)
    for classification in document.get("cpc_classifications", []) or []:
        if isinstance(classification, dict):
            add(classification.get("symbol"))
        elif isinstance(classification, str):
            add(classification)

    parties = document.get("parties") or {}
    for role in ("applicants", "inventors", "assignees", "agents"):
        for entry in parties.get(role, []) or []:
            if isinstance(entry, dict):
                add(entry.get("name"))
                add(entry.get("address"))
            elif isinstance(entry, str):
                add(entry)

    for title in document.get("title", []) or []:
        if isinstance(title, dict):
            add(title.get("text"))
        elif isinstance(title, str):
            add(title)
    for abstract in document.get("abstracts", []) or []:
        if isinstance(abstract, dict):
            add(abstract.get("text"))
        elif isinstance(abstract, str):
            add(abstract)

    for description in document.get("description", []) or []:
        if isinstance(description, dict):
            sections = description.get("sections")
            if isinstance(sections, list):
                for section in sections:
                    add(section)
            else:
                add(sections)
            add(description.get("text"))
        elif isinstance(description, str):
            add(description)

    for claim in document.get("claims", []) or []:
        if isinstance(claim, dict):
            add(claim.get("text"))
        elif isinstance(claim, str):
            add(claim)

    citations = document.get("citations") or {}
    for entry in citations.get("non_patent_citations", []) or []:
        if isinstance(entry, dict):
            add(entry.get("text"))
        elif isinstance(entry, str):
            add(entry)

    for event in document.get("legal_events", []) or []:
        if isinstance(event, dict):
            add(event.get("event_text"))
        elif isinstance(event, str):
            add(event)

    if not parts:
        return None
    return " ".join(parts)


def parse_exchange_document(doc: ET.Element) -> Dict[str, Any]:
    publication_reference = parse_publication_reference(doc)
    application_reference = parse_application_reference(doc)
    priority_claims = parse_priority_claims(doc)
    parties = parse_parties(doc)
    titles = parse_titles(doc)
    abstracts = parse_abstracts(doc)

    document: Dict[str, Any] = {
        "docdb_id": doc.attrib.get("doc-number"),
        "exchange_document_id": doc.attrib.get("doc-id"),
        "docdb_family_id": doc.attrib.get("family-id"),
        "inpadoc_family_id": doc.attrib.get("inpadoc-family-id"),
        "publication_reference": publication_reference,
        "application_reference": application_reference,
        "priority_claims": priority_claims,
        "publication_language": text_or_none(
            doc.find("exch:bibliographic-data/exch:language-of-publication", NS)
        )
        or text_or_none(
            doc.find("exch:bibliographic-data/exch:language-of-filing", NS)
        ),
        "ipc_classifications": parse_ipc_classifications(doc),
        "cpc_classifications": parse_cpc_classifications(doc),
        "parties": parties,
        "citations": {
            "patent_citations": [],
            "non_patent_citations": [],
        },
        "register_events": [],
    }

    if titles:
        document["title"] = titles
    if abstracts:
        document["abstracts"] = abstracts

    fulltext = build_fulltext(document)
    if fulltext:
        document["fulltext"] = fulltext

    return document

def parse_ep_document(doc: ET.Element) -> Dict[str, Any]:
    """Parse an EPO ep-patent-document (NWA1) into a dictionary."""

    def gettext(path: str) -> Optional[str]:
        el = doc.find(path)
        return el.text.strip() if el is not None and el.text else None

    def gettextall(path: str) -> List[str]:
        return [el.text.strip() for el in doc.findall(path) if el is not None and el.text]

    return {
        "exchange_document_id": doc.attrib.get("id"),   # <-- use root @id
        "publication_number": gettext("SDOBI/B100/B110"),
        "kind": gettext("SDOBI/B100/B130"),
        "pub_date": gettext("SDOBI/B100/B140/date"),
        "country": gettext("SDOBI/B100/B190"),

        "application_number": gettext("SDOBI/B200/B210"),
        "application_date": gettext("SDOBI/B200/B220/date"),

        "ipc_classes": gettextall("SDOBI/B500/B510EP/classification-ipcr/text"),
        "cpc_classes": gettextall("SDOBI/B500/B520EP/classifications-cpc/classification-cpc/text"),

        "titles": [
            {"lang": lang.text, "text": title.text}
            for lang, title in zip(
                doc.findall("SDOBI/B500/B540/B541"),
                doc.findall("SDOBI/B500/B540/B542"),
            )
        ],

        "abstracts": [
            {
                "lang": abs.attrib.get("lang"),
                "text": " ".join(p.text.strip() for p in abs.findall("p") if p.text),
            }
            for abs in doc.findall(".//abstract")
        ],

        # "claims": [
        #     {
        #         "num": cl.attrib.get("num"),
        #         "text": " ".join(ct.text.strip() for ct in cl.findall(".//claim-text") if ct.text),
        #     }
        #     for cl in doc.findall(".//claims/claim")
        # ],
        "claims": [
            {
                "num": cl.attrib.get("num"),
                "lang": claims.attrib.get("lang"),
                "text": " ".join(
                    ct.text.strip() for ct in cl.findall(".//claim-text") if ct.text
                ),
            }
            for claims in doc.findall(".//claims")
            for cl in claims.findall("claim")
        ],


        "applicants": [
            {
                "name": appl.findtext("snm"),
                "country": appl.findtext("adr/ctry"),
                "city": appl.findtext("adr/city"),
            }
            for appl in doc.findall("SDOBI/B700/B710/B711")
        ],

        "inventors": [
            {
                "name": inv.findtext("snm"),
                "country": inv.findtext("adr/ctry"),
                "city": inv.findtext("adr/city"),
            }
            for inv in doc.findall("SDOBI/B700/B720/B721")
        ],

        "designated_states": gettextall("SDOBI/B800/B840/ctry"),
        "extension_states": gettextall("SDOBI/B800/B844EP/B845EP/ctry"),
        "validation_states": gettextall("SDOBI/B800/B848EP/B849EP/ctry"),

        "priority_claims": [
            {
                "app_number": prio.findtext("dnum/anum"),
                "date": prio.findtext("date"),
                "lang": prio.findtext("../B862"),
            }
            for prio in doc.findall("SDOBI/B800/B860/B861")
        ],

        "pct_refs": [
            {
                "pub_number": pct.findtext("dnum/pnum"),
                "date": pct.findtext("date"),
                "bnum": pct.findtext("bnum"),
            }
            for pct in doc.findall("SDOBI/B800/B870/B871")
        ],
    }


def parse_exchange_documents(
    xml_path: Path, parser: Optional[ET.XMLParser] | None = None
) -> List[Dict[str, Any]]:
    if parser is not None:
        tree = ET.parse(str(xml_path), parser)
    else:
        tree = ET.parse(str(xml_path))
    root = tree.getroot()
    documents = []
    # DOCDB-style
    for doc in root.findall("exch:exchange-document", NS):
        documents.append(parse_exchange_document(doc))
    
    # EP NWA-style
    if root.tag.endswith("ep-patent-document"):
        documents.append(parse_ep_document(root))


    return documents


def ingest_documents(
    es: Elasticsearch, documents: List[Dict[str, Any]], index: str
) -> None:
    if not documents:
        print("No documents found to ingest.")
        return

    actions = []
    for doc in documents:
        doc_id = doc.get("exchange_document_id") or doc.get("docdb_id")
        if not doc_id:
            raise SystemExit("Each document requires an identifier for ingestion.")
        actions.append(
            {
                "_op_type": "index",
                "_index": index,
                "_id": doc_id,
                "_source": doc,
            }
        )

    try:
        helpers.bulk(es, actions)
    except ApiError as exc:
        print(f"Failed to ingest documents into index '{index}': {exc}")
        raise SystemExit(1) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load DOCDB exchange documents from XML into Elasticsearch."
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path("sample.xml"),
        help="Path to the DOCDB exchange XML file (default: sample.xml).",
    )
    parser.add_argument(
        "--index",
        default="patents",
        help="Elasticsearch index name to ingest into (default: patents).",
    )

    args = parser.parse_args()
    documents = parse_exchange_documents(args.xml)
    es = get_client()
    ingest_documents(es, documents, args.index)
    print(f"Ingested {len(documents)} documents into index '{args.index}'.")


if __name__ == "__main__":
    main()
