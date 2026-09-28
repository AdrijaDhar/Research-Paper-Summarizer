"""Extract citation/reference metadata from a PDF via a local GROBID service.

Requires GROBID running in Docker first — see lessons.md Phase 1 entry for the
`docker run` command. Uses `grobid_client_python`'s single-document API
(`process_pdf`, not the batch `process` method) so the TEI-XML comes back as a
string we parse directly, instead of being written to disk.
"""

from pathlib import Path

from grobid_client.grobid_client import GrobidClient
from lxml import etree

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "grobid.json"


def _text(el) -> str | None:
    return el.text.strip() if el is not None and el.text else None


def _parse_references(tei_xml: str) -> list[dict]:
    root = etree.fromstring(tei_xml.encode("utf-8"))
    references = []
    for bibl in root.findall(".//tei:listBibl/tei:biblStruct", TEI_NS):
        title = _text(bibl.find(".//tei:title", TEI_NS))
        authors = []
        for pers in bibl.findall(".//tei:author/tei:persName", TEI_NS):
            forename = _text(pers.find("tei:forename", TEI_NS))
            surname = _text(pers.find("tei:surname", TEI_NS))
            name = " ".join(part for part in (forename, surname) if part)
            if name:
                authors.append(name)
        year_el = bibl.find(".//tei:date", TEI_NS)
        year = year_el.get("when") if year_el is not None else None
        doi_el = bibl.find(".//tei:idno[@type='DOI']", TEI_NS)
        doi = _text(doi_el)

        references.append(
            {
                "xml_id": bibl.get("{http://www.w3.org/XML/1998/namespace}id"),
                "title": title,
                "authors": authors,
                "year": year,
                "doi": doi,
            }
        )
    return references


def extract_references(pdf_path: Path) -> list[dict]:
    client = GrobidClient(config_path=str(CONFIG_PATH))
    _name, status, tei_xml = client.process_pdf(
        "processFulltextDocument",
        str(pdf_path),
        consolidate_header=True,
        consolidate_citations=True,
        include_raw_citations=False,
        include_raw_affiliations=False,
        tei_coordinates=False,
        segment_sentences=False,
    )
    if status != 200:
        raise RuntimeError(f"GROBID returned status {status} for {pdf_path}")
    return _parse_references(tei_xml)
