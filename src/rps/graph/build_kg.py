"""Build a heterogeneous knowledge graph from Phase 1's parsed outputs.

Node types: paper, section, sentence, entity, table, figure, citation
Edge types: has_section, has_sentence, precedes, mentions, has_table,
has_figure, cites

Schema verified against a real parsed paper's <id>.json (not documentation) —
data/processed/2501.17887/2501.17887.json was inspected directly. Docling's
document tree turns out to be a flat, reading-order sequence in
`body.children` (a list of {"$ref": "#/texts/N"} pointers into the `texts`
/ `tables` / `pictures` collections), not a nested per-section tree — section
membership is inferred here by tracking the most recent `section_header` item
seen while walking that sequence. Items with `content_layer == "furniture"`
(page headers/footers, arXiv watermark line) are skipped.

Entity nodes come from noun-chunk extraction (spaCy's dependency parser), not
named-entity recognition — generic NER (tuned for people/orgs/places) misses
exactly the kind of concept a scientific paper is full of ("table structure
recognition", "layout analysis model"); noun chunks capture those directly.
"""

import json
from pathlib import Path

import networkx as nx
import spacy

MIN_CHUNK_WORDS = 2
MAX_CHUNK_WORDS = 6

SENTENCE_LABELS = {"text", "paragraph", "list_item"}


def _load_json(path: Path) -> list | dict:
    return json.loads(path.read_text())


def _index_by_ref(doc: dict) -> dict:
    by_ref = {}
    for collection in ("texts", "tables", "pictures", "groups"):
        for i, item in enumerate(doc.get(collection, [])):
            by_ref[f"#/{collection}/{i}"] = (collection, item)
    return by_ref


def _resolve(ref_obj: dict, by_ref: dict):
    return by_ref.get(ref_obj.get("$ref"))


def _host_section_for_page(page_no: int | None, section_pages: dict[str, int]) -> str:
    if page_no is None:
        return "paper"
    candidates = [(pg, sid) for sid, pg in section_pages.items() if pg <= page_no]
    return max(candidates)[1] if candidates else "paper"


def build_graph(paper_dir: Path, arxiv_id: str, nlp: spacy.Language | None = None) -> nx.MultiDiGraph:
    doc = _load_json(paper_dir / f"{arxiv_id}.json")
    refs_path = paper_dir / "references.json"
    figs_path = paper_dir / "figures.json"
    references = _load_json(refs_path) if refs_path.exists() else []
    figures = _load_json(figs_path) if figs_path.exists() else []

    nlp = nlp or spacy.load("en_core_web_sm")
    by_ref = _index_by_ref(doc)

    g = nx.MultiDiGraph()
    g.add_node("paper", type="paper", title=doc.get("name", arxiv_id))

    current_section = "paper"
    section_pages: dict[str, int] = {}
    section_count = 0
    sentence_count = 0
    entity_ids: dict[str, str] = {}
    prev_sentence_id = None
    skipped_labels: dict[str, int] = {}
    vlm_pictures: list[dict] = []

    for ref in doc.get("body", {}).get("children", []):
        resolved = _resolve(ref, by_ref)
        if resolved is None:
            continue
        collection, item = resolved

        if item.get("content_layer") == "furniture":
            continue

        if collection == "texts" and item.get("label") == "section_header":
            section_count += 1
            section_id = f"section-{section_count}"
            page_no = item["prov"][0]["page_no"] if item.get("prov") else None
            g.add_node(
                section_id, type="section", title=item.get("text", ""),
                level=item.get("level"), page_no=page_no,
            )
            g.add_edge("paper", section_id, type="has_section")
            current_section = section_id
            if page_no is not None:
                section_pages[section_id] = page_no
            prev_sentence_id = None
            continue

        if collection == "texts" and item.get("label") in SENTENCE_LABELS:
            text = item.get("text", "").strip()
            if not text:
                continue
            spacy_doc = nlp(text)
            for sent in spacy_doc.sents:
                sentence_count += 1
                sentence_id = f"sentence-{sentence_count}"
                g.add_node(sentence_id, type="sentence", text=sent.text)
                g.add_edge(current_section, sentence_id, type="has_sentence")
                if prev_sentence_id is not None:
                    g.add_edge(prev_sentence_id, sentence_id, type="precedes")
                prev_sentence_id = sentence_id

                for chunk in spacy_doc.noun_chunks:
                    if chunk.start < sent.start or chunk.end > sent.end:
                        continue
                    words = [t.text for t in chunk if not t.is_stop and not t.is_punct]
                    if not (MIN_CHUNK_WORDS <= len(words) <= MAX_CHUNK_WORDS):
                        continue
                    key = chunk.text.lower().strip()
                    if key not in entity_ids:
                        entity_ids[key] = f"entity-{len(entity_ids)}"
                        g.add_node(entity_ids[key], type="entity", text=key)
                    g.add_edge(sentence_id, entity_ids[key], type="mentions")
            continue

        if collection == "tables":
            table_id = f"table-{item['self_ref'].rsplit('/', 1)[-1]}"
            page_no = item["prov"][0]["page_no"] if item.get("prov") else None
            caption = ""
            for cap_ref in item.get("captions", []):
                cap = _resolve(cap_ref, by_ref)
                if cap:
                    caption = cap[1].get("text", "")
            g.add_node(table_id, type="table", caption=caption, page_no=page_no)
            g.add_edge(current_section, table_id, type="has_table")
            continue

        if collection == "pictures":
            page_no = item["prov"][0]["page_no"] if item.get("prov") else None
            caption = ""
            for cap_ref in item.get("captions", []):
                cap = _resolve(cap_ref, by_ref)
                if cap:
                    caption = cap[1].get("text", "")
            vlm_pictures.append({"page_no": page_no, "original_caption": caption})
            continue

        skipped_labels[item.get("label", collection)] = skipped_labels.get(item.get("label", collection), 0) + 1

    # figures (crops + VLM captions) come from the separate figure-extraction
    # pass (figures.py runs Docling's standard pipeline, not the VLM pipeline
    # this function reads — a different parse of the same PDF, and the two
    # can disagree on how many pictures they detect). Original paper-authored
    # captions come from `vlm_pictures` above instead. Neither list's index
    # can be trusted to line up with the other or with reading order 1:1, so
    # both problems are solved the same way: group by page number and match
    # positionally within each page — exact when counts agree, best-effort
    # (some figures may end up without an original_caption) when they don't.
    warnings: list[str] = []
    if len(vlm_pictures) != len(figures):
        vlm_pages = [p["page_no"] for p in vlm_pictures]
        fig_pages = [f.get("page_no") for f in figures]
        warnings.append(
            f"picture count mismatch: VLM-pipeline JSON has {len(vlm_pictures)} "
            f"pictures (pages {vlm_pages}), figures.json has {len(figures)} "
            f"figures (pages {fig_pages}) — matched by page position below; "
            "some figures may be missing an original_caption, or a real "
            "figure may be missing entirely."
        )

    pictures_by_page: dict[int | None, list[dict]] = {}
    for pic in vlm_pictures:
        pictures_by_page.setdefault(pic["page_no"], []).append(pic)

    figures_by_page: dict[int | None, list[dict]] = {}
    for fig in figures:
        figures_by_page.setdefault(fig.get("page_no"), []).append(fig)

    for page_no, page_figures in figures_by_page.items():
        page_pictures = pictures_by_page.get(page_no, [])
        for i, fig in enumerate(page_figures):
            original_caption = page_pictures[i]["original_caption"] if i < len(page_pictures) else None
            figure_id = f"figure-{fig['index']}"
            g.add_node(
                figure_id, type="figure", caption=fig.get("caption"),
                original_caption=original_caption, page_no=page_no,
            )
            g.add_edge(_host_section_for_page(page_no, section_pages), figure_id, type="has_figure")

    for i, ref in enumerate(references):
        citation_id = f"citation-{i}"
        g.add_node(citation_id, type="citation", **{k: v for k, v in ref.items() if k != "xml_id"})
        g.add_edge("paper", citation_id, type="cites")

    if skipped_labels:
        g.graph["skipped_labels"] = skipped_labels
    if warnings:
        g.graph["warnings"] = warnings

    return g


def save_graph(g: nx.MultiDiGraph, out_path: Path) -> None:
    data = {
        "nodes": [{"id": n, **d} for n, d in g.nodes(data=True)],
        "edges": [{"source": u, "target": v, **d} for u, v, d in g.edges(data=True)],
    }
    out_path.write_text(json.dumps(data, indent=2))


def summarize(g: nx.MultiDiGraph) -> dict:
    node_counts: dict[str, int] = {}
    for _, data in g.nodes(data=True):
        node_counts[data["type"]] = node_counts.get(data["type"], 0) + 1
    edge_counts: dict[str, int] = {}
    for _, _, data in g.edges(data=True):
        edge_counts[data["type"]] = edge_counts.get(data["type"], 0) + 1
    return {
        "nodes": node_counts,
        "edges": edge_counts,
        "skipped_labels": g.graph.get("skipped_labels", {}),
        "warnings": g.graph.get("warnings", []),
    }
