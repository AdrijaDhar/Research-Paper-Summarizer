"""Build a hypergraph (nodes + incidence matrix) from the KG in graph.json.

Hyperedges here are NOT the pairwise edges from build_kg.py re-packaged —
they're genuinely multi-way groupings, matching DESIGN.md's "hyperedges over
multi-sentence themes" idea:
  - one hyperedge per entity: {entity} + every sentence that mentions it
    (a concept threading through multiple, possibly distant, sentences)
  - one hyperedge per section: {section} + every sentence/table/figure it
    contains (a section-level thematic grouping)
`cites`/`has_section`/`precedes` stay as plain pairwise structure — they
connect at most two nodes, so there's nothing "hyper" about them.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Hypergraph:
    nodes: list[dict]
    node_index: dict[str, int]
    incidence: torch.Tensor  # [num_nodes, num_hyperedges], dense, binary
    hyperedge_labels: list[str]


def build_hypergraph(graph_path: Path) -> Hypergraph:
    data = json.loads(graph_path.read_text())
    nodes = data["nodes"]
    edges = data["edges"]

    node_index = {n["id"]: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    mentions: dict[str, list[str]] = {}
    section_members: dict[str, list[str]] = {}

    for e in edges:
        if e["type"] == "mentions":
            mentions.setdefault(e["target"], []).append(e["source"])
        elif e["type"] in ("has_sentence", "has_table", "has_figure"):
            section_members.setdefault(e["source"], []).append(e["target"])

    hyperedges: list[list[str]] = []
    hyperedge_labels: list[str] = []

    for entity_id, sentence_ids in mentions.items():
        hyperedges.append([entity_id, *sentence_ids])
        hyperedge_labels.append(f"entity:{entity_id}")

    for section_id, member_ids in section_members.items():
        hyperedges.append([section_id, *member_ids])
        hyperedge_labels.append(f"section:{section_id}")

    incidence = torch.zeros(n_nodes, len(hyperedges))
    for j, members in enumerate(hyperedges):
        for member_id in members:
            if member_id in node_index:
                incidence[node_index[member_id], j] = 1.0

    return Hypergraph(nodes, node_index, incidence, hyperedge_labels)


def build_corpus_hypergraph(graph_paths: list[Path]) -> Hypergraph:
    """Combine several papers' hypergraphs into one block-diagonal hypergraph
    for joint training. No cross-paper hyperedges exist (each paper's KG is
    built independently), so this is exactly equivalent to training on each
    paper separately within the same forward pass — the masked attention in
    HypergraphAttentionLayer already keeps a node from attending outside its
    own hyperedges, which here means outside its own paper too, with zero
    changes needed to the layer itself. Node/hyperedge ids are namespaced by
    paper id (the graph.json's parent directory name) since build_kg.py
    generates plain ids like "sentence-1" independently per paper — without
    namespacing, every paper's nodes would collide under the same ids.
    """
    all_nodes: list[dict] = []
    all_labels: list[str] = []
    blocks: list[torch.Tensor] = []

    for path in graph_paths:
        paper_id = path.parent.name
        hg = build_hypergraph(path)
        for node in hg.nodes:
            prefixed = dict(node)
            prefixed["id"] = f"{paper_id}::{node['id']}"
            prefixed["paper_id"] = paper_id
            all_nodes.append(prefixed)
        all_labels.extend(f"{paper_id}::{label}" for label in hg.hyperedge_labels)
        blocks.append(hg.incidence)

    n_nodes = sum(b.shape[0] for b in blocks)
    n_edges = sum(b.shape[1] for b in blocks)
    incidence = torch.zeros(n_nodes, n_edges)

    node_offset = edge_offset = 0
    for b in blocks:
        n, m = b.shape
        incidence[node_offset : node_offset + n, edge_offset : edge_offset + m] = b
        node_offset += n
        edge_offset += m

    node_index = {n["id"]: i for i, n in enumerate(all_nodes)}
    return Hypergraph(all_nodes, node_index, incidence, all_labels)
