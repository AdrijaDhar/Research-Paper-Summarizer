"""Self-supervised pretraining for the hand-built hypergraph encoder.

Pretext task: link (membership) prediction — given a node and a hyperedge,
predict whether that node actually belongs to it. Positives come straight
from the incidence matrix; negatives are randomly sampled (node, hyperedge)
pairs that aren't true memberships. This is the standard self-supervised task
for pretraining graph encoders without labels (no ground-truth summary is
needed yet — that's Phase 3's job). If the encoder is learning a meaningful
structural representation, membership prediction accuracy should climb well
above 50% (chance) as training proceeds.

A held-out validation split (`--val-frac`) checks that this is genuine
generalization, not memorization: some true (node, hyperedge) pairs are
masked out of the incidence matrix *used for message passing itself*, not
just excluded from the training loss. Masking only the loss and not the
graph structure would leak the answer — the encoder would still see the
held-out edge directly in its own forward pass (a node's features are
updated by literally summing the hyperedges it belongs to), so validation
accuracy would be inflated by construction. Held-out pairs are still
correctly excluded from negative sampling on both sides (a val positive is a
real membership — it must never be sampled as a "negative" during training).

Caveat: trained on a single paper's graph, this is a mechanism check — "does
the hand-built layer actually learn generalizable structure" — not a useful
pretrained encoder. Meaningful pretraining needs a corpus of many papers'
graphs, not one; treat the validation accuracy here as "is the mechanism
sound," not "is this encoder ready to use."

Usage:
    python scripts/train_hypergraph_encoder.py 2501.17887
    python scripts/train_hypergraph_encoder.py 2501.17887 1706.03762 1810.04805 ...
    python scripts/train_hypergraph_encoder.py 2501.17887 --device cpu
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from rps.graph.embeddings import embed_nodes
from rps.graph.hypergraph import build_corpus_hypergraph
from rps.graph.hypergraph_gnn import HypergraphEncoder


def sample_negative_pairs(full_incidence: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Sample (node, hyperedge) pairs that are true negatives against the
    FULL incidence matrix — a held-out positive must never be sampled as a
    negative just because it's masked out of the training graph."""
    n_nodes, n_edges = full_incidence.shape
    neg_nodes = torch.randint(0, n_nodes, (n_samples,))
    neg_edges = torch.randint(0, n_edges, (n_samples,))
    is_positive = full_incidence[neg_nodes, neg_edges] > 0
    while is_positive.any():
        n_bad = int(is_positive.sum())
        neg_nodes[is_positive] = torch.randint(0, n_nodes, (n_bad,))
        neg_edges[is_positive] = torch.randint(0, n_edges, (n_bad,))
        is_positive = full_incidence[neg_nodes, neg_edges] > 0
    return torch.stack([neg_nodes, neg_edges], dim=1)


def hyperedge_representations(
    node_repr: torch.Tensor, incidence: torch.Tensor, proj: torch.nn.Linear
) -> torch.Tensor:
    degree = incidence.sum(dim=0, keepdim=True).clamp(min=1)
    return proj((incidence / degree).T @ node_repr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ids", nargs="+", help="one or more arXiv ids with an existing graph.json")
    parser.add_argument("--dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=None, help="defaults to <dir>/_corpus/hypergraph_encoder.pt")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    graph_paths = [args.dir / arxiv_id / "graph.json" for arxiv_id in args.ids]
    missing = [p for p in graph_paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing graph.json for: {[p.parent.name for p in missing]} — run build_graph.py first")

    hg = build_corpus_hypergraph(graph_paths)
    print(f"corpus: {len(args.ids)} papers, {len(hg.nodes)} nodes, {hg.incidence.shape[1]} hyperedges")

    embeddings = embed_nodes(hg.nodes)
    in_dim = embeddings.shape[1]

    device = args.device
    x = embeddings.to(device)
    full_incidence = hg.incidence.to(device)

    # Only hold out pairs from hyperedges with degree >= 3, so every val
    # pair still leaves >=2 real members behind for message passing — a
    # degree-2 hyperedge (common: many entities are mentioned in exactly one
    # sentence, so their hyperedge is just {entity, that sentence}) would
    # otherwise risk losing its *only* other member and collapsing to empty,
    # making that particular validation example structurally meaningless
    # even though the NaN-safety fix in hypergraph_gnn.py keeps it from
    # crashing.
    hyperedge_degree = full_incidence.sum(dim=0)
    full_positive_pairs = full_incidence.nonzero()  # [P, 2]
    eligible = hyperedge_degree[full_positive_pairs[:, 1]] >= 3
    eligible_pairs = full_positive_pairs[eligible]
    ineligible_pairs = full_positive_pairs[~eligible]

    perm = torch.randperm(eligible_pairs.shape[0])
    n_val = int(args.val_frac * full_positive_pairs.shape[0])
    n_val = min(n_val, eligible_pairs.shape[0])
    val_pairs = eligible_pairs[perm[:n_val]]
    train_pairs = torch.cat([eligible_pairs[perm[n_val:]], ineligible_pairs])
    print(f"{train_pairs.shape[0]} train memberships, {val_pairs.shape[0]} held-out val memberships "
          f"({full_positive_pairs.shape[0] - eligible_pairs.shape[0]} pairs from degree<3 hyperedges "
          "excluded from holdout, kept in train)")

    # train_incidence: the graph the encoder actually sees during message
    # passing — held-out val edges are zeroed here so they can't leak into
    # the forward pass, only into the (unused-for-training) evaluation.
    train_incidence = full_incidence.clone()
    train_incidence[val_pairs[:, 0], val_pairs[:, 1]] = 0.0

    encoder = HypergraphEncoder(in_dim=in_dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    edge_scorer = torch.nn.Bilinear(args.hidden_dim, args.hidden_dim, 1).to(device)
    hyperedge_query = torch.nn.Linear(args.hidden_dim, args.hidden_dim).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(edge_scorer.parameters()) + list(hyperedge_query.parameters()),
        lr=args.lr,
    )

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        node_repr = encoder(x, train_incidence)
        hyperedge_repr = hyperedge_representations(node_repr, train_incidence, hyperedge_query)

        neg_pairs = sample_negative_pairs(full_incidence, train_pairs.shape[0])

        pos_scores = edge_scorer(node_repr[train_pairs[:, 0]], hyperedge_repr[train_pairs[:, 1]]).squeeze(-1)
        neg_scores = edge_scorer(node_repr[neg_pairs[:, 0]], hyperedge_repr[neg_pairs[:, 1]]).squeeze(-1)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                train_acc = ((scores > 0).float() == labels).float().mean().item()

                # validation: same train_incidence for message passing (the
                # encoder must never see held-out edges), but scored against
                # the held-out val pairs it was never trained on.
                val_node_repr = encoder(x, train_incidence)
                val_hyperedge_repr = hyperedge_representations(val_node_repr, train_incidence, hyperedge_query)
                val_neg_pairs = sample_negative_pairs(full_incidence, val_pairs.shape[0])
                val_pos_scores = edge_scorer(
                    val_node_repr[val_pairs[:, 0]], val_hyperedge_repr[val_pairs[:, 1]]
                ).squeeze(-1)
                val_neg_scores = edge_scorer(
                    val_node_repr[val_neg_pairs[:, 0]], val_hyperedge_repr[val_neg_pairs[:, 1]]
                ).squeeze(-1)
                val_scores = torch.cat([val_pos_scores, val_neg_scores])
                val_labels = torch.cat([torch.ones_like(val_pos_scores), torch.zeros_like(val_neg_scores)])
                val_acc = ((val_scores > 0).float() == val_labels).float().mean().item()

            print(f"epoch {epoch:4d}  loss {loss.item():.4f}  train_acc {train_acc:.3f}  val_acc {val_acc:.3f}")

    out_path = args.out or (args.dir / "_corpus" / "hypergraph_encoder.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "in_dim": in_dim,
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
        },
        out_path,
    )
    print(f"saved trained encoder -> {out_path}")


if __name__ == "__main__":
    main()
