"""Load the seed ontology from YAML files.

File format (one file per area, all files in a directory are merged)::

    type: Topic                 # default node type for the file
    nodes:
      - id: topic.education.online
        name: Online learning
        parent: topic.education # sugar for SUBTOPIC_OF
        keywords: [...]         # any other key becomes a node attribute
        edges:                  # explicit outgoing edges
          RELATED_TO: [topic.technology]

Criterion nodes may carry ``descriptors: {band: {descriptor, key_features}}``;
each entry becomes a BandDescriptor node linked by HAS_DESCRIPTOR.
"""

from __future__ import annotations

from pathlib import Path
import yaml

from .graph import KGError, KnowledgeGraph
from .schema import Edge, Node, NodeType, Rel

_RESERVED = {"id", "name", "type", "exam", "edges", "parent", "descriptors"}


def _parse_file(path: Path, exam: str) -> tuple[list[Node], list[Edge]]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    default_type = doc.get("type")
    nodes: list[Node] = []
    edges: list[Edge] = []
    for raw in doc.get("nodes", []):
        node_type = raw.get("type", default_type)
        if node_type is None:
            raise KGError(f"{path.name}: node {raw.get('id')} has no type")
        node = Node(
            id=raw["id"],
            type=NodeType(node_type),
            name=raw.get("name", raw["id"]),
            exam=raw.get("exam", exam),
            attrs={k: v for k, v in raw.items() if k not in _RESERVED},
        )
        nodes.append(node)
        if raw.get("parent"):
            edges.append(Edge(source=node.id, target=raw["parent"], rel=Rel.SUBTOPIC_OF))
        for rel_name, targets in (raw.get("edges") or {}).items():
            for target in [targets] if isinstance(targets, str) else targets:
                edges.append(Edge(source=node.id, target=target, rel=Rel(rel_name)))
        for band, spec in (raw.get("descriptors") or {}).items():
            desc = Node(
                id=f"{node.id}.band{band}",
                type=NodeType.BAND_DESCRIPTOR,
                name=f"{node.name} Band {band}",
                exam=node.exam,
                attrs={
                    "criterion": node.id,
                    "band": int(band),
                    "skill": node.attrs.get("skill"),
                    **(spec if isinstance(spec, dict) else {"descriptor": spec}),
                },
            )
            nodes.append(desc)
            edges.append(Edge(source=node.id, target=desc.id, rel=Rel.HAS_DESCRIPTOR))
    return nodes, edges


def load_seed(directory: Path, exam: str = "ielts") -> KnowledgeGraph:
    files = sorted(directory.glob("*.yaml"))
    if not files:
        raise KGError(f"No seed YAML files in {directory}")
    all_nodes: list[Node] = []
    all_edges: list[Edge] = []
    for path in files:
        try:
            nodes, edges = _parse_file(path, exam)
        except (KeyError, ValueError) as exc:
            raise KGError(f"{path.name}: {exc}") from exc
        all_nodes += nodes
        all_edges += edges

    kg = KnowledgeGraph()
    for node in all_nodes:
        kg.add_node(node)
    for edge in all_edges:
        kg.add_edge(edge)
    return kg


def validate(kg: KnowledgeGraph) -> list[str]:
    """Soft checks that a seed graph is usable; returns human-readable warnings."""
    warnings: list[str] = []
    for crit in kg.nodes(NodeType.CRITERION):
        if not kg.descriptors(crit.id):
            warnings.append(f"Criterion {crit.id} has no band descriptors")
        if not kg.into(crit.id, Rel.ASSESSED_BY):
            warnings.append(f"Criterion {crit.id} is not attached to a skill")
    for et in kg.nodes(NodeType.ERROR_TYPE):
        if not kg.out(et.id, Rel.AFFECTS):
            warnings.append(f"ErrorType {et.id} does not affect any criterion")
    for topic in kg.nodes(NodeType.TOPIC):
        if not topic.attrs.get("keywords"):
            warnings.append(f"Topic {topic.id} has no keywords")
    return warnings

