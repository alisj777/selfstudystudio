"""In-process knowledge graph (NetworkX). Exports Cypher so the same data can be loaded into Neo4j."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import networkx as nx

from .schema import EDGE_RULES, Edge, Node, NodeType, Rel


class KGError(ValueError):
    pass


class KnowledgeGraph:
    def __init__(self) -> None:
        self._g = nx.MultiDiGraph()
        self._nodes: dict[str, Node] = {}

    # ---- mutation -------------------------------------------------------
    def add_node(self, node: Node) -> None:
        if node.id in self._nodes:
            raise KGError(f"Duplicate node id: {node.id}")
        self._nodes[node.id] = node
        self._g.add_node(node.id)

    def add_edge(self, edge: Edge) -> None:
        src, dst = self.get(edge.source), self.get(edge.target)
        allowed_src, allowed_dst = EDGE_RULES[edge.rel]
        if src.type not in allowed_src or dst.type not in allowed_dst:
            raise KGError(
                f"{edge.rel} not allowed from {src.type} ({src.id}) to {dst.type} ({dst.id})"
            )
        if self._g.has_edge(edge.source, edge.target, key=edge.rel.value):
            return
        self._g.add_edge(edge.source, edge.target, key=edge.rel.value, **edge.attrs)

    # ---- queries --------------------------------------------------------
    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get(self, node_id: str) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise KGError(f"Unknown node id: {node_id}") from None

    def nodes(self, type: NodeType | None = None, exam: str | None = None) -> list[Node]:
        return [
            n
            for n in self._nodes.values()
            if (type is None or n.type == type) and (exam is None or n.exam == exam)
        ]

    def edges(self) -> list[Edge]:
        return [
            Edge(source=u, target=v, rel=Rel(k), attrs=dict(d))
            for u, v, k, d in self._g.edges(keys=True, data=True)
        ]

    def out(self, node_id: str, rel: Rel) -> list[Node]:
        return [self._nodes[v] for _, v, k in self._g.out_edges(node_id, keys=True) if k == rel.value]

    def into(self, node_id: str, rel: Rel) -> list[Node]:
        return [self._nodes[u] for u, _, k in self._g.in_edges(node_id, keys=True) if k == rel.value]

    def subtopics(self, topic_id: str, recursive: bool = False) -> list[Node]:
        children = self.into(topic_id, Rel.SUBTOPIC_OF)
        if not recursive:
            return children
        result = list(children)
        for child in children:
            result.extend(self.subtopics(child.id, recursive=True))
        return result

    def topic_neighbourhood(self, topic_id: str) -> list[Node]:
        """Parent, children and RELATED_TO topics: one hop for GraphRAG expansion."""
        seen: dict[str, Node] = {}
        for n in (
            self.out(topic_id, Rel.SUBTOPIC_OF)
            + self.into(topic_id, Rel.SUBTOPIC_OF)
            + self.out(topic_id, Rel.RELATED_TO)
            + self.into(topic_id, Rel.RELATED_TO)
        ):
            seen.setdefault(n.id, n)
        return list(seen.values())

    def descriptors(self, criterion_id: str, bands: Iterable[int] | None = None) -> list[Node]:
        wanted = set(bands) if bands is not None else None
        found = [
            d for d in self.out(criterion_id, Rel.HAS_DESCRIPTOR)
            if wanted is None or d.attrs.get("band") in wanted
        ]
        return sorted(found, key=lambda d: d.attrs.get("band", 0))

    def criteria_for(self, skill_id: str) -> list[Node]:
        return self.out(skill_id, Rel.ASSESSED_BY)

    def grammar_for_error(self, error_type_id: str) -> list[Node]:
        return self.out(error_type_id, Rel.INDICATES)

    def prerequisites(self, grammar_id: str) -> list[Node]:
        return self.into(grammar_id, Rel.PREREQUISITE_OF)

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for n in self._nodes.values():
            counts[n.type.value] = counts.get(n.type.value, 0) + 1
        counts["edges"] = self._g.number_of_edges()
        return counts

    # ---- persistence ----------------------------------------------------
    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [n.model_dump(mode="json") for n in self._nodes.values()],
            "edges": [e.model_dump(mode="json") for e in self.edges()],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> KnowledgeGraph:
        payload = json.loads(path.read_text(encoding="utf-8"))
        kg = cls()
        for n in payload["nodes"]:
            kg.add_node(Node.model_validate(n))
        for e in payload["edges"]:
            kg.add_edge(Edge.model_validate(e))
        return kg

    def to_cypher(self) -> str:
        """MERGE statements for loading this graph into Neo4j."""

        def lit(value) -> str:
            return json.dumps(value, ensure_ascii=False)

        lines = []
        for n in self._nodes.values():
            props = {"id": n.id, "name": n.name, "exam": n.exam}
            props.update({k: v for k, v in n.attrs.items() if isinstance(v, (str, int, float, bool, list))})
            body = ", ".join(f"`{k}`: {lit(v)}" for k, v in props.items())
            lines.append(f"MERGE (:{n.type.value} {{{body}}});")
        for e in self.edges():
            lines.append(
                f"MATCH (a {{id: {lit(e.source)}}}), (b {{id: {lit(e.target)}}}) "
                f"MERGE (a)-[:{e.rel.value}]->(b);"
            )
        return "\n".join(lines) + "\n"
