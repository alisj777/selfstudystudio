"""Command line: build the knowledge graph and indexes, search them, tag the corpus."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import get_settings
from .kg import KnowledgeGraph, load_seed, validate
from .rag import build_indexes, get_embedder, load_multirag, multirag_from, save_indexes
from .rag.build import import_scored_csv, load_corpus, load_samples, write_jsonl
from .rag.glossary import Glossary

app = typer.Typer(help="IELTS tutor data tools", no_args_is_help=True)


def _kg_path() -> Path:
    s = get_settings()
    return s.index_dir.parent / f"kg_{s.exam}.json"


def _glossary() -> Glossary | None:
    s = get_settings()
    path = s.exam_dir / s.glossary_file
    return Glossary.load(path) if s.use_glossary and path.exists() else None


@app.command()
def build() -> None:
    """Load the seed ontology, validate it, and build the corpus / rubric / topic indexes."""
    s = get_settings()
    kg = load_seed(s.exam_dir / "kg", exam=s.exam)
    for warning in validate(kg):
        typer.secho(f"warning: {warning}", fg=typer.colors.YELLOW)
    kg.to_json(_kg_path())
    _kg_path().with_suffix(".cypher").write_text(kg.to_cypher(), encoding="utf-8")
    typer.echo(f"knowledge graph: {kg.stats()}")

    chunks = load_corpus(s.exam_dir / "corpus")
    samples = load_samples(s.exam_dir / "scoring")
    counts = save_indexes(build_indexes(kg, chunks, samples, get_embedder(s)), s.index_dir)
    typer.echo(f"indexes ({get_embedder(s).name}) -> {s.index_dir}: {counts}")


@app.command()
def search(
    query: str,
    source: str = typer.Option("all", help="corpus | rubric | topic | all"),
    k: int = typer.Option(5, "-k", "--top-k", help="Number of results"),
    skill: str | None = typer.Option(None, help="writing | speaking | reading | listening"),
) -> None:
    """Search one RAG source, or all three fused with RRF."""
    s = get_settings()
    kg = KnowledgeGraph.from_json(_kg_path())
    glossary = _glossary()
    if glossary:
        matched = glossary.match(query)
        if matched:
            typer.echo(f"(glossary: {', '.join(matched)})")
    rag = load_multirag(s.index_dir, get_embedder(s), kg, glossary)
    if source == "corpus":
        results = rag.corpus.retrieve(query, k, skill=skill)
    elif source == "rubric":
        results = rag.rubric.search_descriptors(query, k, skill=skill)
    elif source == "topic":
        results = rag.topic.retrieve(query, k, skill=skill)
    else:
        results = rag.retrieve(query, k=k, skill=skill)
    for ev in results:
        typer.echo(f"[{ev.source}] {ev.id}  score={ev.score:.4f}")
        typer.echo(f"    {ev.text[:160]}")


@app.command("eval")
def evaluate(
    queries: Path = typer.Option(None, help="Query set YAML (default: data/<exam>/eval/retrieval_queries.yaml)"),
    show_misses: bool = typer.Option(False, help="Print queries whose answer was not ranked first"),
) -> None:
    """Compare retrieval setups: hashing vs semantic embedder, each with and without the glossary."""
    from .rag.embed import HashingEmbedder, SentenceTransformerEmbedder
    from .rag.evaluate import load_queries, run, summarise

    s = get_settings()
    qs = load_queries(queries or s.exam_dir / "eval" / "retrieval_queries.yaml")
    kg = load_seed(s.exam_dir / "kg", exam=s.exam)
    chunks, samples = load_corpus(s.exam_dir / "corpus"), load_samples(s.exam_dir / "scoring")
    glossary = Glossary.load(s.exam_dir / s.glossary_file)

    embedders = [HashingEmbedder(), SentenceTransformerEmbedder(s.embedding_model)]
    rows = []
    for emb in embedders:
        indexes = build_indexes(kg, chunks, samples, emb)
        for use_g in (False, True):
            results = run(multirag_from(indexes, kg, glossary if use_g else None), qs)
            label = f"{emb.name}{' + glossary' if use_g else ''}"
            rows.append((label, summarise(results)))
            if show_misses:
                for r in results:
                    if r.rank != 1:
                        typer.echo(f"  [{label}] {r.query.id} rank={r.rank} q={r.query.query!r} top={r.top}")

    groups = list(rows[0][1].keys())
    for group in groups:
        n = rows[0][1][group]["n"]
        typer.echo(f"\n{group} (n={n})")
        typer.echo(f"  {'setup':45} {'hit@1':>6} {'hit@3':>6} {'MRR':>6}")
        for label, summary in rows:
            m = summary[group]
            typer.echo(f"  {label:45} {m['hit@1']:6.2f} {m['hit@3']:6.2f} {m['mrr']:6.2f}")


@app.command("kg-stats")
def kg_stats() -> None:
    """Show node and edge counts of the built knowledge graph."""
    typer.echo(json.dumps(KnowledgeGraph.from_json(_kg_path()).stats(), indent=1))


@app.command("tag-corpus")
def tag_corpus(
    only_untagged: bool = typer.Option(True, help="Skip items that already have topic_ids"),
    min_confidence: float = 0.7,
) -> None:
    """Ask the configured LLM to tag corpus items; writes accepted tags and a review queue."""
    from .kg.extract import tag_item
    from .llm import get_llm

    s = get_settings()
    kg = load_seed(s.exam_dir / "kg", exam=s.exam)
    llm = get_llm(s)
    chunks = load_corpus(s.exam_dir / "corpus")
    review, tagged = [], 0
    for chunk in chunks:
        if only_untagged and chunk.topic_ids:
            continue
        item = tag_item(llm, kg, chunk.chunk_id, chunk.text, chunk.skill, min_confidence)
        if item.reasons:
            review.append(item)
        else:
            chunk.topic_ids = item.tags.topic_ids
            chunk.question_type = chunk.question_type or item.tags.question_type
            chunk.grammar_ids = item.tags.grammar_ids
            tagged += 1
    out_dir = s.index_dir.parent / "review"
    write_jsonl(out_dir / "tagged_corpus.jsonl", chunks)
    write_jsonl(out_dir / "review_queue.jsonl", review)
    typer.echo(f"auto-accepted {tagged}, sent {len(review)} to {out_dir / 'review_queue.jsonl'}")
    typer.echo(f"tagged corpus written to {out_dir / 'tagged_corpus.jsonl'} (copy into data/ once reviewed)")


@app.command("import-btnotpt")
def import_btnotpt(
    holdout: float = typer.Option(0.15, help="Fraction held out (by band) for evaluating the scorer later"),
) -> None:
    """Import Hugging Face btnotpt/ielts_task_2 into rubric-RAG samples plus a held-out test set.

    Expects train.parquet / test.parquet in var/downloads/btnotpt_ielts_task_2/.
    """
    from .datasets import band_distribution, load_btnotpt, stratified_holdout

    s = get_settings()
    src = s.index_dir.parent / "downloads" / "btnotpt_ielts_task_2"
    samples, report = load_btnotpt(src)
    keep, hold = stratified_holdout(samples, holdout)
    n1 = write_jsonl(s.exam_dir / "scoring" / "btnotpt.jsonl", keep)
    n2 = write_jsonl(s.exam_dir / "holdout" / "btnotpt_holdout.jsonl", hold)
    typer.echo(f"{report}")
    typer.echo(f"calibration samples: {n1}, held out: {n2}")
    typer.echo(f"band distribution: {band_distribution(samples)}")


@app.command("import-scored")
def import_scored(
    csv_path: Path,
    text_col: str = typer.Option(..., help="Column holding the essay"),
    overall_col: str = typer.Option(..., help="Column holding the overall band"),
    tr_col: str | None = None,
    cc_col: str | None = None,
    lr_col: str | None = None,
    gra_col: str | None = None,
    prompt_col: str | None = None,
    task: str = "task2",
    name: str = "imported",
) -> None:
    """Convert a scored-essay CSV into rubric-RAG samples under data/<exam>/scoring/."""
    cols = {"overall": overall_col, "TR": tr_col, "CC": cc_col, "LR": lr_col, "GRA": gra_col}
    samples = import_scored_csv(
        csv_path,
        text_col=text_col,
        score_cols={k: v for k, v in cols.items() if v},
        prompt_col=prompt_col,
        task=task,
        id_prefix=name,
    )
    out = get_settings().exam_dir / "scoring" / f"{name}.jsonl"
    typer.echo(f"wrote {write_jsonl(out, samples)} samples to {out}")


if __name__ == "__main__":
    app()
