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


@app.command("import-ielts-org")
def import_ielts_org() -> None:
    """Download the free official samples from ielts.org and store them locally (git-ignored)."""
    from .importers.ielts_org import import_all

    s = get_settings()
    chunks, samples = import_all(s.index_dir.parent / "downloads")
    n1 = write_jsonl(s.exam_dir / "corpus" / "private" / "ielts_org.jsonl", chunks)
    n2 = write_jsonl(s.exam_dir / "scoring" / "private" / "ielts_org.jsonl", samples)
    typer.echo(f"speaking questions: {n1}, examiner-scored writing responses: {n2}")
    for smp in samples:
        typer.echo(f"  {smp.sample_id}: {smp.task} band {smp.scores['overall']} ({len(smp.response_text.split())} words)")


@app.command("import-cambridge")
def import_cambridge(
    root: Path = typer.Argument(..., help="Folder holding your Cambridge IELTS PDFs (and optional mp3 audio)"),
    books: list[int] = typer.Option(None, "--book", help="Only these book numbers (repeatable)"),
) -> None:
    """Parse your own Cambridge IELTS books into local, git-ignored corpus and scoring files."""
    from .importers.answer_keys import coverage
    from .importers.cambridge import import_books
    from .rag.build import MIN_CORPUS_QUALITY, MIN_SAMPLE_QUALITY

    s = get_settings()
    results = import_books(root, s.index_dir.parent / "cache" / "cambridge", books or None)
    chunks = [c for r in results for c in r.chunks]
    samples = [x for r in results for x in r.samples]
    n1 = write_jsonl(s.exam_dir / "corpus" / "private" / "cambridge.jsonl", chunks)
    n2 = write_jsonl(s.exam_dir / "scoring" / "private" / "cambridge.jsonl", samples)
    n3 = write_jsonl(s.exam_dir / "answers" / "private" / "cambridge.jsonl", [k for r in results for k in r.keys])
    for r in results:
        if r.skipped:
            typer.echo(f"  Cambridge {r.book}: skipped, {r.skipped}")
            continue
        kinds = {k: sum(1 for c in r.chunks if c.skill == k) for k in ("listening", "reading", "writing", "speaking")}
        low = sum(1 for c in r.chunks if (c.quality or 1) < MIN_CORPUS_QUALITY)
        readable = sum(1 for x in r.samples if (x.quality or 0) >= MIN_SAMPLE_QUALITY)
        typer.echo(
            f"  Cambridge {r.book}: " + ", ".join(f"{k} {v}" for k, v in kinds.items())
            + f", audio linked {sum(1 for c in r.chunks if c.media_uri)}"
            + f", scored writing {len(r.samples)} ({readable} readable essays)"
            + (f", {low} low-quality items not indexed" if low else "")
        )
        if r.keys:
            cov = coverage(r.keys)
            gaps = {f"{t}-{sk}": 40 - n for (t, sk), n in sorted(cov.items()) if n < 40}
            typer.echo(f"      answer keys: {sum(1 for n in cov.values() if n == 40)}/{len(cov)} complete sets"
                       + (f", questions missing {gaps}" if gaps else ""))
    typer.echo(f"corpus items: {n1}, examiner-scored writing: {n2}, answer keys: {n3}. "
               "Run `ielts-tutor build` to re-index.")


@app.command("import-explanations")
def import_explanations_cmd(
    root: Path = typer.Argument(..., help="Folder holding the Cambridge material (explanation PDFs in a 解析 folder)"),
    books: list[int] = typer.Option(None, "--book", help="Only these book numbers (repeatable)"),
    workers: int = typer.Option(6, help="Parallel tesseract processes"),
    all_books: bool = typer.Option(False, "--all", help="Every explanation book, even for tests not imported"),
) -> None:
    """OCR the scanned Chinese explanation books and cut them into per-part sections (local, git-ignored).

    By default only books whose answer keys were imported (`import-cambridge`) are processed."""
    from .grading.store import load_keys
    from .importers.explanations import import_explanations
    from .importers.ocr import tesseract_available

    if problem := tesseract_available():
        raise typer.BadParameter(problem)
    s = get_settings()
    if not books and not all_books:
        books = sorted({k.book for k in load_keys(s.exam_dir)})
        if not books:
            raise typer.BadParameter("no answer keys yet: run import-cambridge first, or pass --book / --all")
        typer.echo(f"books with imported tests: {books}")

    def progress(name: str, done: int, total: int) -> None:
        if done == total or done % 50 < workers * 4:
            typer.echo(f"  OCR {name}: {done}/{total} pages")

    records = import_explanations(root, s.index_dir.parent / "cache" / "cambridge_explain", books or None,
                                  workers, progress)
    n = write_jsonl(s.exam_dir / "explanations" / "private" / "cambridge.jsonl", records)
    by_book: dict[int, set] = {}
    for r in records:
        by_book.setdefault(r.book, set()).add((r.test, r.skill))
    for b, sets in sorted(by_book.items()):
        tests = sorted({t for t, _ in sets})
        typer.echo(f"  Cambridge {b}: tests {', '.join(tests)}; "
                   f"{sum(1 for r in records if r.book == b)} sections")
    typer.echo(f"explanation sections: {n}")


@app.command()
def check(
    book: int = typer.Argument(..., help="Cambridge book number, e.g. 17"),
    test: str = typer.Argument(..., help="Test number (1-4, 5-8 for book 12) or A/B for General Training"),
    skill: str = typer.Argument(..., help="listening | reading"),
    answers_file: Path = typer.Argument(None, help='Text file: "1 litter" per line, or one answer per line'),
    answers: str = typer.Option(None, "--answers", "-a", help='Inline answers: "litter, dogs, A, ..."'),
    explain: bool = typer.Option(True, help="Show the Chinese explanation for each wrong answer"),
    as_json: bool = typer.Option(False, "--json", help="Print the marking as JSON"),
) -> None:
    """Mark your Listening/Reading answers for one Cambridge test and estimate the band."""
    from dataclasses import asdict

    from .grading.objective import mark, parse_answer_text
    from .grading.store import keys_for, load_explanations, load_keys
    from .importers.explanations import find_explanation

    s = get_settings()
    skill = skill.lower()
    keys = keys_for(load_keys(s.exam_dir), book, test, skill)
    if not keys:
        raise typer.BadParameter(f"no answer keys for Cambridge {book} Test {test} {skill}; run import-cambridge")
    text = answers if answers is not None else answers_file.read_text(encoding="utf-8") if answers_file else None
    if text is None:
        raise typer.BadParameter("give an answers file or --answers")
    m = mark(keys, parse_answer_text(text))
    sections = load_explanations(s.exam_dir) if explain else []
    by_q = {k.question: k for k in keys}
    out = []
    for r in m.results:
        row = asdict(r)
        if explain and r.correct is False and r.question in by_q:
            k = by_q[r.question]
            hit = find_explanation(sections, book, k.test, skill, r.question, k.answers or k.all_of or [])
            if hit:
                row["explanation"] = {"section": hit[0].section_id, "pages": hit[0].pages, "text": hit[1]}
        out.append(row)
    if as_json:
        typer.echo(json.dumps({"book": book, "test": m.test, "skill": skill, "raw": m.raw, "keyed": m.keyed,
                               "band": m.band, "results": out}, ensure_ascii=False, indent=1))
        return
    for row in out:
        mark_ = {True: "✓", False: "✗", None: "?"}[row["correct"]]
        if row["correct"]:
            continue
        typer.echo(f" {mark_} Q{row['question']:<2}  you: {row['given'] or '-':<24} answer: {' / '.join(row['expected']) or '(no key)'}"
                   + (f"  [{row['note']}]" if row["note"] else ""))
        if ex := row.get("explanation"):
            typer.echo(f"      解析 ({ex['section']}, p.{ex['pages'][0]}): " + ex["text"].replace("\n", " "))
    missing = 40 - m.keyed
    typer.echo(f"\nCambridge {book} Test {m.test} {skill}: {m.raw}/{m.keyed} correct"
               + (f" ({missing} questions have no key and are not counted)" if missing else "")
               + f" -> band {m.band} (estimate)")


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


# ---------------------------------------------------------------- M2: writing loop


def _scorer_samples():
    from .rag.build import load_samples
    from .writing.scorer import CRITERIA

    s = get_settings()
    return [x for x in load_samples(s.exam_dir / "scoring") if x.task == "task2" and all(c in x.scores for c in CRITERIA)]


def _scorer_tools():
    from .importers.textquality import Vocab
    from .writing.scorer import EmbeddingCache

    s = get_settings()
    return EmbeddingCache(get_embedder(s), s.index_dir.parent / "cache" / "scorer" / "embeddings.json"), Vocab.load()


@app.command("train-scorer")
def train_scorer(regressor: str = typer.Option("gbr", help="gbr (gradient boosting) | ridge")) -> None:
    """Train the light writing score model on the scored Task 2 essays (holdout excluded)."""
    from .writing.scorer import WritingScorer

    s = get_settings()
    samples = _scorer_samples()
    cache, vocab = _scorer_tools()
    scorer = WritingScorer(regressor=regressor)
    cv = scorer.fit(samples, cache, vocab)
    cache.flush()
    path = s.index_dir.parent / "models" / "writing_scorer.pkl"
    scorer.save(path)
    typer.echo(f"trained on {len(samples)} essays -> {path}")
    for c, m in cv.items():
        typer.echo(f"  {c}: 5-fold QWK model {m['model']['qwk']}, kNN {m['knn']['qwk']}, fused {m['fused']['qwk']} "
                   f"(within 0.5: {m['fused']['within_0.5']}, weight on model {m['w_model']})")


@app.command("eval-scorer")
def eval_scorer(
    llm: int = typer.Option(0, "--llm", help="Also run the LLM judge on this many held-out essays (costs API calls)"),
    apply_weights: bool = typer.Option(False, help="Use the fitted fusion weights instead of the 0.5/0.5 default"),
) -> None:
    """Evaluate the score model on the held-out essays; with --llm, fit the model/kNN/LLM fusion weights."""
    from .rag.build import load_jsonl
    from .rag.schema import ScoredSample
    from .writing.evaluate import evaluate
    from .writing.scorer import WritingScorer

    s = get_settings()
    models = s.index_dir.parent / "models"
    scorer = WritingScorer.load(models / "writing_scorer.pkl")
    holdout = load_jsonl(s.exam_dir / "holdout" / "btnotpt_holdout.jsonl", ScoredSample)
    cache, vocab = _scorer_tools()
    rag = client = None
    if llm:
        from .llm import get_llm

        rag = load_multirag(s.index_dir, get_embedder(s), KnowledgeGraph.from_json(_kg_path()), _glossary())
        client = get_llm(s)
    rep = evaluate(scorer, holdout, cache, vocab, rag=rag, llm=client, n_llm=llm,
                   judge_cache=s.index_dir.parent / "eval" / f"llm_judge_{s.llm_model}.jsonl",
                   progress=lambda d, n: typer.echo(f"  judged {d}/{n}") if d % 10 == 0 or d == n else None)
    typer.echo(f"held-out essays: {rep['n']}")
    typer.echo(f"  {'':5} {'source':10} {'QWK':>6} {'r':>6} {'exact':>6} {'±0.5':>6} {'MAE':>6}")
    for c, srcs in rep["sources"].items():
        for name, m in srcs.items():
            typer.echo(f"  {c:5} {name:10} {m['qwk']:6.3f} {m['pearson']:6.3f} {m['exact']:6.3f} {m['within_0.5']:6.3f} {m['mae']:6.3f}")
    m = rep["overall"]["model+knn"]
    typer.echo(f"  overall model+knn: QWK {m['qwk']}, within 0.5 {m['within_0.5']}")
    if "with_llm" in rep:
        typer.echo(f"\nLLM judge on {rep['llm_judged']} essays:")
        for c, d in rep["with_llm"].items():
            typer.echo(f"  {c}: model+knn QWK {d['model+knn']['qwk']}, LLM {d['llm']['qwk']}, "
                       f"fused (2-fold) {d['fused (2-fold)']['qwk']}  weights {d['weights']}")
        fitted = models / "fusion_fitted.json"
        fitted.write_text(json.dumps({"task2": rep["weights"]}, indent=1))
        typer.echo(f"fitted fusion weights -> {fitted} (reference only)")
        if apply_weights:
            (models / "fusion.json").write_text(fitted.read_text())
            typer.echo(f"applied -> {models / 'fusion.json'}")
    if llm:
        from .rag.build import load_samples
        from .writing.evaluate import official_check

        official = [x for x in load_samples(s.exam_dir / "scoring" / "private")
                    if x.rater == "official" and x.response_text and (x.quality is None or x.quality >= 0.9)]
        rows = official_check(scorer, official, cache, vocab, rag, client,
                              s.index_dir.parent / "eval" / f"llm_judge_{s.llm_model}.jsonl")
        typer.echo(f"\nreal examiners' bands ({len(rows)} official sample essays):")
        for r in rows:
            typer.echo(f"  {r['id']:28} {r['task']}  examiner {r['examiner']}  model {r['model'] or '-'}  LLM {r['llm']}")
        for src in ("model", "llm"):
            d = [abs(r[src] - r["examiner"]) for r in rows if r[src] is not None]
            if d:
                typer.echo(f"  {src}: mean |diff| {sum(d) / len(d):.2f}, within 0.5: {sum(x <= 0.5 for x in d)}/{len(d)}")


write_app = typer.Typer(help="Writing practice: get a question, submit an essay, see your profile")
app.add_typer(write_app, name="write")


def _tutor(no_llm: bool):
    from .agents.orchestrator import WritingTutor
    from .agents.toolkit import Toolkit

    return WritingTutor(Toolkit.load(use_llm=not no_llm))


@write_app.command("start")
def write_start(
    task: str = typer.Option("task2", help="task1 | task2"),
    topic: str = typer.Option(None, help="Topic id or words, e.g. 'online education'"),
    question_type: str = typer.Option(None, help="e.g. qt.writing.t2.opinion"),
    source: str = typer.Option(None, help="real (a past question) | generate (a new one, uses the LLM)"),
    question: str = typer.Option(None, help="Use your own question instead"),
    learner: str = typer.Option("default", help="Learner id"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Do not call the LLM"),
) -> None:
    """Plan and set a writing question; prints the session id to submit against."""
    req = {k: v for k, v in {"task": task, "topic": topic, "question_type": question_type, "source": source,
                             "question": question}.items() if v}
    session = _tutor(no_llm).start(learner, **req)
    q = session.question
    typer.echo(f"session: {session.session_id}   ({session.target.reason})")
    typer.echo(f"\nWriting {q.task.upper()} - {q.minutes} minutes, at least {q.min_words} words\n")
    typer.echo(q.prompt)
    if q.data_table:
        typer.echo("\n" + q.data_table)
    typer.echo(f"\nsubmit: ielts-tutor write submit {session.session_id} essay.txt")


@write_app.command("submit")
def write_submit(
    session_id: str,
    essay: Path = typer.Argument(..., help="Text file with your essay"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Model score only, rule-based feedback"),
    as_json: bool = typer.Option(False, "--json", help="Print the whole session as JSON"),
) -> None:
    """Score an essay and give feedback (analysis -> score -> feedback -> learner model)."""
    session = _tutor(no_llm).submit(session_id, essay.read_text(encoding="utf-8"))
    if as_json:
        typer.echo(session.model_dump_json(indent=1))
        return
    sc, fb = session.score, session.feedback
    typer.echo(f"Overall band {sc.overall}" + ("   [needs review: " + sc.review_reason + "]" if sc.needs_review else ""))
    for c, cs in sc.criteria.items():
        src = ", ".join(f"{k} {v}" for k, v in (("model", cs.model), ("kNN", cs.knn), ("LLM", cs.llm)) if v is not None)
        typer.echo(f"  {c:4} {cs.band:<4} ({src})  {cs.rationale[:200]}")
    if session.analysis.errors:
        typer.echo("\nErrors:")
        for e in session.analysis.errors[:10]:
            typer.echo(f"  '{e.text}' -> '{e.correction}'  {e.explanation}")
    typer.echo(f"\n{fb.summary}")
    for i, it in enumerate(fb.items, 1):
        typer.echo(f"  {i}. [{it.criterion}] {it.issue}\n     {it.suggestion}" + (f"\n     e.g. {it.example}" if it.example else "")
                   + f"\n     evidence: {', '.join(it.evidence_ids)}")
    for t in fb.next_band_tips:
        typer.echo(f"  * {t}")
    calls = sum(s.llm_calls for s in session.trace)
    typer.echo(f"\nreview next: {', '.join(fb.review_nodes)}\ntrace: " + " -> ".join(f"{s.agent}({s.ms}ms)" for s in session.trace)
               + f"; LLM calls {calls}")


@write_app.command("profile")
def write_profile(learner: str = typer.Option("default")) -> None:
    """Show the learner's estimated criterion bands and weak grammar points."""
    from .agents.learner import load_learner, summary

    s = get_settings()
    typer.echo(json.dumps(summary(load_learner(learner, s.index_dir.parent / "learners")), ensure_ascii=False, indent=1))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, no_llm: bool = typer.Option(False, "--no-llm")) -> None:
    """Run the local web app (writing practice in the browser)."""
    import uvicorn

    from .web.app import create_app

    uvicorn.run(create_app(use_llm=not no_llm), host=host, port=port)


if __name__ == "__main__":
    app()
