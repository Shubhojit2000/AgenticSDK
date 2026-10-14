"""Cross-run memory (Phase 5): experience cards in a Chroma vector store.

A finished run is condensed into an *experience card*: dataset meta-features (train split only),
the strategies tried with their harness validation scores, what failed and the Critic's reason, and
the outcome. Cards are embedded (Gemini embeddings) and stored in Chroma; the planner retrieves the
most similar cards from OTHER datasets through the `search_experience` MCP tool.

Integrity rules, enforced in code:
  * leave-one-dataset-out: a search never returns a card of the dataset being solved (any seed);
  * only validation-side numbers are shown to an agent; the test-based score is kept in the card
    file for analysis but is never rendered or embedded;
  * the 3 unseen datasets cannot be ingested without an explicit flag.

    python -m agenticsdk.memory card cp2_california            # print the card built from runs/<id>
    python -m agenticsdk.memory ingest cp2_california cp3b_eucalyptus
    python -m agenticsdk.memory list
    python -m agenticsdk.memory search "many categorical columns, binary target" --exclude credit_g
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from agenticsdk.harness.datasets import REGISTRY, TARGET, cat_columns, load_split
from agenticsdk.harness.scorer import normalized_score

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
DEFAULT_MEMORY_DIR = ROOT / "memory" / "chroma"
EMBED_CACHE_DIR = ROOT / "llm_cache" / "embeddings"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
COLLECTION = "experience"
CARD_ID_RE = re.compile(r"^CARD (\S+)", re.M)


# ---- the card -------------------------------------------------------------------------------------
class StrategyRecord(BaseModel):
    index: int
    model_family: str
    feature_approach: str
    summary: str
    attempts: int
    best_val: float | None = None
    best_val_norm: float | None = None      # (val - B0) / max(max(B1, B2) - B0, 0.01), on the VALIDATION split
    verdict: str = ""                       # the Critic's last decision on this strategy
    critic_note: str = ""                   # the Critic's reasoning for that decision
    failures: list[str] = Field(default_factory=list)   # short descriptions of attempts that produced no score


class ExperienceCard(BaseModel):
    card_id: str                            # "<dataset>:<run_id>", the handle an agent cites
    run_id: str
    dataset_key: str
    seed: int
    kind: str
    metric: str
    meta: dict                              # dataset meta-features, train split only
    baselines_val: dict[str, float]
    strategies: list[StrategyRecord]
    status: str
    best_strategy: int | None = None
    final_val: float | None = None
    final_val_norm: float | None = None
    attempts_total: int = 0
    memory_used: bool = False               # was this run itself helped by memory? (provenance)
    # NOT rendered, NOT embedded: the test-split result, kept for analysis only
    final_test_normalized: float | None = None
    created: float = Field(default_factory=time.time)

    def dataset_line(self) -> str:
        return describe_dataset(self.meta, self.kind, self.metric)

    def render(self, similarity: float | None = None) -> str:
        """What an agent reads, and what gets embedded. Validation-side numbers only."""
        head = f"CARD {self.card_id}" + (f"  (similarity {similarity:.2f})" if similarity is not None else "")
        lines = [head, "Dataset: " + self.dataset_line()]
        for s in self.strategies:
            score = ("no valid score" if s.best_val is None else
                     f"best val {s.best_val:.4f}, baseline-normalized {s.best_val_norm:+.2f}")
            lines.append(f"Strategy {s.index}: {s.model_family} + {s.feature_approach}: {s.summary} -> {score}"
                         f" ({s.attempts} attempt{'s' if s.attempts != 1 else ''}; Critic: {s.verdict or 'n/a'})")
            if s.critic_note:
                lines.append(f"  Critic said: {s.critic_note}")
            for f in s.failures:
                lines.append(f"  Failed: {f}")
        if self.best_strategy is None:
            lines.append(f"Outcome: {self.status}, no valid predictions.")
        else:
            b = next(s for s in self.strategies if s.index == self.best_strategy)
            lines.append(f"Outcome: {self.status}; best was strategy {b.index} ({b.model_family} + "
                         f"{b.feature_approach}), val {self.final_val:.4f}, baseline-normalized "
                         f"{self.final_val_norm:+.2f}, after {self.attempts_total} attempts.")
        return "\n".join(lines)


def dataset_meta(dataset_key: str, seed: int) -> dict:
    """Meta-features from the TRAIN split only."""
    d = load_split(dataset_key, seed)
    train = d["train"]
    X = train.drop(columns=TARGET)
    cats = cat_columns(X)
    y = train[TARGET]
    meta = {"rows": int(len(train)), "features": int(X.shape[1]), "categorical": len(cats),
            "numeric": int(X.shape[1] - len(cats)), "missing_share": round(float(X.isna().mean().mean()), 4),
            "max_cardinality": int(max((X[c].nunique() for c in cats), default=0))}
    if REGISTRY[dataset_key].kind == "regression":
        meta["target_skew"] = round(float(y.astype(float).skew()), 2)
    else:
        vc = y.astype(str).value_counts(normalize=True)
        meta["classes"] = int(len(vc))
        meta["minority_share"] = round(float(vc.min()), 3)
    return meta


def describe_dataset(meta: dict, kind: str, metric: str) -> str:
    task = {"binary": "binary classification", "multiclass": f"{meta.get('classes', '?')}-class classification",
            "regression": "regression"}[kind]
    out = (f"{task} ({metric}), {meta['rows']} train rows, {meta['features']} features "
           f"({meta['categorical']} categorical, max {meta['max_cardinality']} levels; {meta['numeric']} numeric), "
           f"{meta['missing_share']:.1%} of feature cells missing")
    if "minority_share" in meta:
        out += f", rarest class {meta['minority_share']:.1%}"
    if "target_skew" in meta:
        out += f", target skew {meta['target_skew']}"
    return out


def _val_norm(score: float | None, base: dict[str, float]) -> float | None:
    if score is None or not {"B0", "B1", "B2"} <= set(base):
        return None
    return round(normalized_score(score, base["B0"], max(base["B1"], base["B2"])), 3)


def build_card(run_dir: Path) -> ExperienceCard:
    """Condense a finished run (report.json + events.jsonl) into a card. Deterministic: no LLM call."""
    run_dir = Path(run_dir)
    rep = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if any("strategy_index" not in a for a in rep["attempts"]) or not rep.get("strategy_history"):
        raise ValueError(f"{run_dir.name}: run predates strategy tracking (Phase 2), no card can be built")
    base = rep.get("baselines_val") or {}
    critic = {}     # strategy_index -> last critic event
    for e in events:
        if e["node"] == "critic":
            critic[e["data"].get("strategy_index")] = e
    strategies = []
    for i, st in enumerate(rep["strategy_history"], 1):
        atts = [a for a in rep["attempts"] if a["strategy_index"] == i]
        scored = [a["val_score"] for a in atts if a.get("val_valid")]
        best = max(scored) if scored else None
        ce = critic.get(i)
        note = ""
        if ce:
            note = ce["detail"].split("): ", 1)[-1].strip()
        fails = [f"attempt {a['index']}: " + re.sub(r"\s+", " ", a["error"])[:160]
                 for a in atts if a.get("error")][:3]
        strategies.append(StrategyRecord(
            index=i, model_family=st["model_family"], feature_approach=st["feature_approach"],
            summary=st["summary"][:200], attempts=len(atts), best_val=best, best_val_norm=_val_norm(best, base),
            verdict=(ce["data"].get("decision", "") if ce else ""), critic_note=note[:300], failures=fails))
    best_idx = None
    if rep.get("best_attempt"):
        best_idx = next(a["strategy_index"] for a in rep["attempts"] if a["index"] == rep["best_attempt"])
    return ExperienceCard(
        card_id=f"{rep['dataset_key']}:{rep['run_id']}", run_id=rep["run_id"], dataset_key=rep["dataset_key"],
        seed=rep["seed"], kind=rep["kind"], metric=rep["metric_name"],
        meta=dataset_meta(rep["dataset_key"], rep["seed"]), baselines_val=base, strategies=strategies,
        status=rep["status"], best_strategy=best_idx, final_val=rep.get("final_val_score"),
        final_val_norm=_val_norm(rep.get("final_val_score"), base), attempts_total=len(rep["attempts"]),
        memory_used=bool(rep.get("use_memory")), final_test_normalized=rep.get("final_normalized"))


# ---- embeddings -----------------------------------------------------------------------------------
class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]: ...


class GeminiEmbedder:
    """Gemini embeddings with an on-disk cache keyed by text, so replays and repeated queries cost nothing."""

    def __init__(self, model: str = EMBED_MODEL, dim: int = EMBED_DIM, cache_dir: Path = EMBED_CACHE_DIR,
                 min_interval: float = 0.7):
        self.model, self.dim, self.cache_dir, self.min_interval = model, dim, Path(cache_dir), min_interval
        self.name = f"{model}@{dim}"
        self._client = None
        self._last = -1e9
        self.api_calls = 0

    def _path(self, text: str, kind: str) -> Path:
        h = hashlib.sha256(f"{self.name}|{kind}|{text}".encode()).hexdigest()[:32]
        return self.cache_dir / f"{h}.json"

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            p = self._path(t, kind)
            if p.exists():
                out[i] = json.loads(p.read_text(encoding="utf-8"))
            else:
                todo.append(i)
        for i in todo:
            out[i] = self._embed_one(texts[i], kind)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._path(texts[i], kind).write_text(json.dumps(out[i]), encoding="utf-8")
        return out   # type: ignore[return-value]

    def _embed_one(self, text: str, kind: str) -> list[float]:
        from dotenv import load_dotenv
        from google import genai
        from google.genai import types
        if self._client is None:
            load_dotenv(ROOT / ".env")
            self._client = genai.Client()
        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        delay = 2.0
        for attempt in range(6):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self._client.models.embed_content(
                    model=self.model, contents=[text],
                    config=types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim))
                self.api_calls += 1
                return _unit(list(r.embeddings[0].values))
            except Exception as e:  # noqa: BLE001 - SDK raises many types; only rate limits/outages are retried
                if attempt == 5 or not re.search(r"429|500|503|RESOURCE_EXHAUSTED|UNAVAILABLE", str(e)):
                    raise
                time.sleep(delay)
                delay = min(60.0, delay * 2)
        raise RuntimeError("unreachable")


class HashEmbedder:
    """Deterministic offline embedder (hashed bag of words). For tests and key-less demos only: it
    captures word overlap, not meaning."""

    def __init__(self, dim: int = 256):
        self.dim, self.name = dim, f"hash@{dim}"

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9_]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            out.append(_unit(v))
        return out


def _unit(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v


def get_embedder(name: str = "gemini") -> Embedder:
    return {"gemini": GeminiEmbedder, "hash": HashEmbedder}[name]()


# ---- the store ------------------------------------------------------------------------------------
class ExperienceStore:
    def __init__(self, path: Path = DEFAULT_MEMORY_DIR, embedder: Embedder | None = None):
        import chromadb
        from chromadb.config import Settings
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or GeminiEmbedder()
        self._client = chromadb.PersistentClient(path=str(self.path), settings=Settings(anonymized_telemetry=False))
        self._col = self._client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    def __len__(self) -> int:
        return self._col.count()

    def add(self, card: ExperienceCard) -> None:
        """Insert or replace (same card_id). Embeds the rendered card, which carries no test-side number."""
        emb = self.embedder.embed([card.render()], "document")
        self._col.upsert(ids=[card.card_id], embeddings=emb, documents=[card.model_dump_json()],
                         metadatas=[{"dataset_key": card.dataset_key, "seed": card.seed, "kind": card.kind,
                                     "embedder": self.embedder.name}])

    def cards(self) -> list[ExperienceCard]:
        got = self._col.get()
        return [ExperienceCard.model_validate_json(d) for d in got["documents"]]

    def search(self, query: str, k: int = 3, exclude_datasets: set[str] | frozenset[str] = frozenset()
               ) -> list[tuple[ExperienceCard, float]]:
        """Top-k cards by cosine similarity, never from a dataset in ``exclude_datasets``, at most ONE card
        per dataset (its best match). Seeds of one dataset are near-duplicates and would otherwise fill
        the top k with a single dataset."""
        n = self._col.count()
        if n == 0 or k < 1:
            return []
        where = {"dataset_key": {"$nin": sorted(exclude_datasets)}} if exclude_datasets else None
        emb = self.embedder.embed([query], "query")
        res = self._col.query(query_embeddings=emb, n_results=n, where=where)   # every card, best first
        out, seen = [], set()
        for doc, dist in zip(res["documents"][0], res["distances"][0], strict=True):
            card = ExperienceCard.model_validate_json(doc)
            if card.dataset_key in exclude_datasets or card.dataset_key in seen:
                continue   # the first condition is defence in depth; the filter above should make it dead code
            seen.add(card.dataset_key)
            out.append((card, 1.0 - float(dist)))
            if len(out) == k:
                break
        return out


class MemoryTools:
    """Logic behind the `search_experience` MCP tool (plain Python, testable without MCP)."""

    MAX_K = 5

    def __init__(self, store: ExperienceStore, dataset: str, seed: int = 0):
        self.store, self.dataset = store, dataset
        self.meta = dataset_meta(dataset, seed)
        self.desc = describe_dataset(self.meta, REGISTRY[dataset].kind, REGISTRY[dataset].metric)

    def search_experience(self, query: str = "", k: int = 3) -> str:
        k = max(1, min(int(k), self.MAX_K))
        # the current dataset's own description is always part of the query, so even a vague query ranks by
        # dataset similarity; the leave-one-dataset-out filter is not up to the caller
        hits = self.store.search(f"Dataset: {self.desc}\n{query}".strip(), k, exclude_datasets={self.dataset})
        if not hits:
            return "No experience cards from other datasets are available yet."
        return ("Experience cards from OTHER datasets (hints, not measurements of this dataset; verify with "
                "run_quick_cv before relying on them). Cite a card id in cited_cards only if it changed your plan.\n\n"
                + "\n\n".join(c.render(sim) for c, sim in hits))


# ---- CLI ------------------------------------------------------------------------------------------
def _card_for(run_id: str) -> ExperienceCard:
    return build_card(RUNS_DIR / run_id)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memory-dir", default=str(DEFAULT_MEMORY_DIR))
    ap.add_argument("--embedder", default="gemini", choices=["gemini", "hash"])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("card").add_argument("run_id")
    p = sub.add_parser("ingest")
    p.add_argument("run_ids", nargs="+")
    p.add_argument("--allow-unseen", action="store_true", help="allow cards from the 3 final-unseen datasets")
    sub.add_parser("list")
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--exclude", nargs="*", default=[], help="dataset keys to leave out")
    p.add_argument("-k", type=int, default=3)
    a = ap.parse_args()

    if a.cmd == "card":
        print(_card_for(a.run_id).render())
        return
    store = ExperienceStore(Path(a.memory_dir), get_embedder(a.embedder))
    if a.cmd == "ingest":
        for rid in a.run_ids:
            try:
                c = _card_for(rid)
            except (ValueError, FileNotFoundError) as e:
                print(f"SKIP {rid}: {e}")
                continue
            if REGISTRY[c.dataset_key].group == "unseen" and not a.allow_unseen:
                print(f"SKIP {rid}: {c.dataset_key} is a final-unseen dataset (use --allow-unseen to override)")
                continue
            if not c.strategies:
                print(f"SKIP {rid}: no strategy was produced")
                continue
            store.add(c)
            print(f"added {c.card_id}: {len(c.strategies)} strategies, status {c.status}")
        print(f"store now holds {len(store)} cards ({a.memory_dir})")
    elif a.cmd == "list":
        for c in sorted(store.cards(), key=lambda c: c.card_id):
            print(f"{c.card_id:40s} {c.status:10s} val-norm {c.final_val_norm}  strategies {len(c.strategies)}")
        print(f"{len(store)} cards")
    elif a.cmd == "search":
        for c, sim in store.search(a.query, a.k, set(a.exclude)):
            print(c.render(sim), end="\n\n")


if __name__ == "__main__":
    main()
