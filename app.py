#!/usr/bin/env python3
"""
ComplianceLens - compliance "count" dashboard for the international-student rulebook.

Run:   streamlit run app.py
Needs: rulebook.md next to this file (or upload one in the sidebar).

WHAT IT DOES
  * Reads a student's activity from: free text, TXT / MD / DOC / DOCX / PDF uploads, or a URL.
  * Screens it against every rule in the rulebook and COUNTS, per category, how many rules are
    compliant / possibly violated / need review / not addressed.
  * Draws a plain-language CONCLUSION (verdict, headline, priority actions).
  * Explanations are on demand: click "Explain" anywhere and the right-hand pane opens with
    the rule text, the evidence sentences, why it was flagged, and a next step.

ARCHITECTURE  (one file, strict layers - each layer only depends on the ones above it)
  1 Config            environment-driven settings
  2 Domain            Rule, Evidence, RuleResult, AnalysisReport, CategoryCount (pure data)
  3 Rulebook          RuleBookParser
  4 NLP               NlpEngine (ABC) -> NltkEngine | LiteEngine (zero-dependency fallback)
  5 Analysis          ComplianceAnalyzer (inverted keyword index, memoised lemmas)
  6 Ingestion         Extractor strategies + SSRF-safe UrlFetcher + IngestionService
  7 Persistence       Database (SQLite), HistoryRepository, ArchiveService (gzip cold storage)
  8 Caching           MemoryCache -> RedisCache -> SqliteCache behind TieredCache
  9 Application       AnalysisService, ConclusionBuilder, Container (dependency wiring)
 10 Presentation      Palette, Styles, Components, ExplainPane, Sidebar, ResultView, App

COLOUR THEORY (60-30-10)
  60% cool off-white surfaces + 30% deep navy  -> calm, trust, authority (a compliance product)
  10% semantic accents, graded by urgency:
      red #D64550 critical  >  orange #E5793A attention  >  amber #E8A33D review
      >  green #2E9E6B compliant  >  slate #94A3B8 not addressed (recedes visually)
  Blue #3B6FE0 is reserved for actions; it is the cool complement of the warm alert hues, so
  buttons never look like warnings. Colour is never the only signal: every status also has an
  icon and a text label (colour-blind safe).

NOTE: this is a keyword/heuristic screen, NOT legal advice and NOT proof of (non-)compliance.
"""
from __future__ import annotations

import abc
import base64
import csv
import dataclasses
import enum
import gzip
import hashlib
import html
import inspect
import io
import ipaddress
import json
import logging
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import tempfile
import threading
import time
import zipfile
import zlib
from collections import Counter, OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable, Optional
from urllib.parse import urljoin, urlparse

import streamlit as st

log = logging.getLogger("compliancelens")
ENGINE_VERSION = "2026.11-1"  # bump to invalidate every cached analysis


# ==============================================================================================
# 1. CONFIG
# ==============================================================================================
@dataclass(frozen=True)
class Config:
    base_dir: Path
    data_dir: Path
    rulebook_path: Path
    redis_url: str
    cache_ttl_s: int
    url_cache_ttl_s: int
    archive_after_days: int
    max_upload_bytes: int
    max_url_bytes: int
    url_timeout_s: float
    max_chars: int

    @property
    def db_path(self) -> Path:
        return self.data_dir / "compliancelens.db"

    @property
    def archive_dir(self) -> Path:
        return self.data_dir / "archive"

    @classmethod
    def from_env(cls) -> "Config":
        base = Path(__file__).resolve().parent
        env = os.getenv
        return cls(
            base_dir=base,
            data_dir=Path(env("APP_DATA_DIR", str(base / "data"))),
            rulebook_path=Path(env("RULEBOOK_PATH", str(base / "rulebook.md"))),
            redis_url=env("REDIS_URL", ""),
            cache_ttl_s=int(env("CACHE_TTL_SECONDS", "86400")),
            url_cache_ttl_s=int(env("URL_CACHE_TTL_SECONDS", "3600")),
            archive_after_days=int(env("ARCHIVE_AFTER_DAYS", "30")),
            max_upload_bytes=int(env("MAX_UPLOAD_MB", "15")) * 1024 * 1024,
            max_url_bytes=int(env("MAX_URL_MB", "5")) * 1024 * 1024,
            url_timeout_s=float(env("URL_TIMEOUT_SECONDS", "10")),
            max_chars=int(env("MAX_CHARS", "400000")),
        )


# ==============================================================================================
# 2. DOMAIN (pure data, JSON round-trippable)
# ==============================================================================================
CATEGORIES: dict[str, str] = {
    "FED": "Federal",
    "QC": "Provincial (Québec)",
    "SCH": "Schools",
    "ASC": "Associations",
    "CORP": "Corporate",
    "ENG": "Engineering ethics",
    "IP": "Painting & IP",
    "UNI": "Student unions",
}


class Status(str, enum.Enum):
    COMPLIANT = "COMPLIANT"
    VIOLATION = "POSSIBLE_VIOLATION"
    REVIEW = "NEEDS_REVIEW"
    NONE = "NOT_ADDRESSED"


class Level(str, enum.Enum):
    CRITICAL = "critical"
    LEGAL = "legal/financial"
    PRACTICE = "good practice"
    UNRATED = "unrated"


LEVEL_BY_EMOJI = {"\U0001F534": Level.CRITICAL, "\U0001F7E0": Level.LEGAL, "\U0001F7E1": Level.PRACTICE}
LEVEL_RANK = {Level.CRITICAL: 0, Level.LEGAL: 1, Level.PRACTICE: 2, Level.UNRATED: 3}
EVIDENCE_RANK = {"violation": 2, "compliant": 1, "neutral": 0}


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    level: Level
    title: str
    text: str

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        return cls(d["id"], d["category"], Level(d["level"]), d["title"], d["text"])


@dataclass(frozen=True)
class Evidence:
    label: str                      # violation | compliant | neutral
    sentence: str
    matched: tuple[str, ...] = ()   # rule keywords found in the sentence
    compliance_signals: int = 0
    violation_signals: int = 0
    note: str = ""                  # e.g. numeric check explanation

    @classmethod
    def from_dict(cls, d: dict) -> "Evidence":
        return cls(d["label"], d["sentence"], tuple(d.get("matched", ())),
                   d.get("compliance_signals", 0), d.get("violation_signals", 0), d.get("note", ""))


@dataclass(frozen=True)
class RuleResult:
    rule: Rule
    status: Status
    evidence: tuple[Evidence, ...] = ()

    @classmethod
    def from_dict(cls, d: dict) -> "RuleResult":
        return cls(Rule.from_dict(d["rule"]), Status(d["status"]),
                   tuple(Evidence.from_dict(e) for e in d["evidence"]))


@dataclass
class CategoryCount:
    total: int = 0
    compliant: int = 0
    violation: int = 0
    review: int = 0
    none: int = 0
    critical_violations: int = 0

    @property
    def addressed(self) -> int:
        return self.compliant + self.violation + self.review

    @property
    def rate(self) -> Optional[float]:
        return self.compliant / self.addressed if self.addressed else None

    def add(self, other: "CategoryCount") -> None:
        for f in dataclasses.fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))


@dataclass(frozen=True)
class AnalysisReport:
    key: str
    created_at: str
    source_label: str
    source_type: str
    engine: str
    n_sentences: int
    truncated: bool
    results: tuple[RuleResult, ...]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisReport":
        return cls(d["key"], d["created_at"], d["source_label"], d["source_type"], d["engine"],
                   d["n_sentences"], d["truncated"], tuple(RuleResult.from_dict(r) for r in d["results"]))

    def tally(self) -> dict[str, CategoryCount]:
        table = {code: CategoryCount() for code in CATEGORIES}
        for r in self.results:
            c = table.setdefault(r.rule.category, CategoryCount())
            c.total += 1
            if r.status is Status.COMPLIANT:
                c.compliant += 1
            elif r.status is Status.VIOLATION:
                c.violation += 1
                if r.rule.level is Level.CRITICAL:
                    c.critical_violations += 1
            elif r.status is Status.REVIEW:
                c.review += 1
            else:
                c.none += 1
        return table

    def totals(self) -> CategoryCount:
        total = CategoryCount()
        for c in self.tally().values():
            total.add(c)
        return total

    def result_for(self, rule_id: str) -> Optional[RuleResult]:
        return next((r for r in self.results if r.rule.id == rule_id), None)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(text: str | bytes) -> str:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()


# ==============================================================================================
# 3. RULEBOOK
# ==============================================================================================
class RulebookError(ValueError):
    pass


@dataclass(frozen=True)
class RuleBook:
    name: str
    sha: str
    rules: tuple[Rule, ...]


def clean_md(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*_`#>]+", " ", text)).strip()


class RuleBookParser:
    """Parses lines like:  **FED-1 🔴 Keep valid status.** body text..."""
    RULE_RE = re.compile(r"^\*\*(?P<id>[A-Z]+-\d+)\s+(?P<lvl>\S+)\s+(?P<title>.+?)\*\*\s*(?P<body>.*)$")

    @classmethod
    def parse(cls, markdown: str, name: str = "rulebook.md") -> RuleBook:
        rules: list[Rule] = []
        for line in markdown.splitlines():
            m = cls.RULE_RE.match(line.strip())
            if not m:
                continue
            prefix = m["id"].split("-")[0]
            if prefix not in CATEGORIES:
                continue
            rules.append(Rule(m["id"], prefix, LEVEL_BY_EMOJI.get(m["lvl"], Level.UNRATED),
                              clean_md(m["title"]).rstrip("."), clean_md(m["body"])))
        if not rules:
            raise RulebookError("No rules found. Expected lines like: **FED-1 🔴 Keep valid status.** ...")
        return RuleBook(name, sha(markdown), tuple(rules))

    @classmethod
    def load_path(cls, path: Path) -> RuleBook:
        if not path.exists():
            raise RulebookError(f"Rulebook not found at {path}. Put rulebook.md next to app.py or upload one.")
        return cls.parse(path.read_text(encoding="utf-8"), path.name)


# ==============================================================================================
# 4. NLP ENGINES (Strategy pattern)
# ==============================================================================================
class Lexicon:
    GENERIC = frozenset({
        "must", "may", "also", "canada", "canadian", "quebec", "student", "school", "rule", "check",
        "ensure", "use", "make", "get", "take", "need", "know", "follow", "give", "include", "example",
        "often", "way", "year", "time", "something", "someone", "anyone"})
    COMPLIANT_CUES = frozenset({
        "renew", "confirm", "approve", "log", "file", "register", "obtain", "submit", "complete", "verify",
        "maintain", "comply", "track", "report", "sign", "secure", "enrol", "enroll", "pay", "insure",
        "apply", "declare", "disclose", "supervise", "cite", "license", "licence", "attend", "authorize",
        "authorise", "keep", "record", "document", "receive", "hold", "extend", "invoice"})
    VIOLATION_CUES = frozenset({
        "exceed", "skip", "ignore", "forget", "miss", "plagiarize", "plagiarise", "plagiarism", "cheat",
        "fabricate", "bribe", "overstay", "falsify", "forge", "steal", "breach", "violate", "vandalize",
        "vandalise", "blockade", "harass", "haze", "conceal", "evade", "underreport"})
    VIOLATION_PHRASES = tuple(re.compile(p, re.I) for p in (
        r"\bwithout (?:a |an |the |my |any )?(?:valid )?(?:permit|permission|authori[sz]ation|"
        r"approval|sin|licen[sc]e|insurance|contract|consent|supervision)\b",
        r"\b(?:has|have|had|is|was|already) expired\b",
        r"\bunpaid (?:work|job|internship|trial)\b",
        r"\bpaid (?:in )?cash\b|\bcash[- ]in[- ]hand\b",
        r"\b(?:over|above|more than|exceed\w*) (?:the )?(?:24|limit|cap)\b",
        r"\bopt(?:ed|s)? out of [^.]{0,40}(?:health|insurance)",
        r"\b(?:copied|traced) (?:from|someone|another|an? )",
        r"\b(?:didn'?t|did not|never) (?:renew|file|log|declare|register|report)\b",
    ))
    CLAUSE_SPLIT = re.compile(r"[;,]|\b(?:and|but|however|while|although|yet)\b", re.I)
    HOURS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b[^.]{0,30}\b(?:week|wk)\b", re.I)
    BREAK_WORDS = re.compile(r"\b(?:break|holiday|vacation|summer|reading week|winter|scheduled)\b", re.I)
    HOURS_RULES = ("FED-4", "FED-5")
    HOURS_CAP = 24.0


class NlpEngine(abc.ABC):
    name: str

    @abc.abstractmethod
    def split_sentences(self, text: str) -> list[str]: ...

    @abc.abstractmethod
    def lemmas(self, sentence: str) -> tuple[str, ...]: ...

    @abc.abstractmethod
    def negated_tokens(self, clause: str) -> list[tuple[str, bool]]: ...

    @abc.abstractmethod
    def forms(self, token: str) -> set[str]: ...

    def expand_cues(self, words: Iterable[str]) -> frozenset[str]:
        return frozenset(words)

    @staticmethod
    def _lines(text: str) -> list[str]:
        """One logical line at a time, so un-punctuated checklist lines don't merge."""
        return [clean_md(ln) for ln in text.splitlines() if clean_md(ln)]


class NltkEngine(NlpEngine):
    """High-fidelity engine: punkt + POS tagger + WordNet lemmatizer + mark_negation."""
    name = "nltk"
    _RESOURCES = [("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab", "punkt_tab"),
                  ("corpora/stopwords", "stopwords"), ("corpora/wordnet", "wordnet"),
                  ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
                  ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng")]

    def __init__(self) -> None:
        import nltk  # raises ImportError -> factory falls back to LiteEngine
        for path, pkg in self._RESOURCES:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:  # offline
                    pass
        from nltk.corpus import stopwords, wordnet
        from nltk.sentiment.util import mark_negation
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import sent_tokenize, word_tokenize
        try:  # smoke test: raises LookupError if any resource is still missing
            nltk.pos_tag(word_tokenize("test sentence"))
            self._stop = set(stopwords.words("english"))
            wordnet.synsets("test")
        except LookupError as exc:
            raise RuntimeError(f"NLTK data unavailable: {exc}") from exc
        self._nltk, self._wn = nltk, wordnet
        self._lem = WordNetLemmatizer()
        self._mark, self._sent, self._word = mark_negation, sent_tokenize, word_tokenize

    def split_sentences(self, text: str) -> list[str]:
        out: list[str] = []
        for line in self._lines(text):
            out.extend(s.strip() for s in self._sent(line) if len(s.strip()) > 3)
        return out

    def _wn_pos(self, tag: str):
        return {"J": self._wn.ADJ, "V": self._wn.VERB, "N": self._wn.NOUN}.get(tag[:1])

    def lemmas(self, sentence: str) -> tuple[str, ...]:
        out = []
        for tok, tag in self._nltk.pos_tag(self._word(sentence)):
            pos, w = self._wn_pos(tag), tok.lower()
            if not pos or not w.isalpha() or len(w) < 3 or w in self._stop:
                continue
            lem = self._lem.lemmatize(w, pos)
            if lem not in Lexicon.GENERIC:
                out.append(lem)
        return tuple(out)

    def negated_tokens(self, clause: str) -> list[tuple[str, bool]]:
        toks = self._mark([t.lower() for t in self._word(clause)])
        return [((t[:-4], True) if t.endswith("_NEG") else (t, False)) for t in toks]

    def forms(self, token: str) -> set[str]:
        return {token, self._lem.lemmatize(token, "v"), self._lem.lemmatize(token, "n")}


class LiteEngine(NlpEngine):
    """Zero-dependency fallback: regex tokenizer + light suffix stemmer + clause-scoped negation."""
    name = "lite"
    _STOP = frozenset("""a about above after again all also am an and any are as at be because been before
        being below between both but by can could did do does doing down during each few for from further
        had has have having he her here hers him his how i if in into is it its itself just me more most my
        myself no nor not now of off on once only or other our ours out over own same she should so some such
        than that the their theirs them then there these they this those through to too under until up very
        was we were what when where which while who whom why will with would you your yours yourself""".split())
    _NEG = frozenset("never no nothing nowhere noone none not cannot".split())
    _SENT = re.compile(r"(?<=[.!?])\s+")
    _TOK = re.compile(r"[a-z]+(?:'[a-z]+)?")

    def __init__(self) -> None:
        self._generic_stems = {self.stem(w) for w in Lexicon.GENERIC}

    @staticmethod
    def stem(w: str) -> str:
        w = w.lower()
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)]
                break
        if w.endswith("e") and len(w) > 3:
            w = w[:-1]
        if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "ls":
            w = w[:-1]
        return w

    def split_sentences(self, text: str) -> list[str]:
        out: list[str] = []
        for line in self._lines(text):
            out.extend(s.strip() for s in self._SENT.split(line) if len(s.strip()) > 3)
        return out

    def lemmas(self, sentence: str) -> tuple[str, ...]:
        out = []
        for w in self._TOK.findall(sentence.lower()):
            if len(w) < 3 or w in self._STOP or w in Lexicon.GENERIC:
                continue
            s = self.stem(w)
            if s not in self._generic_stems:
                out.append(s)
        return tuple(out)

    def negated_tokens(self, clause: str) -> list[tuple[str, bool]]:
        out, neg = [], False
        for t in self._TOK.findall(clause.lower()):
            if t in self._NEG or t.endswith("n't"):
                neg = True
                out.append((t, False))
            else:
                out.append((t, neg))
        return out

    def forms(self, token: str) -> set[str]:
        return {token, self.stem(token)}

    def expand_cues(self, words: Iterable[str]) -> frozenset[str]:
        ws = set(words)
        return frozenset(ws | {self.stem(w) for w in ws})


def build_engine(prefer: str = "auto") -> NlpEngine:
    if prefer in ("auto", "nltk"):
        try:
            return NltkEngine()
        except Exception as exc:  # ImportError, RuntimeError, LookupError ...
            log.warning("NLTK engine unavailable (%s) - using lite engine", exc)
    return LiteEngine()


# ==============================================================================================
# 5. ANALYSIS
# ==============================================================================================
@dataclass(frozen=True)
class AnalysisParams:
    min_score: float = 0.12
    min_hits: int = 2
    top_k: int = 3

    def fingerprint(self) -> str:
        return f"{self.min_score:.3f}|{self.min_hits}|{self.top_k}"


class RuleIndex:
    """Per-rule weighted keywords plus an inverted index (lemma -> rule ids).
    The inverted index means a sentence is only scored against rules that share a lemma
    with it, instead of against every rule in the book."""

    def __init__(self, rules: Iterable[Rule], lemmas: Callable[[str], tuple[str, ...]], top_n: int = 15):
        self.by_id: dict[str, Rule] = {}
        self.kw: dict[str, dict[str, int]] = {}
        self.totals: dict[str, int] = {}
        self.inverted: dict[str, list[str]] = defaultdict(list)
        for r in rules:
            w: Counter = Counter()
            for lem in lemmas(r.title):
                w[lem] += 2
            for lem in lemmas(r.text):
                w[lem] += 1
            top = dict(w.most_common(top_n))
            self.by_id[r.id], self.kw[r.id] = r, top
            self.totals[r.id] = sum(top.values()) or 1
            for lem in top:
                self.inverted[lem].append(r.id)


class ComplianceAnalyzer:
    def __init__(self, engine: NlpEngine) -> None:
        self.engine = engine
        self._lemmas = lru_cache(maxsize=50_000)(engine.lemmas)
        self._violation = engine.expand_cues(Lexicon.VIOLATION_CUES)
        self._compliant = engine.expand_cues(Lexicon.COMPLIANT_CUES)
        self._indexes: dict[str, RuleIndex] = {}
        self._lock = threading.Lock()

    def index_for(self, book: RuleBook) -> RuleIndex:
        with self._lock:
            if book.sha not in self._indexes:
                self._indexes[book.sha] = RuleIndex(book.rules, self._lemmas)
            return self._indexes[book.sha]

    def polarity(self, sentence: str) -> tuple[int, int]:
        """(compliance_signals, violation_signals); negation is scoped to a clause."""
        c = v = 0
        for clause in Lexicon.CLAUSE_SPLIT.split(sentence):
            for tok, neg in self.engine.negated_tokens(clause):
                forms = self.engine.forms(tok)
                if forms & self._violation:
                    c, v = (c + 1, v) if neg else (c, v + 1)
                elif forms & self._compliant:
                    c, v = (c, v + 1) if neg else (c + 1, v)
        v += 2 * sum(1 for p in Lexicon.VIOLATION_PHRASES if p.search(sentence))
        return c, v

    @staticmethod
    def _keep(store: dict[str, Evidence], ev: Evidence) -> None:
        old = store.get(ev.sentence)
        if old is not None:
            if EVIDENCE_RANK[ev.label] < EVIDENCE_RANK[old.label]:
                return
            ev = dataclasses.replace(ev, matched=ev.matched or old.matched, note=ev.note or old.note)
        store[ev.sentence] = ev

    def analyze(self, book: RuleBook, text: str, params: AnalysisParams) -> tuple[list[RuleResult], int]:
        idx = self.index_for(book)
        sentences = self.engine.split_sentences(text)
        evidence: dict[str, dict[str, Evidence]] = defaultdict(dict)

        for sent in sentences:
            lem = set(self._lemmas(sent))
            c, v = self.polarity(sent)
            label = "violation" if v > c else "compliant" if c > v else "neutral"

            candidates: set[str] = set()
            for l in lem:
                candidates.update(idx.inverted.get(l, ()))
            scored = []
            for rid in candidates:
                hits = lem & idx.kw[rid].keys()
                if len(hits) < params.min_hits:
                    continue
                score = sum(idx.kw[rid][h] for h in hits) / idx.totals[rid]
                if score >= params.min_score:
                    scored.append((score, rid, tuple(sorted(hits))))
            scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
            for _, rid, hits in scored[: params.top_k]:
                self._keep(evidence[rid], Evidence(label, sent, hits, c, v))

            m = Lexicon.HOURS_RE.search(sent)
            if m and float(m.group(1)) > Lexicon.HOURS_CAP and not Lexicon.BREAK_WORDS.search(sent):
                note = (f"Numeric check: {float(m.group(1)):g} hours/week is above the "
                        f"{Lexicon.HOURS_CAP:g} h/week off-campus cap while classes are in session.")
                for rid in Lexicon.HOURS_RULES:
                    if rid in idx.by_id:
                        self._keep(evidence[rid], Evidence("violation", sent, (), c, v, note))

        results = []
        for r in book.rules:
            ev = sorted(evidence.get(r.id, {}).values(), key=lambda e: -EVIDENCE_RANK[e.label])
            labels = {e.label for e in ev}
            status = (Status.NONE if not ev else Status.VIOLATION if "violation" in labels
                      else Status.COMPLIANT if "compliant" in labels else Status.REVIEW)
            results.append(RuleResult(r, status, tuple(ev[:4])))
        return results, len(sentences)


# ==============================================================================================
# 6. INGESTION (URL / TXT / DOC / DOCX / PDF / prompt)
# ==============================================================================================
class IngestionError(Exception):
    """User-facing ingestion problem (safe to show verbatim)."""


@dataclass(frozen=True)
class SourceDocument:
    label: str
    kind: str          # prompt | text | html | docx | doc | pdf | url | bundle
    text: str
    truncated: bool = False
    from_cache: bool = False


def decode_bytes(data: bytes, charset: str | None = None) -> str:
    for enc in filter(None, (charset, "utf-8-sig", "cp1252")):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="ignore")


class Extractor(abc.ABC):
    @abc.abstractmethod
    def extract(self, data: bytes) -> str: ...


class TextExtractor(Extractor):
    def extract(self, data: bytes) -> str:
        return decode_bytes(data)


class _HtmlText(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "template"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
             "table", "ul", "ol", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip, self.parts = 0, []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


class HtmlExtractor(Extractor):
    def extract(self, data: bytes) -> str:
        parser = _HtmlText()
        parser.feed(decode_bytes(data))
        return "".join(parser.parts)


class DocxExtractor(Extractor):
    """Regex over word/document.xml: no XML parser, so no entity-expansion attack surface."""
    MAX_XML = 40 * 1024 * 1024

    def extract(self, data: bytes) -> str:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                info = zf.getinfo("word/document.xml")
                if info.file_size > self.MAX_XML:
                    raise IngestionError("The DOCX is too large to process safely.")
                xml = zf.read(info).decode("utf-8", errors="ignore")
        except (zipfile.BadZipFile, KeyError) as exc:
            raise IngestionError("That file is not a valid .docx document.") from exc
        paragraphs = []
        for p in xml.split("</w:p>"):
            line = "".join(re.findall(r"<w:t(?:\s[^>]*)?>([^<]*)</w:t>", p))
            if line.strip():
                paragraphs.append(html.unescape(line))
        return "\n".join(paragraphs)


class LegacyDocExtractor(Extractor):
    """Old binary .doc: needs antiword or catdoc on the server (apt install antiword)."""

    def extract(self, data: bytes) -> str:
        tool = shutil.which("antiword") or shutil.which("catdoc")
        if not tool:
            raise IngestionError("Legacy .doc files need 'antiword' or 'catdoc' installed on the server. "
                                 "Please save the file as .docx and upload it again.")
        with tempfile.NamedTemporaryFile(suffix=".doc") as fh:
            fh.write(data)
            fh.flush()
            try:
                out = subprocess.run([tool, fh.name], capture_output=True, timeout=20, check=True)
            except (subprocess.SubprocessError, OSError) as exc:
                raise IngestionError("Could not read that .doc file. Try saving it as .docx.") from exc
        return decode_bytes(out.stdout)


class PdfExtractor(Extractor):
    MAX_PAGES = 300

    def extract(self, data: bytes) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise IngestionError("PDF support needs the 'pypdf' package (pip install pypdf).") from exc
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted and not reader.decrypt(""):
                raise IngestionError("That PDF is password-protected.")
            pages = [(p.extract_text() or "") for p in reader.pages[: self.MAX_PAGES]]
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError("Could not read that PDF (it may be corrupted).") from exc
        text = "\n".join(pages).strip()
        if not text:
            raise IngestionError("No text layer found - this looks like a scanned PDF (OCR is not supported).")
        return text


def detect_kind(name: str, data: bytes, content_type: str = "") -> str:
    ext, ct = Path(name).suffix.lower(), content_type.lower()
    if data[:5] == b"%PDF-" or "pdf" in ct or ext == ".pdf":
        return "pdf"
    if data[:8] == bytes.fromhex("D0CF11E0A1B11AE1") or ext == ".doc":
        return "doc"
    if data[:4] == b"PK\x03\x04" or ext == ".docx" or "wordprocessingml" in ct:
        return "docx"
    head = data.lstrip()[:15].lower()
    if ext in (".html", ".htm") or "html" in ct or head.startswith((b"<!doctype", b"<html")):
        return "html"
    return "text"


class UrlFetcher:
    """Fetches public http(s) pages only: blocks private/loopback/link-local targets (SSRF),
    re-validates on every redirect hop, and enforces timeout + size caps."""
    MAX_REDIRECTS = 4

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @staticmethod
    def _assert_public(host: str) -> None:
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror as exc:
            raise IngestionError(f"Could not resolve '{host}'.") from exc
        for info in infos:
            if not ipaddress.ip_address(info[4][0]).is_global:
                raise IngestionError("That address points to a private or internal network, which is blocked.")

    def fetch(self, url: str) -> tuple[bytes, str, str]:
        try:
            import requests
        except ImportError as exc:
            raise IngestionError("URL support needs the 'requests' package.") from exc
        current = url.strip()
        for _ in range(self.MAX_REDIRECTS + 1):
            parts = urlparse(current)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                raise IngestionError("Please enter a full http(s):// URL.")
            if parts.port not in (None, 80, 443):
                raise IngestionError("Only standard web ports (80/443) are allowed.")
            self._assert_public(parts.hostname)
            try:
                resp = requests.get(current, stream=True, timeout=self.cfg.url_timeout_s, allow_redirects=False,
                                    headers={"User-Agent": "ComplianceLens/1.0 (+compliance screen)"})
            except requests.RequestException as exc:
                raise IngestionError(f"Could not fetch the URL ({type(exc).__name__}).") from exc
            try:
                if resp.is_redirect and resp.headers.get("Location"):
                    current = urljoin(current, resp.headers["Location"])
                    continue
                if resp.status_code >= 400:
                    raise IngestionError(f"The page answered with HTTP {resp.status_code}.")
                buf = bytearray()
                for chunk in resp.iter_content(65536):
                    buf += chunk
                    if len(buf) > self.cfg.max_url_bytes:
                        raise IngestionError("The page is larger than the allowed size limit.")
                return bytes(buf), resp.headers.get("Content-Type", ""), current
            finally:
                resp.close()
        raise IngestionError("Too many redirects.")


class IngestionService:
    def __init__(self, cfg: Config, cache: "TieredCache", fetcher: UrlFetcher | None = None) -> None:
        self.cfg, self.cache = cfg, cache
        self.fetcher = fetcher or UrlFetcher(cfg)
        self._extractors: dict[str, Extractor] = {
            "text": TextExtractor(), "html": HtmlExtractor(), "docx": DocxExtractor(),
            "doc": LegacyDocExtractor(), "pdf": PdfExtractor()}

    def _finish(self, label: str, kind: str, text: str, from_cache: bool = False) -> SourceDocument:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            raise IngestionError("No readable text was found in that source.")
        truncated = len(text) > self.cfg.max_chars
        return SourceDocument(label, kind, text[: self.cfg.max_chars], truncated, from_cache)

    def from_prompt(self, text: str) -> SourceDocument:
        return self._finish("Pasted text", "prompt", text)

    def from_upload(self, name: str, data: bytes) -> SourceDocument:
        if len(data) > self.cfg.max_upload_bytes:
            raise IngestionError(f"{name} is larger than {self.cfg.max_upload_bytes // 1024 // 1024} MB.")
        kind = detect_kind(name, data)
        return self._finish(name, kind, self._extractors[kind].extract(data))

    def from_uploads(self, files: list[tuple[str, bytes]]) -> SourceDocument:
        docs = [self.from_upload(n, b) for n, b in files]
        if len(docs) == 1:
            return docs[0]
        text = "\n\n".join(d.text for d in docs)
        return self._finish(f"{len(docs)} files", "bundle", text)

    def from_url(self, url: str) -> SourceDocument:
        url = url.strip()
        key = "url:" + sha(url)
        raw, _ = self.cache.get(key)
        if raw:
            d = json.loads(raw)
            return self._finish(d["label"], d["kind"], d["text"], from_cache=True)
        data, content_type, final = self.fetcher.fetch(url)
        kind = detect_kind(urlparse(final).path, data, content_type)
        text = self._extractors[kind].extract(data)
        host = urlparse(final).hostname or url
        doc = self._finish(host, "url", text)
        self.cache.set(key, json.dumps({"label": doc.label, "kind": "url", "text": doc.text}),
                       self.cfg.url_cache_ttl_s)
        return doc


# ==============================================================================================
# 7. PERSISTENCE: SQLite + history + archiving
# ==============================================================================================
def pack(s: str) -> bytes:
    return zlib.compress(s.encode("utf-8"), 3)


def unpack(b: bytes) -> str:
    return zlib.decompress(b).decode("utf-8")


class Database:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS analyses(
        id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL,
        source_label TEXT, source_type TEXT, engine TEXT, n_sentences INTEGER,
        verdict TEXT, violations INTEGER, critical INTEGER, compliant INTEGER, payload BLOB NOT NULL);
    CREATE INDEX IF NOT EXISTS ix_analyses_created ON analyses(created_at);
    CREATE TABLE IF NOT EXISTS archive_index(
        id INTEGER PRIMARY KEY, key TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL, archived_at TEXT NOT NULL,
        source_label TEXT, source_type TEXT, verdict TEXT, violations INTEGER, critical INTEGER,
        compliant INTEGER, file TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS cache(
        key TEXT PRIMARY KEY, value BLOB NOT NULL, expires_at REAL NOT NULL);
    CREATE INDEX IF NOT EXISTS ix_cache_exp ON cache(expires_at);
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(self.SCHEMA)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=15)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()


@dataclass(frozen=True)
class HistoryRow:
    id: int
    created_at: str
    source_label: str
    verdict: str
    violations: int
    critical: int
    compliant: int


class HistoryRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def save(self, report: AnalysisReport, verdict: str) -> None:
        t = report.totals()
        with self.db.conn() as c:
            if c.execute("SELECT 1 FROM archive_index WHERE key=?", (report.key,)).fetchone():
                return  # already archived; restore it explicitly instead of duplicating
            c.execute(
                "INSERT OR IGNORE INTO analyses(key,created_at,source_label,source_type,engine,n_sentences,"
                "verdict,violations,critical,compliant,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (report.key, report.created_at, report.source_label, report.source_type, report.engine,
                 report.n_sentences, verdict, t.violation, t.critical_violations, t.compliant,
                 pack(json.dumps(report.to_dict()))))

    def recent(self, limit: int = 6) -> list[HistoryRow]:
        with self.db.conn() as c:
            rows = c.execute("SELECT id,created_at,source_label,verdict,violations,critical,compliant "
                             "FROM analyses ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)).fetchall()
        return [HistoryRow(**dict(r)) for r in rows]

    def get(self, row_id: int) -> Optional[AnalysisReport]:
        with self.db.conn() as c:
            row = c.execute("SELECT payload FROM analyses WHERE id=?", (row_id,)).fetchone()
        return AnalysisReport.from_dict(json.loads(unpack(row["payload"]))) if row else None

    def counts(self) -> tuple[int, int]:
        with self.db.conn() as c:
            hot = c.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
            cold = c.execute("SELECT COUNT(*) FROM archive_index").fetchone()[0]
        return hot, cold


@dataclass(frozen=True)
class ArchiveSummary:
    archived: int
    files: tuple[str, ...]
    purged_cache_rows: int


class ArchiveService:
    """Hot/cold storage. Analyses older than N days are appended to monthly gzip-JSONL files
    (cold), leaving a tiny row in archive_index so history stays listable and restorable.
    Expired cache rows are purged in the same pass."""

    def __init__(self, db: Database, archive_dir: Path) -> None:
        self.db, self.dir = db, archive_dir
        archive_dir.mkdir(parents=True, exist_ok=True)

    def run(self, retention_days: int) -> ArchiveSummary:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat(timespec="seconds")
        with self.db.conn() as c:
            rows = c.execute("SELECT * FROM analyses WHERE created_at < ? ORDER BY created_at", (cutoff,)).fetchall()
        by_month: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for r in rows:
            by_month[r["created_at"][:7]].append(r)

        files: list[str] = []
        for month, group in by_month.items():
            path = self.dir / f"analyses_{month}.jsonl.gz"
            with gzip.open(path, "at", encoding="utf-8") as fh:  # concatenated gzip members are valid
                for r in group:
                    rec = {k: r[k] for k in r.keys() if k != "payload"}
                    rec["payload"] = json.loads(unpack(r["payload"]))
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            files.append(path.name)
            with self.db.conn() as c:  # file is durable before the hot rows are removed
                for r in group:
                    c.execute("INSERT OR REPLACE INTO archive_index(id,key,created_at,archived_at,source_label,"
                              "source_type,verdict,violations,critical,compliant,file) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                              (r["id"], r["key"], r["created_at"], now_iso(), r["source_label"], r["source_type"],
                               r["verdict"], r["violations"], r["critical"], r["compliant"], path.name))
                    c.execute("DELETE FROM analyses WHERE id=?", (r["id"],))
        with self.db.conn() as c:
            purged = c.execute("DELETE FROM cache WHERE expires_at < ?", (time.time(),)).rowcount
        return ArchiveSummary(len(rows), tuple(files), purged)

    def list_archived(self, limit: int = 8) -> list[sqlite3.Row]:
        with self.db.conn() as c:
            return c.execute("SELECT * FROM archive_index ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()

    def restore(self, row_id: int) -> bool:
        with self.db.conn() as c:
            idx = c.execute("SELECT * FROM archive_index WHERE id=?", (row_id,)).fetchone()
        if not idx or not (self.dir / idx["file"]).exists():
            return False
        with gzip.open(self.dir / idx["file"], "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["id"] != row_id:
                    continue
                with self.db.conn() as c:
                    c.execute(
                        "INSERT OR REPLACE INTO analyses(id,key,created_at,source_label,source_type,engine,"
                        "n_sentences,verdict,violations,critical,compliant,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (rec["id"], rec["key"], rec["created_at"], rec["source_label"], rec["source_type"],
                         rec["engine"], rec["n_sentences"], rec["verdict"], rec["violations"], rec["critical"],
                         rec["compliant"], pack(json.dumps(rec["payload"]))))
                    c.execute("DELETE FROM archive_index WHERE id=?", (row_id,))
                return True
        return False


# ==============================================================================================
# 8. CACHING: memory -> Redis -> SQLite
# ==============================================================================================
class CacheLayer(abc.ABC):
    name: str

    @abc.abstractmethod
    def get(self, key: str) -> Optional[str]: ...

    @abc.abstractmethod
    def set(self, key: str, value: str, ttl_s: int) -> None: ...

    @property
    def healthy(self) -> bool:
        return True


class MemoryCache(CacheLayer):
    name = "memory"

    def __init__(self, max_items: int = 256) -> None:
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._max, self._lock = max_items, threading.Lock()

    def get(self, key):
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            if item[0] < time.time():
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return item[1]

    def set(self, key, value, ttl_s):
        with self._lock:
            self._data[key] = (time.time() + ttl_s, value)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)


class RedisCache(CacheLayer):
    """Optional hot cache. Any Redis failure trips a 30s circuit breaker so the app never stalls."""
    name = "redis"
    COOLDOWN_S = 30

    def __init__(self, url: str = "", client=None, prefix: str = "cl:v1:") -> None:
        self._url, self._client, self._prefix = url, client, prefix
        self._down_until, self._ok = 0.0, False
        self._connect()

    def _connect(self) -> None:
        try:
            if self._client is None:
                import redis
                self._client = redis.Redis.from_url(self._url, socket_timeout=1.0, socket_connect_timeout=1.0)
            self._client.ping()
            self._ok = True
        except Exception as exc:
            log.warning("Redis unavailable (%s)", exc)
            self._ok = False
            self._down_until = time.time() + self.COOLDOWN_S

    @property
    def healthy(self) -> bool:
        if time.time() < self._down_until:
            return False
        if not self._ok:
            self._connect()  # cool-down over: try again (sets a new cool-down on failure)
        return self._ok and time.time() >= self._down_until

    def _trip(self, exc: Exception) -> None:
        log.warning("Redis error (%s) - pausing for %ss", exc, self.COOLDOWN_S)
        self._down_until = time.time() + self.COOLDOWN_S

    def get(self, key):
        if not self.healthy:
            return None
        try:
            raw = self._client.get(self._prefix + key)
            return unpack(raw) if raw else None
        except Exception as exc:
            self._trip(exc)
            return None

    def set(self, key, value, ttl_s):
        if not self.healthy:
            return
        try:
            self._client.setex(self._prefix + key, ttl_s, pack(value))
        except Exception as exc:
            self._trip(exc)


class SqliteCache(CacheLayer):
    name = "sql"

    def __init__(self, db: Database) -> None:
        self.db = db

    def get(self, key):
        with self.db.conn() as c:
            row = c.execute("SELECT value FROM cache WHERE key=? AND expires_at>?", (key, time.time())).fetchone()
        return unpack(row["value"]) if row else None

    def set(self, key, value, ttl_s):
        with self.db.conn() as c:
            c.execute("INSERT OR REPLACE INTO cache(key,value,expires_at) VALUES(?,?,?)",
                      (key, pack(value), time.time() + ttl_s))


class TieredCache:
    """Read-through, write-through, with promotion of lower-layer hits to faster layers."""

    def __init__(self, layers: list[CacheLayer], default_ttl_s: int) -> None:
        self.layers, self.default_ttl_s = layers, default_ttl_s

    def get(self, key: str) -> tuple[Optional[str], Optional[str]]:
        for i, layer in enumerate(self.layers):
            value = layer.get(key)
            if value is not None:
                for upper in self.layers[:i]:
                    upper.set(key, value, self.default_ttl_s)
                return value, layer.name
        return None, None

    def set(self, key: str, value: str, ttl_s: Optional[int] = None) -> None:
        for layer in self.layers:
            layer.set(key, value, ttl_s or self.default_ttl_s)

    def status(self) -> dict[str, bool]:
        return {layer.name: layer.healthy for layer in self.layers}


# ==============================================================================================
# 9. APPLICATION SERVICES
# ==============================================================================================
class Verdict(str, enum.Enum):
    CRITICAL = "critical"
    ATTENTION = "attention"
    REVIEW = "review"
    GOOD = "good"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class Conclusion:
    verdict: Verdict
    headline: str
    summary: str
    coverage: float
    compliance_rate: Optional[float]
    actions: tuple[RuleResult, ...]
    totals: CategoryCount


class ConclusionBuilder:
    @staticmethod
    def _plural(n: int, word: str) -> str:
        return f"{n} {word}{'' if n == 1 else 's'}"

    @classmethod
    def build(cls, report: AnalysisReport) -> Conclusion:
        t, table = report.totals(), report.tally()
        coverage = t.addressed / t.total if t.total else 0.0
        actions = tuple(sorted((r for r in report.results if r.status is Status.VIOLATION),
                               key=lambda r: (LEVEL_RANK[r.rule.level], r.rule.id)))
        if t.addressed == 0:
            verdict, headline = Verdict.INSUFFICIENT, "Not enough information to draw a conclusion"
        elif t.critical_violations:
            verdict = Verdict.CRITICAL
            headline = f"{cls._plural(t.critical_violations, 'critical violation')} need immediate action"
        elif t.violation:
            verdict, headline = Verdict.ATTENTION, f"{cls._plural(t.violation, 'possible violation')} to fix"
        elif t.review:
            verdict, headline = Verdict.REVIEW, f"{cls._plural(t.review, 'item')} need a closer look"
        else:
            verdict, headline = Verdict.GOOD, "No violations detected in what you described"

        parts = [f"Your text touches {t.addressed} of {t.total} rules ({coverage:.0%} coverage)."]
        if t.addressed:
            parts.append(f"{t.compliant} show compliant evidence, {t.violation} show possible violations "
                         f"({t.critical_violations} critical) and {t.review} need review.")
        worst = max(table.items(), key=lambda kv: (kv[1].critical_violations, kv[1].violation), default=None)
        if worst and worst[1].violation:
            parts.append(f"Weakest area: {CATEGORIES.get(worst[0], worst[0])} "
                         f"({cls._plural(worst[1].violation, 'violation')}).")
        return Conclusion(verdict, headline, " ".join(parts), coverage, t.rate, actions, t)


@dataclass(frozen=True)
class Outcome:
    source: str      # "computed" or the cache layer that answered
    elapsed_ms: float


class AnalysisService:
    def __init__(self, cfg: Config, analyzer: ComplianceAnalyzer, cache: TieredCache, history: HistoryRepository):
        self.cfg, self.analyzer, self.cache, self.history = cfg, analyzer, cache, history

    def _key(self, book: RuleBook, doc: SourceDocument, params: AnalysisParams) -> str:
        return sha(f"{ENGINE_VERSION}|{self.analyzer.engine.name}|{book.sha}|{params.fingerprint()}|{sha(doc.text)}")

    def analyze(self, book: RuleBook, doc: SourceDocument, params: AnalysisParams) -> tuple[AnalysisReport, Outcome]:
        t0, key = time.perf_counter(), self._key(book, doc, params)
        raw, layer = self.cache.get(key)
        if raw:
            try:
                report = AnalysisReport.from_dict(json.loads(raw))
                report = dataclasses.replace(report, source_label=doc.label, source_type=doc.kind,
                                             truncated=doc.truncated)
                self.history.save(report, ConclusionBuilder.build(report).verdict.value)
                return report, Outcome(layer or "cache", (time.perf_counter() - t0) * 1000)
            except (KeyError, ValueError, TypeError):
                log.warning("Discarding unreadable cache entry %s", key)
        results, n = self.analyzer.analyze(book, doc.text, params)
        report = AnalysisReport(key, now_iso(), doc.label, doc.kind, self.analyzer.engine.name, n,
                                doc.truncated, tuple(results))
        self.cache.set(key, json.dumps(report.to_dict()), self.cfg.cache_ttl_s)
        self.history.save(report, ConclusionBuilder.build(report).verdict.value)
        return report, Outcome("computed", (time.perf_counter() - t0) * 1000)


@dataclass
class Container:
    """Composition root: the only place concrete classes are wired together."""
    cfg: Config
    engine: NlpEngine
    db: Database
    cache: TieredCache
    history: HistoryRepository
    archive: ArchiveService
    ingestion: IngestionService
    analysis: AnalysisService

    @classmethod
    def build(cls) -> "Container":
        cfg = Config.from_env()
        db = Database(cfg.db_path)
        layers: list[CacheLayer] = [MemoryCache()]
        if cfg.redis_url:
            layers.append(RedisCache(cfg.redis_url))
        layers.append(SqliteCache(db))
        cache = TieredCache(layers, cfg.cache_ttl_s)
        history, archive, engine = HistoryRepository(db), ArchiveService(db, cfg.archive_dir), build_engine()
        try:
            archive.run(cfg.archive_after_days)  # cheap housekeeping at start-up
        except Exception:
            log.exception("Start-up archiving failed")
        return cls(cfg, engine, db, cache, history, archive, IngestionService(cfg, cache),
                   AnalysisService(cfg, ComplianceAnalyzer(engine), cache, history))


# ==============================================================================================
# 10. PRESENTATION
# ==============================================================================================
@dataclass(frozen=True)
class Palette:
    navy: str = "#1B2A49"
    ink: str = "#1B2333"
    muted: str = "#64748B"
    surface: str = "#F5F7FB"
    card: str = "#FFFFFF"
    border: str = "#E3E8F0"
    accent: str = "#3B6FE0"
    ok: str = "#2E9E6B"
    ok_bg: str = "#E7F5EE"
    bad: str = "#D64550"
    bad_bg: str = "#FDECEE"
    orange: str = "#E5793A"
    orange_bg: str = "#FDEFE6"
    warn: str = "#E8A33D"
    warn_bg: str = "#FDF3E1"
    none: str = "#94A3B8"
    none_bg: str = "#EEF1F6"


P = Palette()

STATUS_STYLE = {  # status -> (label, icon, colour, background)
    Status.COMPLIANT: ("Compliant", "✔", P.ok, P.ok_bg),
    Status.VIOLATION: ("Possible violation", "✖", P.bad, P.bad_bg),
    Status.REVIEW: ("Needs review", "?", P.warn, P.warn_bg),
    Status.NONE: ("Not addressed", "–", P.none, P.none_bg),
}
VERDICT_STYLE = {  # verdict -> (label, icon, colour, background)
    Verdict.CRITICAL: ("Action required", "⛔", P.bad, P.bad_bg),
    Verdict.ATTENTION: ("Attention needed", "⚠", P.orange, P.orange_bg),
    Verdict.REVIEW: ("Review suggested", "🔎", P.warn, P.warn_bg),
    Verdict.GOOD: ("Looks compliant", "✔", P.ok, P.ok_bg),
    Verdict.INSUFFICIENT: ("Insufficient data", "ℹ", P.none, P.none_bg),
}
STATUS_ORDER = {Status.VIOLATION: 0, Status.REVIEW: 1, Status.COMPLIANT: 2, Status.NONE: 3}
LEVEL_STYLE = {
    Level.CRITICAL: ("Status-critical", P.bad), Level.LEGAL: ("Legal / financial", P.orange),
    Level.PRACTICE: ("Good practice", P.warn), Level.UNRATED: ("Unrated", P.none)}
STATUS_EXPLANATION = {
    Status.COMPLIANT: "At least one sentence matched this rule's keywords and reads as compliant, "
                      "and no violation cue was found.",
    Status.VIOLATION: "At least one matching sentence carries a violation cue: a negated compliance action "
                      "(e.g. “did not register”), a risky phrase (e.g. “paid in cash”) or a numeric limit breach.",
    Status.REVIEW: "Your text mentions this topic, but without a clear compliant or violating cue. "
                   "Read the sentence and judge it yourself.",
    Status.NONE: "Nothing in your text matched this rule's keywords. That is not proof of compliance: "
                 "it only means the rule was not discussed.",
}
NEXT_STEP = {
    "FED": "Start with your school's International Student Services (ISS). For anything touching your status, "
           "confirm with a lawyer, Québec notary or RCIC.",
    "QC": "Ask ISS first; for pay or workplace abuse contact the CNESST, and for tax Revenu Québec.",
    "SCH": "Talk to ISS and your department, and keep every answer in writing.",
    "ASC": "Check the association's bylaws with its executive and your school's student-life office.",
    "CORP": "Confirm with ISS before accepting or continuing work; for pay or safety issues contact the CNESST.",
    "ENG": "Speak to your supervising engineer and the OIQ; ask ISS if you fear retaliation.",
    "IP": "Consult a lawyer or the Canadian Intellectual Property Office, and ask ISS before selling art regularly.",
    "UNI": "Use your association's grievance process or the school ombudsperson; ask ISS about any insurance gap.",
}
SAMPLE_TEXT = """I renewed my study permit 100 days before it expired and I keep copies of every document.
Last term I worked 32 hours per week at a restaurant, paid in cash.
I did not register for a SIN before starting my job.
I have health insurance through my school's group plan and I confirmed it in writing.
I never plagiarized and I cited every source in my capstone report.
I sold three paintings last month without a contract, and I did not declare the income.
I opted out of the student union health plan before my RAMQ coverage was confirmed.
I filed my federal and Quebec tax returns on time."""


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def minify(s: str) -> str:
    """Collapse indentation/newlines so Markdown never mistakes HTML for a code block."""
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


def stretch(fn) -> dict:
    """Full-width kwargs that work across Streamlit versions (width='stretch' vs use_container_width)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in params:
        return {"width": "stretch"}
    return {"use_container_width": True} if "use_container_width" in params else {}


def first_sentence(text: str, limit: int = 170) -> str:
    s = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def pct(x: Optional[float]) -> str:
    return "–" if x is None else f"{x:.0%}"


class Styles:
    @staticmethod
    def css() -> str:
        return minify(f"""
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="st-"], .stMarkdown {{ font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }}
        #MainMenu, footer {{ visibility: hidden; }}
        header[data-testid="stHeader"] {{ background: transparent; }}
        .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1480px; }}
        section[data-testid="stSidebar"] > div {{ padding-top: 1rem; }}
        .cl-brand {{ display:flex; align-items:center; gap:.6rem; margin-bottom:.4rem; }}
        .cl-logo {{ width:34px; height:34px; border-radius:10px; display:grid; place-items:center; color:#fff;
                    background:linear-gradient(135deg,{P.navy},{P.accent}); font-weight:800; }}
        .cl-brand b {{ font-size:1.1rem; letter-spacing:-.01em; }}
        .cl-brand small {{ display:block; opacity:.65; font-size:.72rem; margin-top:-2px; }}
        .cl-hero {{ background:linear-gradient(120deg,{P.navy} 0%,#27406F 100%); color:#fff; border-radius:18px;
                    padding:1.5rem 1.7rem; margin-bottom:1.1rem; }}
        .cl-hero h1 {{ color:#fff; font-size:1.7rem; margin:0 0 .3rem; letter-spacing:-.02em; padding:0; }}
        .cl-hero p {{ color:#D6E0F5; margin:0; font-size:.98rem; max-width:62ch; }}
        .cl-steps {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:.8rem; margin:1rem 0; }}
        .cl-step {{ background:{P.card}; color:{P.ink}; border:1px solid {P.border}; border-radius:14px; padding:1rem; }}
        .cl-step b {{ display:block; margin:.35rem 0 .15rem; }}
        .cl-step span {{ color:{P.muted}; font-size:.88rem; }}
        .cl-verdict {{ background:{P.card}; color:{P.ink}; border:1px solid {P.border}; border-left:7px solid var(--vc);
                       border-radius:16px; padding:1.2rem 1.4rem; display:flex; gap:1.3rem; align-items:center;
                       box-shadow:0 4px 18px rgba(27,42,73,.07); margin-bottom:.5rem; }}
        .cl-verdict h2 {{ font-size:1.45rem; margin:.35rem 0 .3rem; letter-spacing:-.02em; padding:0; color:{P.ink}; }}
        .cl-verdict p {{ margin:.15rem 0; color:#3B4660; font-size:.95rem; }}
        .cl-verdict .cl-meta {{ color:{P.muted}; font-size:.8rem; }}
        .cl-chip {{ display:inline-block; color:#fff; font-weight:700; font-size:.78rem; padding:.22rem .7rem;
                    border-radius:999px; letter-spacing:.02em; }}
        .cl-ring {{ flex:0 0 96px; width:96px; height:96px; border-radius:50%; display:grid; place-items:center;
                    position:relative; background:conic-gradient({P.ok} calc(var(--p) * 1%), {P.border} 0); }}
        .cl-ring::before {{ content:""; position:absolute; inset:10px; background:{P.card}; border-radius:50%; }}
        .cl-ring div {{ position:relative; text-align:center; line-height:1.05; }}
        .cl-ring b {{ font-size:1.25rem; color:{P.ink}; }}
        .cl-ring small {{ display:block; font-size:.62rem; color:{P.muted}; text-transform:uppercase; letter-spacing:.05em; }}
        .cl-kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.8rem; margin:.8rem 0 1rem; }}
        .cl-kpi {{ background:{P.card}; color:{P.ink}; border:1px solid {P.border}; border-radius:14px; padding:.85rem 1rem; }}
        .cl-kpi b {{ display:block; font-size:1.9rem; line-height:1.1; color:var(--c); }}
        .cl-kpi span {{ font-size:.8rem; color:{P.muted}; }}
        .cl-kpi i {{ font-style:normal; font-size:.72rem; color:var(--c); font-weight:600; }}
        .cl-pill {{ display:inline-flex; align-items:center; gap:.3rem; padding:.14rem .6rem; border-radius:999px;
                    font-size:.76rem; font-weight:600; color:var(--c); background:var(--bg); white-space:nowrap; }}
        .cl-catname b {{ display:block; font-size:.95rem; }}
        .cl-catname span {{ font-size:.76rem; color:{P.muted}; }}
        .cl-bar {{ display:flex; height:12px; border-radius:999px; overflow:hidden; background:{P.none_bg}; }}
        .cl-bar i {{ display:block; height:100%; min-width:4px; }}
        .cl-chips {{ display:flex; gap:.7rem; margin-top:.35rem; font-size:.76rem; font-weight:600; flex-wrap:wrap; }}
        .cl-chips span {{ color:var(--c); }}
        .cl-action {{ background:{P.card}; color:{P.ink}; border:1px solid {P.border}; border-left:5px solid var(--c);
                      border-radius:12px; padding:.7rem .9rem; }}
        .cl-action b {{ font-size:.92rem; }}
        .cl-action p {{ margin:.2rem 0 0; font-size:.84rem; color:#4A5670; }}
        .cl-quote {{ background:{P.card}; color:{P.ink}; border:1px solid {P.border}; border-left:5px solid var(--c);
                     border-radius:10px; padding:.65rem .85rem; margin:.45rem 0; font-size:.9rem; }}
        .cl-quote em {{ display:block; margin-top:.4rem; font-style:normal; font-size:.78rem; color:{P.muted}; }}
        .cl-kw {{ display:inline-block; background:{P.none_bg}; color:#38465F; border-radius:6px; padding:.05rem .45rem;
                  margin:.1rem .2rem .1rem 0; font-size:.74rem; font-family:ui-monospace,Menlo,monospace; }}
        .cl-box {{ background:{P.surface}; color:{P.ink}; border:1px solid {P.border}; border-radius:12px;
                   padding:.75rem .9rem; font-size:.88rem; margin:.4rem 0; }}
        .cl-sec {{ font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:{P.muted};
                   margin:1rem 0 .3rem; }}
        .cl-tier {{ display:flex; gap:.6rem; align-items:flex-start; padding:.45rem .6rem; border-radius:10px;
                    font-size:.86rem; color:{P.ink}; background:{P.surface}; margin:.25rem 0; opacity:.62; }}
        .cl-tier.on {{ opacity:1; background:var(--bg); border:1px solid var(--c); }}
        div[data-testid="stColumn"]:has(#explain-pane), div[data-testid="column"]:has(#explain-pane) {{
            position:sticky; top:3.6rem; align-self:flex-start; max-height:calc(100vh - 4.6rem); overflow-y:auto;
            border:1px solid rgba(128,128,128,.28); border-radius:18px; padding:1rem 1.1rem;
            box-shadow:0 8px 28px rgba(27,42,73,.10); }}
        @media (max-width: 900px) {{
            div[data-testid="stColumn"]:has(#explain-pane), div[data-testid="column"]:has(#explain-pane) {{
                position:static; max-height:none; }}
            .cl-verdict {{ flex-direction:column; align-items:flex-start; }}
        }}
        button[kind="primary"], button[data-testid="stBaseButton-primary"] {{
            background:{P.accent}; border-color:{P.accent}; color:#fff; font-weight:600; }}
        button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {{
            background:#2F5CC4; border-color:#2F5CC4; }}
        button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {{ border-radius:10px; font-weight:500; }}
        """)


class Components:
    """Stateless HTML builders. Every dynamic value is escaped."""

    @staticmethod
    def pill(label: str, icon: str, color: str, bg: str) -> str:
        return f'<span class="cl-pill" style="--c:{color};--bg:{bg}">{esc(icon)} {esc(label)}</span>'

    @classmethod
    def status_pill(cls, status: Status) -> str:
        label, icon, color, bg = STATUS_STYLE[status]
        return cls.pill(label, icon, color, bg)

    @classmethod
    def level_pill(cls, level: Level) -> str:
        label, color = LEVEL_STYLE[level]
        return cls.pill(label, "●", color, P.none_bg)

    @staticmethod
    def hero(title: str, subtitle: str) -> str:
        return minify(f'<div class="cl-hero"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>')

    @staticmethod
    def brand() -> str:
        return minify('<div class="cl-brand"><div class="cl-logo">C</div>'
                      '<div><b>ComplianceLens</b><small>International-student rulebook screen</small></div></div>')

    @staticmethod
    def verdict_card(c: Conclusion, report: AnalysisReport, outcome: Optional[Outcome]) -> str:
        label, icon, color, _ = VERDICT_STYLE[c.verdict]
        ring = 0 if c.compliance_rate is None else round(c.compliance_rate * 100)
        meta = [f"Source: {report.source_label}", f"{report.n_sentences} sentences", f"engine: {report.engine}"]
        if outcome:
            if outcome.source == "history":
                meta.append("🕘 loaded from history")
            elif outcome.source == "computed":
                meta.append(f"⚡ computed in {outcome.elapsed_ms:.0f} ms")
            else:
                meta.append(f"⚡ served from {outcome.source} cache in {outcome.elapsed_ms:.0f} ms")
        note = ""
        if c.verdict is not Verdict.INSUFFICIENT and c.coverage < 0.15:
            note = ('<p class="cl-meta">Only a small part of the rulebook was mentioned. '
                    'Absence of evidence is not evidence of compliance.</p>')
        if report.truncated:
            note += '<p class="cl-meta">The source was very long and was truncated before analysis.</p>'
        return minify(f"""
            <div class="cl-verdict" style="--vc:{color}">
              <div class="cl-ring" style="--p:{ring}"><div><b>{esc(pct(c.compliance_rate))}</b><small>compliant</small></div></div>
              <div>
                <span class="cl-chip" style="background:{color}">{esc(icon)} {esc(label.upper())}</span>
                <h2>{esc(c.headline)}</h2>
                <p>{esc(c.summary)}</p>{note}
                <p class="cl-meta">{esc(' · '.join(meta))}</p>
              </div>
            </div>""")

    @staticmethod
    def kpis(t: CategoryCount) -> str:
        def tile(n: int, label: str, color: str, sub: str = "") -> str:
            return f'<div class="cl-kpi" style="--c:{color}"><b>{n}</b><span>{esc(label)}</span><br><i>{esc(sub)}</i></div>'
        crit = f"{t.critical_violations} critical" if t.critical_violations else "none critical"
        return minify('<div class="cl-kpis">' + tile(t.violation, "Possible violations", P.bad, crit)
                      + tile(t.compliant, "Compliant", P.ok, "evidence found")
                      + tile(t.review, "Needs review", P.warn, "unclear wording")
                      + tile(t.none, "Not addressed", P.none, f"of {t.total} rules") + "</div>")

    @staticmethod
    def category_label(name: str, c: CategoryCount) -> str:
        return minify(f'<div class="cl-catname"><b>{esc(name)}</b>'
                      f'<span>{c.total} rules · {esc(pct(c.rate))} compliant when addressed</span></div>')

    @staticmethod
    def category_bar(c: CategoryCount) -> str:
        segs = [(c.violation, P.bad), (c.review, P.warn), (c.compliant, P.ok), (c.none, P.none_bg)]
        bar = "".join(f'<i style="width:{n / c.total * 100:.2f}%;background:{col}"></i>' for n, col in segs if n and c.total)
        chips = (f'<span style="--c:{P.bad}">✖ {c.violation}</span><span style="--c:#B9791A">? {c.review}</span>'
                 f'<span style="--c:{P.ok}">✔ {c.compliant}</span><span style="--c:{P.muted}">– {c.none}</span>')
        return minify(f'<div class="cl-bar">{bar}</div><div class="cl-chips">{chips}</div>')

    @classmethod
    def action_card(cls, r: RuleResult) -> str:
        color = LEVEL_STYLE[r.rule.level][1]
        return minify(f'<div class="cl-action" style="--c:{color}"><b>{esc(r.rule.id)} · {esc(r.rule.title)}</b> '
                      f'{cls.level_pill(r.rule.level)}<p>{esc(first_sentence(r.rule.text))}</p></div>')

    @staticmethod
    def quote(e: Evidence) -> str:
        color = {"violation": P.bad, "compliant": P.ok}.get(e.label, P.warn)
        label = {"violation": "Reads as a violation", "compliant": "Reads as compliant"}.get(e.label, "Unclear wording")
        kws = "".join(f'<span class="cl-kw">{esc(k)}</span>' for k in e.matched)
        extra = f"<em>{esc(e.note)}</em>" if e.note else ""
        sig = f"<em>{label} · cues: {e.compliance_signals} compliant / {e.violation_signals} violation</em>"
        return minify(f'<div class="cl-quote" style="--c:{color}">“{esc(e.sentence)}”{extra}{sig}'
                      f'<div style="margin-top:.35rem">{kws}</div></div>')

    @staticmethod
    def section(title: str) -> str:
        return f'<div class="cl-sec">{esc(title)}</div>'

    @staticmethod
    def box(text: str) -> str:
        return f'<div class="cl-box">{esc(text)}</div>'


# ---------------------------------------------------------------------------------------------
# Session state helpers (callbacks run BEFORE the rerun, so widget state can be changed safely)
# ---------------------------------------------------------------------------------------------
class State:
    REPORT, OUTCOME, FOCUS, TEXT, TABLE_SEL = "report", "outcome", "focus", "text_input", "table_sel"
    NOTICE = "notice"

    @staticmethod
    def open_pane(kind: str, ident: str = "") -> None:
        st.session_state[State.FOCUS] = (kind, ident)

    @staticmethod
    def close_pane() -> None:
        st.session_state[State.FOCUS] = None

    @staticmethod
    def load_sample() -> None:
        st.session_state[State.TEXT] = SAMPLE_TEXT

    @staticmethod
    def focus() -> Optional[tuple[str, str]]:
        return st.session_state.get(State.FOCUS)


class ExplainPane:
    """Right-hand pane, rendered only on demand."""

    def __init__(self, report: AnalysisReport, conclusion: Conclusion, params: AnalysisParams) -> None:
        self.report, self.c, self.params = report, conclusion, params

    def render(self, kind: str, ident: str) -> None:
        st.markdown('<div id="explain-pane"></div>', unsafe_allow_html=True)
        head, close = st.columns([6, 1], vertical_alignment="center")
        head.markdown(Components.section("Explanation"), unsafe_allow_html=True)
        close.button("✕", key="close_pane", on_click=State.close_pane, help="Close")
        if kind == "verdict":
            self._verdict()
        elif kind == "category":
            self._category(ident)
        elif kind == "rule":
            self._rule(ident)

    # -- verdict ---------------------------------------------------------------------------------
    def _verdict(self) -> None:
        label, icon, color, _ = VERDICT_STYLE[self.c.verdict]
        st.markdown(f"### {icon} {label}")
        st.caption(self.c.headline)
        tiers = [(Verdict.CRITICAL, "One or more status-critical rules show a possible violation."),
                 (Verdict.ATTENTION, "No critical violation, but other rules show a possible violation."),
                 (Verdict.REVIEW, "No violations, but some mentioned rules have unclear wording."),
                 (Verdict.GOOD, "Rules were mentioned and all of them read as compliant."),
                 (Verdict.INSUFFICIENT, "Your text did not match any rule.")]
        rows = ""
        for v, desc in tiers:
            lab, ic, col, bg = VERDICT_STYLE[v]
            on = " on" if v is self.c.verdict else ""
            rows += (f'<div class="cl-tier{on}" style="--c:{col};--bg:{bg}"><b>{esc(ic)} {esc(lab)}</b>'
                     f'<span>{esc(desc)}</span></div>')
        st.markdown(Components.section("How the verdict is decided (first match wins)") + minify(rows),
                    unsafe_allow_html=True)
        t = self.c.totals
        st.markdown(Components.section("Numbers behind it"), unsafe_allow_html=True)
        st.markdown(f"- **{t.total}** rules in the rulebook, **{t.addressed}** mentioned ({self.c.coverage:.0%})\n"
                    f"- **{t.violation}** possible violations (**{t.critical_violations}** status-critical)\n"
                    f"- **{t.compliant}** compliant · **{t.review}** need review · **{t.none}** not addressed\n"
                    f"- Compliance rate = compliant ÷ mentioned = **{pct(self.c.compliance_rate)}**")
        st.markdown(Components.section("How matching works"), unsafe_allow_html=True)
        st.markdown(Components.box(
            f"Engine: {self.report.engine}. A sentence matches a rule when it shares at least "
            f"{self.params.min_hits} weighted keywords and scores ≥ {self.params.min_score:.2f}; "
            f"up to {self.params.top_k} rules per sentence. Negation is read per clause, so “I never "
            f"plagiarized and I cited my sources” stays compliant. Change these in the sidebar."),
            unsafe_allow_html=True)
        st.markdown(Components.box("Heuristic screen only: not legal advice and not proof of (non-)compliance. "
                                   "Confirm anything that affects your status with your school's ISS office."),
                    unsafe_allow_html=True)

    # -- category --------------------------------------------------------------------------------
    def _category(self, code: str) -> None:
        name, c = CATEGORIES.get(code, code), self.report.tally().get(code, CategoryCount())
        st.markdown(f"### {name}")
        st.markdown(Components.category_bar(c), unsafe_allow_html=True)
        st.caption(f"{c.total} rules · {c.addressed} mentioned · {c.critical_violations} critical violation(s)")
        results = [r for r in self.report.results if r.rule.category == code]
        addressed = sorted((r for r in results if r.status is not Status.NONE),
                           key=lambda r: (STATUS_ORDER[r.status], LEVEL_RANK[r.rule.level], r.rule.id))
        if not addressed:
            st.markdown(Components.box("None of the rules in this category were mentioned in your text."),
                        unsafe_allow_html=True)
        for r in addressed:
            lab, icon, *_ = STATUS_STYLE[r.status]
            with st.expander(f"{icon} {r.rule.id} · {r.rule.title}", expanded=r.status is Status.VIOLATION):
                st.markdown(Components.status_pill(r.status) + " " + Components.level_pill(r.rule.level)
                            + Components.box(r.rule.text) + "".join(Components.quote(e) for e in r.evidence),
                            unsafe_allow_html=True)
                st.button("Open full explanation", key=f"pane_open_{r.rule.id}", on_click=State.open_pane,
                          args=("rule", r.rule.id))
        rest = [r for r in results if r.status is Status.NONE]
        if rest:
            with st.expander(f"– {len(rest)} rules not addressed"):
                st.markdown("\n".join(f"- **{r.rule.id}** {r.rule.title}" for r in rest))
        st.markdown(Components.section("Where to get help") + Components.box(NEXT_STEP.get(code, "")),
                    unsafe_allow_html=True)

    # -- rule ------------------------------------------------------------------------------------
    def _rule(self, rule_id: str) -> None:
        r = self.report.result_for(rule_id)
        if r is None:
            st.info("That rule is not part of this report.")
            return
        st.markdown(f"### {r.rule.id} · {r.rule.title}")
        st.markdown(Components.status_pill(r.status) + " " + Components.level_pill(r.rule.level)
                    + f' <span class="cl-pill" style="--c:{P.navy};--bg:{P.none_bg}">{esc(CATEGORIES.get(r.rule.category, r.rule.category))}</span>',
                    unsafe_allow_html=True)
        st.markdown(Components.section("What the rulebook requires") + Components.box(r.rule.text),
                    unsafe_allow_html=True)
        st.markdown(Components.section("What your text says"), unsafe_allow_html=True)
        if r.evidence:
            st.markdown("".join(Components.quote(e) for e in r.evidence), unsafe_allow_html=True)
        else:
            st.markdown(Components.box("No sentence matched this rule."), unsafe_allow_html=True)
        st.markdown(Components.section("Why it was classified this way") + Components.box(STATUS_EXPLANATION[r.status]),
                    unsafe_allow_html=True)
        if r.status in (Status.VIOLATION, Status.REVIEW):
            st.markdown(Components.section("Suggested next step")
                        + Components.box(NEXT_STEP.get(r.rule.category, "Ask your school's ISS office.")),
                        unsafe_allow_html=True)
        st.button(f"See all {CATEGORIES.get(r.rule.category, '')} rules", key="back_to_cat",
                  on_click=State.open_pane, args=("category", r.rule.category))


class Exporter:
    @staticmethod
    def json_bytes(report: AnalysisReport) -> bytes:
        return json.dumps(report.to_dict(), indent=2, ensure_ascii=False).encode("utf-8")

    @staticmethod
    def csv_bytes(report: AnalysisReport) -> bytes:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "category", "level", "status", "title", "evidence"])
        for r in report.results:
            w.writerow([r.rule.id, CATEGORIES.get(r.rule.category, r.rule.category), r.rule.level.value,
                        r.status.value, r.rule.title, " | ".join(e.sentence for e in r.evidence)])
        return buf.getvalue().encode("utf-8-sig")

    @staticmethod
    def summary_text(report: AnalysisReport, c: Conclusion) -> bytes:
        lines = [f"ComplianceLens report - {report.created_at}", f"Source: {report.source_label}", "",
                 c.headline, c.summary, "", "Count by category:"]
        for code, cc in report.tally().items():
            lines.append(f"  {CATEGORIES[code]:<22} rules={cc.total:<3} ok={cc.compliant:<3} "
                         f"violations={cc.violation:<3} review={cc.review:<3} none={cc.none:<3}")
        if c.actions:
            lines += ["", "Priority actions:"] + [f"  [{r.rule.level.value}] {r.rule.id} {r.rule.title}" for r in c.actions]
        lines += ["", "Heuristic screen only - not legal advice. Confirm with your school's ISS office."]
        return "\n".join(lines).encode("utf-8")


class ResultView:
    def __init__(self, report: AnalysisReport, outcome: Optional[Outcome], params: AnalysisParams) -> None:
        self.report, self.outcome, self.params = report, outcome, params
        self.conclusion = ConclusionBuilder.build(report)

    def render(self) -> None:
        focus = State.focus()
        if focus:
            main, pane = st.columns([2.1, 1.1], gap="large")
            with pane:
                ExplainPane(self.report, self.conclusion, self.params).render(*focus)
        else:
            main = st.container()
        with main:
            overview, rules, export = st.tabs(["Overview", "All rules", "Export"])
            with overview:
                self._overview()
            with rules:
                self._rules_table()
            with export:
                self._export()

    def _overview(self) -> None:
        c = self.conclusion
        st.markdown(Components.verdict_card(c, self.report, self.outcome), unsafe_allow_html=True)
        st.button("Why this verdict?  ›", key="why_verdict", on_click=State.open_pane, args=("verdict", ""))
        st.markdown(Components.kpis(c.totals), unsafe_allow_html=True)

        st.markdown("#### Count by category")
        for code, cc in self.report.tally().items():
            if not cc.total:
                continue
            a, b, d = st.columns([2.6, 5.2, 1.5], vertical_alignment="center")
            a.markdown(Components.category_label(CATEGORIES[code], cc), unsafe_allow_html=True)
            b.markdown(Components.category_bar(cc), unsafe_allow_html=True)
            d.button("Explain ›", key=f"cat_{code}", on_click=State.open_pane, args=("category", code),
                     **stretch(st.button))

        if c.actions:
            st.markdown("#### Priority actions")
            st.caption("Most urgent first. Open any item for the evidence behind it.")
            for r in c.actions[:6]:
                left, right = st.columns([6, 1.5], vertical_alignment="center")
                left.markdown(Components.action_card(r), unsafe_allow_html=True)
                right.button("Explain ›", key=f"act_{r.rule.id}", on_click=State.open_pane, args=("rule", r.rule.id),
                             **stretch(st.button))
            if len(c.actions) > 6:
                st.caption(f"+ {len(c.actions) - 6} more in the “All rules” tab.")

    def _rules_table(self) -> None:
        import pandas as pd
        f1, f2, f3 = st.columns([3, 2, 2])
        statuses = f1.multiselect("Status", list(Status), default=[Status.VIOLATION, Status.REVIEW, Status.COMPLIANT],
                                  format_func=lambda s: f"{STATUS_STYLE[s][1]} {STATUS_STYLE[s][0]}")
        cats = f2.multiselect("Category", list(CATEGORIES), format_func=lambda k: CATEGORIES[k])
        query = f3.text_input("Search", placeholder="e.g. SIN, permit…")
        rows = [r for r in self.report.results if r.status in statuses
                and (not cats or r.rule.category in cats)
                and (not query or query.lower() in f"{r.rule.id} {r.rule.title} {r.rule.text}".lower())]
        rows.sort(key=lambda r: (STATUS_ORDER[r.status], LEVEL_RANK[r.rule.level], r.rule.id))
        if not rows:
            st.info("No rules match these filters.")
            return
        df = pd.DataFrame([{
            "Rule": r.rule.id, "Status": f"{STATUS_STYLE[r.status][1]} {STATUS_STYLE[r.status][0]}",
            "Level": LEVEL_STYLE[r.rule.level][0], "Title": r.rule.title, "Matches": len(r.evidence)} for r in rows])
        st.caption(f"{len(rows)} rules · select a row to open its explanation on the right")
        event = st.dataframe(df, hide_index=True, on_select="rerun", selection_mode="single-row",
                             key="rules_df", **stretch(st.dataframe))
        selected = event.selection.rows if event and getattr(event, "selection", None) else []
        if selected:
            rule_id = rows[selected[0]].rule.id
            if st.session_state.get(State.TABLE_SEL) != rule_id:  # only react to a NEW selection
                st.session_state[State.TABLE_SEL] = rule_id
                State.open_pane("rule", rule_id)
                st.rerun()

    def _export(self) -> None:
        stem = f"compliancelens_{self.report.created_at[:10]}"
        st.caption("Download the full result set. Archived history is stored as monthly .jsonl.gz files.")
        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇ Summary (.txt)", Exporter.summary_text(self.report, self.conclusion), f"{stem}.txt",
                           "text/plain", **stretch(st.download_button))
        c2.download_button("⬇ Per-rule CSV", Exporter.csv_bytes(self.report), f"{stem}.csv", "text/csv",
                           **stretch(st.download_button))
        c3.download_button("⬇ Full JSON", Exporter.json_bytes(self.report), f"{stem}.json", "application/json",
                           **stretch(st.download_button))


class Sidebar:
    def __init__(self, container: Container) -> None:
        self.ct = container

    # Returns (RuleBook | None, AnalysisParams)
    def render_settings(self) -> tuple[Optional[RuleBook], AnalysisParams]:
        with st.expander("⚙ Rulebook & sensitivity"):
            up = st.file_uploader("Use a different rulebook (.md)", type=["md", "txt"], key="rulebook_upload")
            try:
                book = (RuleBookParser.parse(decode_bytes(up.getvalue()), up.name) if up
                        else RuleBookParser.load_path(self.ct.cfg.rulebook_path))
                st.caption(f"✔ {book.name} · {len(book.rules)} rules · v{book.sha[:8]}")
            except RulebookError as exc:
                st.error(str(exc))
                book = None
            min_score = st.slider("Match strictness", 0.05, 0.40, 0.12, 0.01,
                                  help="Higher = fewer, more precise matches.")
            min_hits = st.slider("Minimum keyword hits", 1, 4, 2)
            top_k = st.slider("Rules per sentence", 1, 6, 3)
        return book, AnalysisParams(min_score, min_hits, top_k)

    def render_inputs(self, book: Optional[RuleBook], params: AnalysisParams) -> None:
        tab_text, tab_file, tab_url = st.tabs(["✍ Text", "📎 Files", "🔗 URL"])
        disabled = book is None
        with tab_text:
            st.text_area("Describe your situation", key=State.TEXT, height=190, label_visibility="collapsed",
                         placeholder="e.g. Last term I worked 30 hours a week at a restaurant…")
            c1, c2 = st.columns([3, 2])
            if c1.button("Analyze text", type="primary", disabled=disabled, key="go_text", **stretch(st.button)):
                text = st.session_state.get(State.TEXT, "")
                self._run(book, params, lambda: self.ct.ingestion.from_prompt(text))
            c2.button("Sample", on_click=State.load_sample, key="sample_btn", **stretch(st.button))
        with tab_file:
            files = st.file_uploader("TXT, MD, DOC, DOCX or PDF", type=["txt", "md", "doc", "docx", "pdf"],
                                     accept_multiple_files=True, key="uploads")
            if st.button("Analyze files", type="primary", disabled=disabled or not files, key="go_files",
                         **stretch(st.button)):
                payload = [(f.name, f.getvalue()) for f in files]
                self._run(book, params, lambda: self.ct.ingestion.from_uploads(payload))
        with tab_url:
            url = st.text_input("Public page, PDF or document URL", placeholder="https://…", key="url_input")
            if st.button("Fetch & analyze", type="primary", disabled=disabled or not url.strip(), key="go_url",
                         **stretch(st.button)):
                self._run(book, params, lambda: self.ct.ingestion.from_url(url))

    def _run(self, book: RuleBook, params: AnalysisParams, source: Callable[[], SourceDocument]) -> None:
        try:
            with st.spinner("Analysing…"):
                doc = source()
                report, outcome = self.ct.analysis.analyze(book, doc, params)
        except IngestionError as exc:
            st.error(str(exc))
            return
        except Exception:
            log.exception("Analysis failed")
            st.error("Something went wrong while analysing. Please try again or use a different source.")
            return
        st.session_state.update({State.REPORT: report, State.OUTCOME: outcome, State.FOCUS: None,
                                 State.TABLE_SEL: None, "params": params})
        st.rerun()

    def render_history(self) -> None:
        hot, cold = self.ct.history.counts()
        with st.expander(f"🕘 History ({hot})"):
            for row in self.ct.history.recent(6):
                label, icon, *_ = VERDICT_STYLE.get(Verdict(row.verdict), VERDICT_STYLE[Verdict.INSUFFICIENT])
                if st.button(f"{icon} {row.source_label[:26]} · {row.created_at[5:10]}", key=f"hist_{row.id}",
                             help=f"{label} · {row.violations} violation(s)", **stretch(st.button)):
                    report = self.ct.history.get(row.id)
                    if report:
                        st.session_state.update({State.REPORT: report, State.OUTCOME: Outcome("history", 0.0),
                                                 State.FOCUS: None})
                        st.rerun()
            if not hot:
                st.caption("Nothing yet.")
        with st.expander(f"🗄 Archive ({cold})"):
            st.caption(f"Analyses older than {self.ct.cfg.archive_after_days} days move to compressed cold storage "
                       "automatically.")
            if st.button("Archive now", key="archive_now", **stretch(st.button)):
                s = self.ct.archive.run(self.ct.cfg.archive_after_days)
                st.toast(f"Archived {s.archived} analyses, purged {s.purged_cache_rows} expired cache rows.")
            for row in self.ct.archive.list_archived(5):
                if st.button(f"↩ Restore {row['source_label'][:22]} · {row['created_at'][:10]}",
                             key=f"restore_{row['id']}", **stretch(st.button)):
                    if self.ct.archive.restore(row["id"]):
                        st.rerun()
        status = self.ct.cache.status()
        dots = " · ".join(f"{'🟢' if ok else '🔴'} {name}" for name, ok in status.items())
        st.caption(f"Cache layers: {dots} · NLP: {self.ct.engine.name}")


class App:
    def __init__(self) -> None:
        st.set_page_config(page_title="ComplianceLens", page_icon="🧭", layout="wide",
                           initial_sidebar_state="expanded")
        st.markdown(f"<style>{Styles.css()}</style>", unsafe_allow_html=True)
        self.ct = self._container()

    @staticmethod
    @st.cache_resource(show_spinner="Starting services…")
    def _container() -> Container:
        return Container.build()

    def run(self) -> None:
        sidebar = Sidebar(self.ct)
        with st.sidebar:
            st.markdown(Components.brand(), unsafe_allow_html=True)
            book, params = sidebar.render_settings()
            sidebar.render_inputs(book, params)
            st.divider()
            sidebar.render_history()

        report: Optional[AnalysisReport] = st.session_state.get(State.REPORT)
        if report is None:
            self._welcome(book, params)
            return
        ResultView(report, st.session_state.get(State.OUTCOME), st.session_state.get("params", params)).render()

    def _welcome(self, book: Optional[RuleBook], params: AnalysisParams) -> None:
        st.markdown(Components.hero(
            "See where you stand, in one glance.",
            "Paste text, upload a file or point to a URL. ComplianceLens counts which rules you meet, "
            "which you may be breaking, and explains each result on demand."), unsafe_allow_html=True)
        steps = [("1", "Add your activity", "Text, TXT, DOC, DOCX, PDF or a web link."),
                 ("2", "Read the count", "Compliant, violations, review and gaps per category."),
                 ("3", "Open the explanation", "Click Explain to see the rule, evidence and next step.")]
        st.markdown(minify('<div class="cl-steps">' + "".join(
            f'<div class="cl-step"><span class="cl-pill" style="--c:#fff;--bg:{P.accent}">{n}</span>'
            f'<b>{esc(t)}</b><span>{esc(d)}</span></div>' for n, t, d in steps) + "</div>"), unsafe_allow_html=True)
        if st.button("Try the sample activity", type="primary", disabled=book is None, key="welcome_sample"):
            Sidebar(self.ct)._run(book, params, lambda: self.ct.ingestion.from_prompt(SAMPLE_TEXT))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    App().run()


if __name__ == "__main__":
    main()
