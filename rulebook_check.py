#!/usr/bin/env python3
"""
rulebook_check.py - NLTK-based compliance checker for the
"Compliance Rulebook: International Students in Montreal, Quebec, Canada".

It reads the rulebook (Markdown), reads a student's activity text (diary, self-report,
checklist answers, emails...), and counts, per rule category, how many rules are
compliant, possibly violated, need review, or are not addressed at all.

Usage
  python rulebook_check.py rulebook.md activity.txt
  python rulebook_check.py rulebook.md notes/ --detail --json out.json --csv out.csv
  cat activity.txt | python rulebook_check.py rulebook.md -

NLTK is used for: sentence + word tokenizing, POS tagging, WordNet lemmatizing,
stop-word removal, and negation marking (nltk.sentiment.util.mark_negation).
This is a keyword/heuristic screen, NOT legal advice and NOT proof of (non)compliance.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import nltk

# ----------------------------------------------------------------------------
# NLTK resources (downloaded once; for offline use run:
#   python -m nltk.downloader punkt punkt_tab stopwords wordnet \
#          averaged_perceptron_tagger averaged_perceptron_tagger_eng)
# ----------------------------------------------------------------------------
RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]


def ensure_nltk() -> None:
    for path, pkg in RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
    try:  # final smoke test
        from nltk.tokenize import word_tokenize
        nltk.pos_tag(word_tokenize("test sentence"))
        from nltk.corpus import stopwords, wordnet
        stopwords.words("english")
        wordnet.synsets("test")
    except LookupError as exc:
        sys.exit(f"NLTK data missing and could not be downloaded:\n{exc}\n"
                 "Run: python -m nltk.downloader punkt punkt_tab stopwords wordnet "
                 "averaged_perceptron_tagger averaged_perceptron_tagger_eng")


ensure_nltk()

from nltk.corpus import stopwords, wordnet            # noqa: E402
from nltk.sentiment.util import mark_negation          # noqa: E402
from nltk.stem import WordNetLemmatizer                # noqa: E402
from nltk.tokenize import sent_tokenize, word_tokenize  # noqa: E402

# ----------------------------------------------------------------------------
# Rulebook model
# ----------------------------------------------------------------------------
CATEGORIES = {
    "FED": "Federal",
    "QC": "Provincial (Quebec)",
    "SCH": "Schools",
    "ASC": "Associations",
    "CORP": "Corporate",
    "ENG": "Engineering ethics",
    "IP": "Painting & IP",
    "UNI": "Student unions",
}
LEVELS = {"\U0001F534": "critical", "\U0001F7E0": "legal/financial", "\U0001F7E1": "good practice"}

# **FED-1 🔴 Keep valid status.** body text...
RULE_RE = re.compile(r"^\*\*(?P<id>[A-Z]+-\d+)\s+(?P<lvl>\S+)\s+(?P<title>.+?)\*\*\s*(?P<body>.*)$")


def clean_md(text: str) -> str:
    text = re.sub(r"[*_`#>]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_rules(md_path: Path) -> list[dict]:
    rules = []
    for line in md_path.read_text(encoding="utf-8").splitlines():
        m = RULE_RE.match(line.strip())
        if not m:
            continue
        prefix = m["id"].split("-")[0]
        if prefix not in CATEGORIES:
            continue
        rules.append({
            "id": m["id"],
            "category": prefix,
            "level": LEVELS.get(m["lvl"], "unrated"),
            "title": clean_md(m["title"]).rstrip("."),
            "text": clean_md(m["body"]),
        })
    return rules


# ----------------------------------------------------------------------------
# Language analysis
# ----------------------------------------------------------------------------
GENERIC = {
    "must", "may", "also", "canada", "canadian", "quebec", "student", "school",
    "rule", "check", "ensure", "use", "make", "get", "take", "need", "know",
    "follow", "give", "include", "example", "often", "way",
    "year", "time", "something", "someone", "anyone",
}
# cue lemmas (verbs/nouns) -----------------------------------------------------
COMPLIANT_CUES = {
    "renew", "confirm", "approve", "log", "file", "register", "obtain", "submit",
    "complete", "verify", "maintain", "comply", "track", "report", "sign",
    "secure", "enrol", "enroll", "pay", "insure", "apply", "declare", "disclose",
    "supervise", "cite", "license", "licence", "attend", "authorize", "authorise",
    "keep", "record", "document", "receive", "hold", "extend", "invoice", "disclose",
}
VIOLATION_CUES = {
    "exceed", "skip", "ignore", "forget", "miss", "plagiarize", "plagiarise",
    "plagiarism", "cheat", "fabricate", "bribe", "overstay", "falsify", "forge",
    "steal", "breach", "violate", "vandalize", "vandalise", "blockade", "harass",
    "haze", "conceal", "evade", "underreport",
}
VIOLATION_PHRASES = [re.compile(p, re.I) for p in (
    r"\bwithout (?:a |an |the |my |any )?(?:valid )?(?:permit|permission|authori[sz]ation|"
    r"approval|sin|licen[sc]e|insurance|contract|consent|supervision)\b",
    r"\b(?:has|have|had|is|was|already) expired\b",
    r"\bunpaid (?:work|job|internship|trial)\b",
    r"\bpaid (?:in )?cash\b|\bcash[- ]in[- ]hand\b",
    r"\b(?:over|above|more than|exceed\w*) (?:the )?(?:24|limit|cap)\b",
    r"\bopt(?:ed|s)? out of [^.]{0,40}(?:health|insurance)",
    r"\b(?:copied|traced) (?:from|someone|another|an? )",
    r"\b(?:didn'?t|did not|never) (?:renew|file|log|declare|register|report)\b",
)]
CLAUSE_SPLIT = re.compile(r"[;,]|\b(?:and|but|however|while|although|yet)\b", re.I)
HOURS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b[^.]{0,30}\b(?:week|wk)\b", re.I)
BREAK_WORDS = re.compile(r"\b(?:break|holiday|vacation|summer|reading week|winter|scheduled)\b", re.I)
HOURS_RULES = ("FED-4", "FED-5")


class Analyzer:
    def __init__(self) -> None:
        self.lem = WordNetLemmatizer()
        self.stop = set(stopwords.words("english"))

    @staticmethod
    def _wn_pos(tag: str):
        return {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN}.get(tag[:1])

    def sentences(self, text: str) -> list[str]:
        return [s.strip() for s in sent_tokenize(clean_md(text)) if len(s.strip()) > 3]

    def lemmas(self, sentence: str) -> list[str]:
        out = []
        for tok, tag in nltk.pos_tag(word_tokenize(sentence)):
            pos = self._wn_pos(tag)
            w = tok.lower()
            if not pos or not w.isalpha() or len(w) < 3 or w in self.stop:
                continue
            lem = self.lem.lemmatize(w, pos)
            if lem not in GENERIC:
                out.append(lem)
        return out

    def polarity(self, sentence: str) -> tuple[int, int]:
        """Return (compliance_signals, violation_signals).

        Negation is scoped to a clause (split at commas, 'and', 'but'...), so
        'I never plagiarized and I cited my sources' is not misread.
        """
        c = v = 0
        for clause in CLAUSE_SPLIT.split(sentence):
            tokens = [t.lower() for t in word_tokenize(clause)]
            for tok in mark_negation(tokens):
                neg = tok.endswith("_NEG")
                base = tok[:-4] if neg else tok
                forms = {base, self.lem.lemmatize(base, "v"), self.lem.lemmatize(base, "n")}
                if forms & VIOLATION_CUES:
                    if neg: c += 1
                    else: v += 1
                elif forms & COMPLIANT_CUES:
                    if neg: v += 1
                    else: c += 1
        v += 2 * sum(1 for p in VIOLATION_PHRASES if p.search(sentence))  # explicit patterns weigh more
        return c, v


# ----------------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------------
def rule_keywords(an: Analyzer, rule: dict, top_n: int = 15) -> dict[str, int]:
    weights: Counter = Counter()
    for lem in an.lemmas(rule["title"]):
        weights[lem] += 2
    for lem in an.lemmas(rule["text"]):
        weights[lem] += 1
    return dict(weights.most_common(top_n))


def read_inputs(paths: list[str]) -> str:
    chunks = []
    for p in paths:
        if p == "-":
            chunks.append(sys.stdin.read())
            continue
        path = Path(p)
        files = sorted(f for f in path.rglob("*") if f.suffix.lower() in {".txt", ".md"}) \
            if path.is_dir() else [path]
        for f in files:
            chunks.append(f.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def evaluate(rules, text, min_score, min_hits, top_k):
    an = Analyzer()
    kw = {r["id"]: rule_keywords(an, r) for r in rules}
    totals = {rid: sum(w.values()) or 1 for rid, w in kw.items()}
    evidence = defaultdict(list)          # rule id -> [(label, sentence)]
    sentences = an.sentences(text)

    for sent in sentences:
        lem_set = set(an.lemmas(sent))
        c, v = an.polarity(sent)
        label = "violation" if v > c else "compliant" if c > v else "neutral"
        scored = []
        for r in rules:
            hits = lem_set & kw[r["id"]].keys()
            score = sum(kw[r["id"]][h] for h in hits) / totals[r["id"]]
            if len(hits) >= min_hits and score >= min_score:
                scored.append((score, r["id"]))
        for _, rid in sorted(scored, reverse=True)[:top_k]:
            evidence[rid].append((label, sent))

        # numeric check: > 24 off-campus hours/week during term
        m = HOURS_RE.search(sent)
        if m and float(m.group(1)) > 24 and not BREAK_WORDS.search(sent):
            for rid in HOURS_RULES:
                if any(r["id"] == rid for r in rules):
                    evidence[rid].append(("violation", sent))

    results = []
    for r in rules:
        best: dict[str, str] = {}
        rank = {"violation": 2, "compliant": 1, "neutral": 0}
        for lab, snt in evidence.get(r["id"], []):
            if snt not in best or rank[lab] > rank[best[snt]]:
                best[snt] = lab
        ev = [(lab, snt) for snt, lab in best.items()]
        labels = {lab for lab, _ in ev}
        if not ev:
            status = "NOT_ADDRESSED"
        elif "violation" in labels:
            status = "POSSIBLE_VIOLATION"
        elif "compliant" in labels:
            status = "COMPLIANT"
        else:
            status = "NEEDS_REVIEW"
        results.append({**r, "status": status, "evidence": ev[:3]})
    return results, len(sentences)


# ----------------------------------------------------------------------------
# Reporting ("the count")
# ----------------------------------------------------------------------------
def tally(results):
    table = {k: Counter() for k in CATEGORIES}
    for r in results:
        table[r["category"]][r["status"]] += 1
        table[r["category"]]["TOTAL"] += 1
        if r["status"] == "POSSIBLE_VIOLATION" and r["level"] == "critical":
            table[r["category"]]["CRITICAL_VIOL"] += 1
    return table


def rate(c: Counter) -> str:
    addressed = c["COMPLIANT"] + c["POSSIBLE_VIOLATION"] + c["NEEDS_REVIEW"]
    return f"{100 * c['COMPLIANT'] / addressed:.0f}%" if addressed else "-"


def print_report(results, n_sentences, detail):
    table = tally(results)
    head = f"{'CATEGORY':<22}{'RULES':>6}{'OK':>5}{'VIOL':>6}{'REVIEW':>7}{'NONE':>6}{'CRIT':>6}{'RATE':>6}  VIOLATIONS"
    print(f"\nSentences analysed: {n_sentences}   Rules loaded: {len(results)}\n")
    print(head)
    print("-" * len(head) + "-----")
    grand = Counter()
    for key, name in CATEGORIES.items():
        c = table[key]
        if not c["TOTAL"]:
            continue
        grand.update(c)
        print(f"{name:<22}{c['TOTAL']:>6}{c['COMPLIANT']:>5}{c['POSSIBLE_VIOLATION']:>6}"
              f"{c['NEEDS_REVIEW']:>7}{c['NOT_ADDRESSED']:>6}{c['CRITICAL_VIOL']:>6}{rate(c):>6}  "
              f"{'#' * c['POSSIBLE_VIOLATION']}")
    print("-" * len(head) + "-----")
    print(f"{'TOTAL':<22}{grand['TOTAL']:>6}{grand['COMPLIANT']:>5}{grand['POSSIBLE_VIOLATION']:>6}"
          f"{grand['NEEDS_REVIEW']:>7}{grand['NOT_ADDRESSED']:>6}{grand['CRITICAL_VIOL']:>6}{rate(grand):>6}")
    print("\nOK=compliant evidence  VIOL=possible violation  REVIEW=mentioned, unclear  "
          "NONE=not addressed  CRIT=critical-level violations")
    print("RATE = OK / (OK + VIOL + REVIEW), i.e. among rules your text actually mentions.")

    flagged = [r for r in results if r["status"] == "POSSIBLE_VIOLATION"]
    if flagged:
        print("\nPOSSIBLE VIOLATIONS (critical first):")
        for r in sorted(flagged, key=lambda x: (x["level"] != "critical", x["id"])):
            print(f"  [{r['level']}] {r['id']} {r['title']}")
            if detail:
                for lab, s in r["evidence"]:
                    print(f"      - ({lab}) {s[:140]}")
    if detail:
        review = [r for r in results if r["status"] == "NEEDS_REVIEW"]
        if review:
            print("\nNEEDS REVIEW:")
            for r in review:
                print(f"  {r['id']} {r['title']}")
                for lab, s in r["evidence"]:
                    print(f"      - ({lab}) {s[:140]}")
    print("\nHeuristic screen only. Not legal advice; confirm with your school's ISS office.\n")


def write_exports(results, json_path, csv_path):
    if json_path:
        Path(json_path).write_text(json.dumps(
            {"counts": {k: dict(v) for k, v in tally(results).items()},
             "rules": [{**r, "evidence": [{"label": l, "sentence": s} for l, s in r["evidence"]]}
                       for r in results]}, indent=2, ensure_ascii=False), encoding="utf-8")
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "category", "level", "status", "title", "evidence"])
            for r in results:
                w.writerow([r["id"], CATEGORIES[r["category"]], r["level"], r["status"], r["title"],
                            " | ".join(s for _, s in r["evidence"])])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rulebook", help="rulebook Markdown file (exported from the doc)")
    ap.add_argument("inputs", nargs="+", help="activity text file(s), folder(s), or - for stdin")
    ap.add_argument("--min-score", type=float, default=0.12, help="min keyword-overlap score (default 0.12)")
    ap.add_argument("--min-hits", type=int, default=2, help="min distinct keyword hits (default 2)")
    ap.add_argument("--top-k", type=int, default=3, help="max rules matched per sentence (default 3)")
    ap.add_argument("--detail", action="store_true", help="show evidence sentences")
    ap.add_argument("--json", dest="json_path", help="write full results to JSON")
    ap.add_argument("--csv", dest="csv_path", help="write per-rule results to CSV")
    ap.add_argument("--fail-on-critical", action="store_true", help="exit 1 if any critical violation")
    args = ap.parse_args()

    rules = parse_rules(Path(args.rulebook))
    if not rules:
        sys.exit("No rules found. Expected lines like: **FED-1 \U0001F534 Keep valid status.** ...")
    text = read_inputs(args.inputs)
    if not text.strip():
        sys.exit("Input text is empty.")

    results, n = evaluate(rules, text, args.min_score, args.min_hits, args.top_k)
    print_report(results, n, args.detail)
    write_exports(results, args.json_path, args.csv_path)
    if args.fail_on_critical and any(
            r["status"] == "POSSIBLE_VIOLATION" and r["level"] == "critical" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
