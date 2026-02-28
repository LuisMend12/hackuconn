#!/usr/bin/env python3
"""
content_filter.py

A practical, explainable content-filtering pipeline for AI training data.

Usage:
    python3 content_filter.py --in data.jsonl --out labeled.jsonl --write_cleaned
"""

from __future__ import annotations

import re
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable


# =============================
# Utilities
# =============================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    ent = 0.0
    n = len(s)
    for c in freq.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


# =============================
# Regex Detectors
# =============================

RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
RE_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
RE_CREDIT_CARD = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

SELF_HARM_PHRASES = [
    "kill myself", "end my life", "suicide",
    "i want to die", "no reason to live"
]

SEXUAL_PHRASES = [
    "porn", "nude", "xxx", "onlyfans"
]

VIOLENCE_PHRASES = [
    "gore", "massacre", "torture", "beheading"
]

PROMPT_INJECTION_PHRASES = [
    "ignore previous instructions",
    "system prompt",
    "developer message",
    "bypass safety"
]


# =============================
# Data Classes
# =============================

@dataclass
class Finding:
    category: str
    severity: float
    reason: str
    spans: List[Tuple[int, int]]

@dataclass
class FilterResult:
    decision: str
    risk: float
    categories: Dict[str, float]
    findings: List[Finding]
    cleaned_text: str


# =============================
# Core Filter
# =============================

class ContentFilter:

    CATEGORY_WEIGHTS = {
        "pii": 1.0,
        "self_harm": 1.0,
        "sexual": 0.8,
        "violence": 0.7,
        "prompt_injection": 0.7,
        "low_quality": 0.4,
    }

    KEEP_MAX_RISK = 0.20
    DROP_MIN_RISK = 0.85

    def analyze(self, text: str) -> FilterResult:
        text = normalize_whitespace(text or "")
        findings = []
        scores = {k: 0.0 for k in self.CATEGORY_WEIGHTS.keys()}

        self._detect_pii(text, findings, scores)
        self._detect_phrase_list(text, findings, scores)
        self._detect_prompt_injection(text, findings, scores)
        self._detect_low_quality(text, findings, scores)

        risk = self._aggregate(scores)
        decision = self._decide(risk)

        cleaned = self._redact(text)

        return FilterResult(decision, risk, scores, findings, cleaned)

    def _aggregate(self, scores):
        p = 1.0
        for k, w in self.CATEGORY_WEIGHTS.items():
            s = clamp(scores.get(k, 0), 0, 1)
            p *= (1 - w * s)
        return clamp(1 - p, 0, 1)

    def _decide(self, risk):
        if risk >= self.DROP_MIN_RISK:
            return "DROP"
        if risk <= self.KEEP_MAX_RISK:
            return "KEEP"
        return "QUARANTINE"

    def _add(self, findings, scores, cat, sev, reason, spans):
        findings.append(Finding(cat, sev, reason, spans))
        scores[cat] = max(scores.get(cat, 0), sev)

    def _detect_pii(self, text, findings, scores):
        spans = []
        for regex in [RE_EMAIL, RE_PHONE, RE_SSN, RE_CREDIT_CARD]:
            for m in regex.finditer(text):
                spans.append((m.start(), m.end()))
        if spans:
            self._add(findings, scores, "pii", 0.9, "PII detected", spans)

    def _detect_phrase_list(self, text, findings, scores):
        lower = text.lower()

        def scan(phrases, cat, sev):
            spans = []
            for ph in phrases:
                idx = lower.find(ph)
                while idx != -1:
                    spans.append((idx, idx + len(ph)))
                    idx = lower.find(ph, idx + 1)
            if spans:
                self._add(findings, scores, cat, sev, f"{cat} cue detected", spans)

        scan(SELF_HARM_PHRASES, "self_harm", 0.9)
        scan(SEXUAL_PHRASES, "sexual", 0.7)
        scan(VIOLENCE_PHRASES, "violence", 0.7)

    def _detect_prompt_injection(self, text, findings, scores):
        lower = text.lower()
        spans = []
        for ph in PROMPT_INJECTION_PHRASES:
            idx = lower.find(ph)
            while idx != -1:
                spans.append((idx, idx + len(ph)))
                idx = lower.find(ph, idx + 1)
        if spans:
            self._add(findings, scores, "prompt_injection", 0.7, "Prompt injection detected", spans)

    def _detect_low_quality(self, text, findings, scores):
        if len(text) < 5:
            self._add(findings, scores, "low_quality", 0.6, "Very short text", [])
        if shannon_entropy(text) > 5.5:
            self._add(findings, scores, "low_quality", 0.6, "High entropy text", [])

    def _redact(self, text):
        text = RE_EMAIL.sub("[REDACTED_EMAIL]", text)
        text = RE_PHONE.sub("[REDACTED_PHONE]", text)
        text = RE_SSN.sub("[REDACTED_SSN]", text)
        text = RE_CREDIT_CARD.sub("[REDACTED_CC]", text)
        return text


# =============================
# JSONL I/O
# =============================

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =============================
# CLI
# =============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--write_cleaned", action="store_true")
    parser.add_argument("--keep_only", action="store_true")
    args = parser.parse_args()

    filt = ContentFilter()
    output_rows = []

    for row in read_jsonl(args.in_path):
        text = str(row.get(args.text_field, ""))
        res = filt.analyze(text)

        row_out = dict(row)
        row_out["filter_decision"] = res.decision
        row_out["filter_risk"] = round(res.risk, 4)
        row_out["filter_categories"] = res.categories
        row_out["filter_findings"] = [
            {
                "category": f.category,
                "severity": f.severity,
                "reason": f.reason,
                "spans": f.spans,
            }
            for f in res.findings
        ]

        if args.write_cleaned:
            row_out["cleaned_text"] = res.cleaned_text

        if args.keep_only and res.decision != "KEEP":
            continue

        output_rows.append(row_out)

    write_jsonl(args.out_path, output_rows)
    print(f"Done. Wrote: {args.out_path}")


if __name__ == "__main__":
    main()