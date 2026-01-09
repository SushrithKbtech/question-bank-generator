from __future__ import annotations
import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{2,4}[-.\s]?){2,4}\d{2,4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def detect_pii(text: str) -> list[str]:
    hits: list[str] = []
    if EMAIL_RE.search(text):
        hits.append("email")
    if SSN_RE.search(text):
        hits.append("ssn")
    if PHONE_RE.search(text):
        hits.append("phone")
    return hits


def scan_pages_for_pii(pages: list[dict]) -> list[dict]:
    findings: list[dict] = []
    for page in pages:
        text = page.get("text", "")
        hits = detect_pii(text)
        if hits:
            findings.append({"page": page.get("page"), "types": hits})
    return findings
