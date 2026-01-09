from __future__ import annotations
import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


def questions_to_dataframe(questions: list[dict]) -> pd.DataFrame:
    rows = []
    for q in questions:
        citations = "; ".join(
            [f"{c.get('source')} p{c.get('page')}" for c in q.get("source_citation", [])]
        )
        rows.append(
            {
                "id": q.get("id"),
                "question_text": q.get("question_text"),
                "bloom_level": q.get("bloom_level"),
                "co_mapping": q.get("co_mapping"),
                "difficulty": q.get("difficulty"),
                "marks": q.get("marks"),
                "answer_key": q.get("answer_key"),
                "detailed_rubric": q.get("detailed_rubric"),
                "source_citation": citations,
            }
        )
    return pd.DataFrame(rows)


def questions_to_csv_bytes(questions: list[dict]) -> bytes:
    df = questions_to_dataframe(questions)
    return df.to_csv(index=False).encode("utf-8")


def questions_to_pdf_bytes(questions: list[dict], coverage_report: dict) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - inch

    c.setFont("Helvetica-Bold", 14)
    c.drawString(inch, y, "Question Bank")
    y -= 0.4 * inch

    c.setFont("Helvetica", 10)
    c.drawString(inch, y, f"Total questions: {coverage_report.get('total_questions', 0)}")
    y -= 0.3 * inch

    c.setFont("Helvetica-Bold", 11)
    c.drawString(inch, y, "Coverage Report")
    y -= 0.25 * inch

    c.setFont("Helvetica", 9)
    for key in ["co_distribution", "bloom_distribution", "difficulty_distribution"]:
        c.drawString(inch, y, f"{key}: {coverage_report.get(key)}")
        y -= 0.2 * inch
        if y < inch:
            c.showPage()
            y = height - inch

    c.setFont("Helvetica-Bold", 11)
    c.drawString(inch, y, "Questions")
    y -= 0.3 * inch

    c.setFont("Helvetica", 9)
    for q in questions:
        lines = [
            f"{q.get('id')} ({q.get('marks')} marks) [{q.get('co_mapping')} | {q.get('bloom_level')} | {q.get('difficulty')}]",
            q.get("question_text", ""),
            "Answer: " + (q.get("answer_key", "") or ""),
            "Rubric: " + (q.get("detailed_rubric", "") or ""),
        ]
        for line in lines:
            for chunk in _wrap_text(line, width - 2 * inch):
                c.drawString(inch, y, chunk)
                y -= 0.18 * inch
                if y < inch:
                    c.showPage()
                    y = height - inch
        y -= 0.2 * inch
        if y < inch:
            c.showPage()
            y = height - inch

    c.save()
    return buffer.getvalue()


def _wrap_text(text: str, max_width: float) -> list[str]:
    # Approximate wrap by character count (90 chars per line).
    max_chars = int(max_width / 6)
    words = text.split()
    lines = []
    line = ""
    for w in words:
        if len(line) + len(w) + 1 > max_chars:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    return lines
