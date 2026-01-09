from __future__ import annotations

from app.agents.generator import GeneratorAgent
from app.agents.auditor import AuditorAgent
from app.schemas import AuditIssue


def run_generation_loop(
    course_name: str,
    targets: dict,
    context_snippets: list[dict],
    subject_profile: dict | None,
    question_mix: dict | None,
    max_iters: int = 2,
    model: str = "gpt-4o-mini",
):
    gen = GeneratorAgent(model=model)
    aud = AuditorAgent(model=model)

    logs: list[dict] = []
    critique: str | None = None
    qb = None
    audit = None

    for i in range(max_iters):
        qb = gen.generate(
            course_name=course_name,
            targets=targets,
            context_snippets=context_snippets,
            subject_profile=subject_profile,
            question_mix=question_mix,
            critique=critique,
        )
        audit = aud.audit(qb.model_dump(), context_snippets, targets)

        # Enforce quantity if context is reasonably sized
        target_count = int(targets.get("num_questions", 0) or 0)
        if target_count and len(context_snippets) >= max(10, target_count):
            if len(qb.questions) < target_count:
                audit.passed = False
                audit.issues.append(
                    AuditIssue(
                        id=None,
                        category="Quantity",
                        detail=f"Requested {target_count} questions, generated {len(qb.questions)}.",
                    )
                )

        logs.append(
            {
                "iteration": i + 1,
                "generator_response": "Regenerated question bank with the latest critique applied."
                if critique
                else "Generated initial question bank.",
                "auditor_summary": audit.summary,
                "auditor_issues": [issue.model_dump() for issue in audit.issues],
                "passed": audit.passed,
            }
        )

        if audit.passed:
            break

        critique = audit.summary + "\n" + "\n".join(
            [f"{iss.category}: {iss.detail}" for iss in audit.issues]
        )

    return qb, audit, logs
