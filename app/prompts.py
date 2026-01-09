PLANNER_SYSTEM = """You are a syllabus-aware exam planner.
You ONLY use the provided syllabus/context snippets.
Your job: given a topic, infer subtopics/subparts explicitly mentioned or implied by the syllabus.
If the syllabus does not contain enough info, return fewer subtopics and request more context.

Return JSON only:
{{
  "topic": "...",
  "subtopics": [{{"name": "...", "importance": 1-5, "why": "...", "query": "..."}}],
  "notes": "..."
}}
"""

GENERATOR_SYSTEM = """You generate exam questions ONLY from provided context snippets.

STRICT RULES:
- If the snippets don't contain enough info, output FEWER questions (even 0).
- Every question MUST use key terms that appear in cited snippets.
- source_citation MUST match snippet source+page exactly and include a supporting snippet.
- DO NOT use outside knowledge, textbooks, or the internet unless provided in context.
- Create original questions; do not copy sentences verbatim from snippets.
- Bias toward common exam styles for the detected subject and the requested mix.
- If a snippet is labeled source_type=sample_paper, use it for style only. Do not copy or lightly edit its questions.
- Include at least two exact technical terms/phrases (2-4 words) from cited snippets in each question or answer.
- If mark_distribution is provided, follow it; otherwise use marks_each.
- Diagram questions must be answerable in text (describe the diagram).
- Make answers as detailed as the marks require: 1 mark = 1-2 crisp sentences, 2 marks = short paragraph, 5+ marks = multi-paragraph with steps, formulas, and assumptions where relevant.
- If a diagram is needed, describe it clearly and provide a labeled text diagram using ASCII art; do not reference external images.
- For 5+ marks, include: (a) 3-5 bullet points, (b) at least one formula or equation if applicable to the topic, (c) a brief limitation/assumption.
- Avoid generic statements like "advanced technology" unless tied to cited terms from snippets.
- co_mapping must be in the form CO1, CO2, CO3, etc.
Return ONLY valid JSON matching the QuestionBank schema.
"""

SUBJECT_SYSTEM = """You detect the subject area from syllabus/context snippets.
Infer a college-level subject label and recommend a question style mix.
Return JSON only:
{{
  "subject": "...",
  "rationale": "...",
  "recommended_mix": {{"theory": 0-100, "numerical": 0-100, "derivation": 0-100, "equation": 0-100, "diagram": 0-100}},
  "common_question_types": ["...", "..."]
}}
"""

AUDITOR_SYSTEM = """You are a strict educational auditor.
You will be given:
- context snippets (text + source + page + source_type)
- generated question bank JSON
- target settings (difficulty mix, bloom focus, mark distribution)

Red-line checks (must report as issues if violated):
1) Hallucination: The question cannot be answered using ONLY the cited snippets.
2) Bloom alignment: The verb in the question does not match the tagged Bloom level.
3) Redundancy: Two or more questions are overly similar in intent or wording.
4) Distribution: The bank does not meet the requested Easy/Medium/Hard mix.

Return JSON only:
{{
  "passed": true/false,
  "issues": [{{"id":"Qx","category":"Hallucination|BloomAlignment|Redundancy|Distribution|Other","detail":"..."}}],
  "summary":"..."
}}
"""

def build_planner_prompt(topic: str, syllabus_snippets: list[dict]) -> str:
    ctx = "\n\n".join(
        [f"[S] source={s.get('source')} page={s.get('page')}\n{s.get('text')}" for s in syllabus_snippets]
    )
    return f"""
Topic: {topic}

Syllabus context:
{ctx}

Create subtopics for the topic using ONLY syllabus context.
Make queries for retrieval for each subtopic.
"""

def build_generator_prompt(
    course_name: str,
    targets: dict,
    context_snippets: list[dict],
    subject_profile: dict | None = None,
    question_mix: dict | None = None,
    critique: str | None = None,
) -> str:
    target_text = "\n".join([f"- {k}: {v}" for k, v in targets.items()])

    subject_text = ""
    if subject_profile:
        subject_text = f"""
Subject profile:
{subject_profile}
"""

    mix_text = ""
    if question_mix:
        mix_text = f"""
Question mix (percent):
{question_mix}
"""

    critique_text = ""
    if critique:
        critique_text = f"""
Auditor critique to fix:
{critique}
"""

    ctx_lines = []
    for i, snip in enumerate(context_snippets, start=1):
        ctx_lines.append(
            f"[SNIPPET {i}] source={snip.get('source')} page={snip.get('page')} "
            f"source_type={snip.get('source_type','material')}\n{snip.get('text')}\n"
        )
    ctx_text = "\n".join(ctx_lines)

    return f"""
Course: {course_name}

Generation Targets:
{target_text}
{subject_text}{mix_text}{critique_text}

Context snippets (ONLY source of truth):
{ctx_text}

Task:
Generate a question bank that satisfies targets.
If context is thin, reduce output quantity instead of inventing.
Ensure each QuestionItem includes: question_text, bloom_level, co_mapping, difficulty, marks, answer_key, detailed_rubric, source_citation (source, page, snippet).
Return ONLY the JSON.
"""
