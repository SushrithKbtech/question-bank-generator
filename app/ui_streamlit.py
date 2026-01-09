import streamlit as st

from app.rag.ingest import save_uploaded_pdf, extract_pages_from_pdf
from app.rag.chunks import chunk_pages
from app.rag.vectorstore import get_client, get_collection, get_embedder, upsert_chunks
from app.rag.retrieve import retrieve_top_k_strict
from app.rag.pii import scan_pages_for_pii

from app.agents.generator import GeneratorAgent
from app.agents.loop import run_generation_loop
from app.reporting import compute_coverage_report
from app.export import questions_to_csv_bytes, questions_to_pdf_bytes

st.set_page_config(page_title="Outcome-QBank (Syllabus-Aware)", layout="wide")
st.title("Outcome-Aligned Question Bank - Syllabus-Aware (Planner -> RAG -> Generator -> Audit)")

st.markdown(
    """
<style>
    .block-container { padding-top: 1.2rem; }
    .stExpander { border-radius: 8px; }
    .stDownloadButton button { width: 100%; }
</style>
""",
    unsafe_allow_html=True,
)

# -------- State --------
st.session_state.setdefault("ingested", False)
st.session_state.setdefault("syllabus_snippets", [])
st.session_state.setdefault("planned", None)  # dict plan
st.session_state.setdefault("ctx", [])
st.session_state.setdefault("last_query_debug", [])
st.session_state.setdefault("subject_profile", None)
st.session_state.setdefault(
    "mix",
    {"theory": 35, "numerical": 30, "derivation": 15, "equation": 10, "diagram": 10},
)
st.session_state.setdefault("last_subject_mix", None)
st.session_state.setdefault("logs", [])
st.session_state.setdefault("last_audit", None)
st.session_state.setdefault("last_qb", None)
st.session_state.setdefault("coverage_report", None)

# -------- Sidebar controls --------
with st.sidebar:
    st.header("Vector DB")
    collection_name = st.text_input("Chroma collection", "course_material")

    st.header("Chunking")
    chunk_size = st.slider("Chunk size (chars)", 600, 2000, 1000, 100)
    overlap = st.slider("Overlap (chars)", 0, 400, 200, 20)

    st.header("Retrieval")
    top_k = st.slider("Top-K per subtopic", 2, 10, 4)
    min_importance = st.slider("Min importance to retrieve", 1, 5, 2)
    max_total_ctx = st.slider("Max total context chunks", 10, 80, 24, 2)
    include_sample_papers = st.checkbox("Include sample papers in retrieval", value=True)

    st.header("Generation")
    num_q = st.slider("Total questions", 5, 100, 20)
    marks_each = st.selectbox("Marks each", [2, 5, 10], index=0)
    bloom_focus = st.selectbox("Bloom focus", ["Mixed", "Apply+Analyze heavy", "Remember+Understand heavy"], index=1)
    difficulty_mix = st.selectbox("Difficulty mix", ["Mostly Medium", "Easy/Medium", "Medium/Hard"], index=0)
    use_auto_mix = st.checkbox("Use detected subject mix", value=True)

    subject_profile = st.session_state.get("subject_profile")
    if use_auto_mix and subject_profile:
        rec = subject_profile.get("recommended_mix")
        if rec and st.session_state.get("last_subject_mix") != rec:
            merged = dict(st.session_state.mix)
            merged.update(rec)
            st.session_state.mix = merged
            st.session_state.last_subject_mix = rec

    st.markdown("Question mix (percent)")
    mix_theory = st.slider("Theory / conceptual", 0, 100, st.session_state.mix.get("theory", 40))
    mix_numerical = st.slider("Numerical / problem-solving", 0, 100, st.session_state.mix.get("numerical", 30))
    mix_derivation = st.slider("Derivation / proof", 0, 100, st.session_state.mix.get("derivation", 20))
    mix_equation = st.slider("Equation / mechanism", 0, 100, st.session_state.mix.get("equation", 10))
    mix_diagram = st.slider("Diagram-based (describe the diagram)", 0, 100, st.session_state.mix.get("diagram", 10))
    st.session_state.mix = {
        "theory": mix_theory,
        "numerical": mix_numerical,
        "derivation": mix_derivation,
        "equation": mix_equation,
        "diagram": mix_diagram,
    }

    st.markdown("Marks distribution (optional)")
    use_mark_dist = st.checkbox("Use 1/2/5-mark distribution", value=False)
    one_mark = st.number_input("Count of 1-mark questions", min_value=0, max_value=100, value=0, step=1)
    two_mark = st.number_input("Count of 2-mark questions", min_value=0, max_value=100, value=0, step=1)
    five_mark = st.number_input("Count of 5-mark questions", min_value=0, max_value=100, value=0, step=1)
    include_numerical = st.checkbox("Include numerical questions", value=True)
    include_diagram = st.checkbox("Include diagram-based questions", value=False)

# -------- Upload PDFs --------
st.subheader("1) Upload PDFs (Syllabus + optional Textbook)")
col_mat, col_samp = st.columns(2)
with col_mat:
    uploaded_files = st.file_uploader("Upload course material PDFs", type=["pdf"], accept_multiple_files=True)
with col_samp:
    sample_files = st.file_uploader(
        "Upload sample papers / practice questions (optional)",
        type=["pdf"],
        accept_multiple_files=True,
    )
    st.caption("Sample papers are used for style only, never copied.")

col_ing1, col_ing2 = st.columns([1, 2])
with col_ing1:
    ingest_clicked = st.button("Ingest PDFs into Vector DB")
with col_ing2:
    st.caption("Tip: Upload syllabus + lecture notes/textbook chapter for high accuracy. Syllabus alone = shallow questions.")

if (uploaded_files or sample_files) and ingest_clicked:
    with st.spinner("Ingesting PDFs..."):
        client = get_client("data/vector_db")
        embedder = get_embedder()
        collection = get_collection(client, collection_name, embedder)

        pii_report = []
        total_chunks = 0
        for uf in uploaded_files or []:
            pdf_path = save_uploaded_pdf(uf, save_dir="data/uploads")
            pages = extract_pages_from_pdf(pdf_path)
            pii_hits = scan_pages_for_pii(pages)
            if pii_hits:
                pii_report.append({"file": uf.name, "findings": pii_hits})
                continue

            chunks = chunk_pages(
                pages,
                source=uf.name,
                chunk_size=chunk_size,
                overlap=overlap,
                source_type="material",
            )
            total_chunks += len(chunks)

            upsert_chunks(collection, chunks)

        for sf in sample_files or []:
            pdf_path = save_uploaded_pdf(sf, save_dir="data/uploads")
            pages = extract_pages_from_pdf(pdf_path)
            pii_hits = scan_pages_for_pii(pages)
            if pii_hits:
                pii_report.append({"file": sf.name, "findings": pii_hits})
                continue

            chunks = chunk_pages(
                pages,
                source=sf.name,
                chunk_size=chunk_size,
                overlap=overlap,
                source_type="sample_paper",
            )
            total_chunks += len(chunks)

            upsert_chunks(collection, chunks)

        if pii_report:
            st.warning("PII detected in uploads. Proceeding as requested.")
            st.json(pii_report)

        st.session_state.ingested = True

    if st.session_state.ingested:
        st.success(f"Ingested OK. chunks={total_chunks}")

st.divider()

# -------- Topic-driven workflow --------
st.subheader("2) Topic -> Subtopics (Planner) -> Retrieval -> Generate")

course_name = st.text_input("Course name", value="Fundamentals of Quantum Computing")
topic = st.text_input("Enter your topic", value="Grover's Search Algorithm")

# -------- Load syllabus context --------
st.subheader("2A) Load syllabus context (auto)")
auto_queries = ["Course outcomes", "Module - 3", "Quantum Algorithms", "Syllabus"]
custom_queries = st.text_input("Optional: add extra syllabus queries (comma-separated)", value="")

if st.button("Load syllabus context (auto)"):
    if not st.session_state.ingested:
        st.error("Ingest PDFs first.")
    else:
        client = get_client("data/vector_db")
        embedder = get_embedder()
        collection = get_collection(client, collection_name, embedder)

        syllabus_ctx = []
        for q in auto_queries:
            syllabus_ctx += retrieve_top_k_strict(
                collection,
                q,
                k=3,
                allowed_source_types=["material"],
            )

        if custom_queries.strip():
            for q in [x.strip() for x in custom_queries.split(",") if x.strip()]:
                syllabus_ctx += retrieve_top_k_strict(
                    collection,
                    q,
                    k=3,
                    allowed_source_types=["material"],
                )

        # de-dup by source+page+text start
        seen = set()
        cleaned = []
        for c in syllabus_ctx:
            key = (c.get("source"), c.get("page"), (c.get("text") or "")[:120])
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(c)

        st.session_state.syllabus_snippets = cleaned
        st.success(f"Loaded {len(cleaned)} syllabus snippets OK.")

        with st.expander("Show syllabus snippets"):
            for s in cleaned:
                st.markdown(f"**{s['source']} | p{s['page']} | kw={s.get('kw_score', '-') }**")
                st.write(s["text"])
                st.divider()

st.divider()

# -------- Subject detection --------
st.subheader("2B) Detect subject profile (auto)")

detect_clicked = st.button("Detect subject from syllabus context")
if detect_clicked:
    if not st.session_state.syllabus_snippets:
        st.error("Load syllabus context first.")
    else:
        with st.spinner("Detecting subject profile..."):
            agent = GeneratorAgent(model="gpt-4o-mini")
            profile = agent.classify_subject(st.session_state.syllabus_snippets)
            st.session_state.subject_profile = profile.model_dump()
        st.success("Subject profile detected.")

if st.session_state.get("subject_profile"):
    st.markdown("### Subject profile")
    st.json(st.session_state.get("subject_profile"))

st.divider()

# -------- Planner --------
st.subheader("3) Planner (Syllabus-aware)")

plan_clicked = st.button("Plan subtopics for this topic")

if plan_clicked:
    if not st.session_state.syllabus_snippets:
        st.error("Load syllabus context first.")
    else:
        with st.spinner("Planning subtopics from syllabus..."):
            agent = GeneratorAgent(model="gpt-4o-mini")
            plan_obj = agent.plan(topic, st.session_state.syllabus_snippets)

            # IMPORTANT: convert Pydantic -> dict for session_state + later .get usage
            plan_dict = plan_obj.model_dump()  # <-- fixes your crash
            st.session_state.planned = plan_dict

        st.success("Plan created OK.")

if st.session_state.planned:
    st.markdown("### Current plan")
    st.json(st.session_state.planned)

    # Make plan editable (helps you enforce exam relevance)
    st.markdown("### Edit subtopics (optional)")
    subtopics = st.session_state.planned.get("subtopics", [])
    if subtopics:
        edited = st.data_editor(
            subtopics,
            width="stretch",
            num_rows="dynamic",
            column_config={
                "importance": st.column_config.NumberColumn("importance", min_value=1, max_value=5, step=1),
            },
        )
        # Save back edits
        st.session_state.planned["subtopics"] = edited
    else:
        st.warning("Planner returned no subtopics. This usually means syllabus context is too thin. Upload notes/textbook PDF.")

st.divider()

# -------- Retrieval per subtopic --------
st.subheader("4) Retrieve best chunks per subtopic")

if st.button("Retrieve context for planned subtopics"):
    if not st.session_state.planned:
        st.error("Run Planner first.")
    elif not st.session_state.ingested:
        st.error("Ingest PDFs first.")
    else:
        client = get_client("data/vector_db")
        embedder = get_embedder()
        collection = get_collection(client, collection_name, embedder)

        planned_subs = st.session_state.planned.get("subtopics", []) or []
        # filter by min importance
        planned_subs = [s for s in planned_subs if int(s.get("importance", 3)) >= int(min_importance)]

        if not planned_subs:
            st.warning("No subtopics meet the minimum importance. Lower the min importance slider or edit subtopics.")
        else:
            all_ctx = []
            debug = []

            for sub in planned_subs:
                q = sub.get("query") or sub.get("name")
                allowed_types = ["material", "sample_paper"] if include_sample_papers else ["material"]
                hits = retrieve_top_k_strict(
                    collection,
                    q,
                    k=top_k,
                    allowed_source_types=allowed_types,
                )

                debug.append({"subtopic": sub.get("name"), "query": q, "hits": len(hits)})

                for h in hits:
                    h["subtopic"] = sub.get("name")
                    h["importance"] = sub.get("importance", 3)
                all_ctx += hits

            # rank overall by importance then kw_score then distance
            all_ctx.sort(
                key=lambda x: (
                    -int(x.get("importance", 3)),
                    -int(x.get("kw_score", 0)),
                    float(x.get("distance", 9e9)),
                )
            )

            # keep top N overall
            all_ctx = all_ctx[:max_total_ctx]
            st.session_state.ctx = all_ctx
            st.session_state.last_query_debug = debug

            if not all_ctx:
                st.warning("No usable context retrieved. Upload textbook/notes PDF for deeper content.")
            else:
                st.success(f"Retrieved {len(all_ctx)} best chunks OK.")

                with st.expander("Retrieval debug (subtopic -> query -> hits)"):
                    st.write(st.session_state.last_query_debug)

                with st.expander("Show retrieved chunks"):
                    for c in all_ctx:
                        st.markdown(
                            f"**[{c.get('subtopic')}] {c['source']} | p{c['page']} | kw={c.get('kw_score')} dist={c['distance']:.4f}**"
                        )
                        st.write(c["text"])
                        st.divider()

st.divider()

# -------- Generate + Audit --------
st.subheader("5) Generate Questions + Audit (Double-Agent)")

difficulty_distribution_map = {
    "Mostly Medium": {"Easy": 20, "Medium": 60, "Hard": 20},
    "Easy/Medium": {"Easy": 40, "Medium": 40, "Hard": 20},
    "Medium/Hard": {"Easy": 20, "Medium": 40, "Hard": 40},
}

targets = {
    "topic": topic,
    "num_questions": num_q,
    "marks_each": marks_each,
    "bloom_focus": bloom_focus,
    "difficulty_mix": difficulty_mix,
    "difficulty_distribution": difficulty_distribution_map.get(difficulty_mix),
    "mark_distribution": {"1": one_mark, "2": two_mark, "5": five_mark} if use_mark_dist else None,
    "question_type_preferences": {
        "include_numerical": include_numerical,
        "include_diagram": include_diagram,
    },
    "instruction": (
        "Generate exam-relevant questions strictly from context. "
        "Try to reach the requested count if context supports it. "
        "If context is shallow, output fewer questions instead of inventing. "
        "Prefer questions aligned with syllabus structure."
    ),
}

if st.button("Generate (and audit)"):
    if not st.session_state.ctx:
        st.error("Retrieve context first.")
    else:
        if use_mark_dist:
            dist_total = one_mark + two_mark + five_mark
            if dist_total > num_q:
                st.warning("Mark distribution exceeds total questions; generator will prioritize distribution.")
        mix_sum = sum(st.session_state.mix.values()) or 1
        norm_mix = {k: int(round(v * 100 / mix_sum)) for k, v in st.session_state.mix.items()}
        with st.spinner("Generating question bank..."):
            qb, audit, logs = run_generation_loop(
                course_name=course_name,
                targets=targets,
                context_snippets=st.session_state.ctx,
                subject_profile=st.session_state.get("subject_profile"),
                question_mix=norm_mix,
                max_iters=3,
                model="gpt-4o-mini",
            )

        st.session_state.last_qb = qb.model_dump() if qb else None
        st.session_state.last_audit = audit.model_dump() if audit else None
        st.session_state.logs = logs

        if not qb:
            st.error("Generation failed. Try again with more context.")
        else:
            if audit and not audit.passed:
                st.error("Audit did not pass after 3 iterations. Review issues below.")
            else:
                st.success("Audit passed.")

            if audit:
                st.caption(audit.summary)
                if audit.issues:
                    with st.expander("Audit issues"):
                        st.json([issue.model_dump() for issue in audit.issues])

            st.subheader("Improvement loop logs (Generator <-> Auditor)")
            st.json(st.session_state.logs)

            qb_dict = qb.model_dump()
            coverage = compute_coverage_report(qb_dict.get("questions", []))
            st.session_state.coverage_report = coverage

            st.subheader("Coverage report")
            st.json(coverage)

            csv_bytes = questions_to_csv_bytes(qb_dict.get("questions", []))
            pdf_bytes = questions_to_pdf_bytes(qb_dict.get("questions", []), coverage)

            st.download_button("Download CSV", data=csv_bytes, file_name="question_bank.csv")
            st.download_button("Download PDF", data=pdf_bytes, file_name="question_bank.pdf")

            for q in qb.questions:
                st.markdown(f"### {q.id} - {q.marks} marks")
                st.markdown(f"**{q.co_mapping} | {q.bloom_level} | {q.difficulty}**")
                st.write(q.question_text)

                with st.expander("Answer key"):
                    st.write(q.answer_key)

                with st.expander("Rubric"):
                    st.write(q.detailed_rubric)

                citations = ", ".join([f"{s.source} p{s.page}" for s in q.source_citation])
                st.caption("Grounded in: " + citations)
                with st.expander("Citations"):
                    for c in q.source_citation:
                        st.markdown(f"- {c.source} p{c.page}")
                        st.write(c.snippet)
                st.divider()
