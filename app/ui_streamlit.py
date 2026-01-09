from pathlib import Path
import sys
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Outcome-QBank", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600&display=swap');
:root {
  --bg: #0a0b0e;
  --panel: #12161d;
  --panel-2: #0f1319;
  --ink: #e7edf6;
  --ink-2: #9aa7ba;
  --border: #1f2836;
  --accent: #7ea6d9;
}
html, body, [class*="css"] {
  font-family: "Manrope", Arial, sans-serif;
  color: var(--ink);
}
.stApp {
  background: var(--bg);
}
.block-container {
  max-width: 1100px;
  padding-top: 1.4rem;
}
.hero {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.2rem 1.4rem;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
}
.hero h1 {
  font-size: 1.9rem;
  margin: 0;
}
.hero p {
  margin: 0.35rem 0 0;
  color: var(--ink-2);
}
.card {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.9rem 1rem;
}
.label {
  font-weight: 600;
  margin-bottom: 0.4rem;
}
.hint {
  color: var(--ink-2);
  font-size: 0.92rem;
}
.stButton>button, .stDownloadButton>button {
  background: #1a2230;
  color: #e7edf6;
  border: 1px solid #2a3a52;
  border-radius: 10px;
  padding: 0.55rem 1rem;
}
.stButton>button:hover, .stDownloadButton>button:hover {
  background: #212d3f;
}
section[data-testid="stSidebar"] {
  background: #0b0f15;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * {
  color: var(--ink);
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>Outcome-QBank</h1>
  <p>Upload your outcomes and materials. Get clean, grounded questions.</p>
</div>
""",
    unsafe_allow_html=True,
)

# -------- State --------
st.session_state.setdefault("ingested", False)
st.session_state.setdefault("syllabus_snippets", [])
st.session_state.setdefault("planned", None)
st.session_state.setdefault("ctx", [])
st.session_state.setdefault("subject_profile", None)
st.session_state.setdefault("mix", {"theory": 35, "numerical": 30, "derivation": 15, "equation": 10, "diagram": 10})
st.session_state.setdefault("logs", [])
st.session_state.setdefault("last_audit", None)
st.session_state.setdefault("last_qb", None)
st.session_state.setdefault("coverage_report", None)
st.session_state.setdefault("last_upload_sig", None)

# -------- Minimal settings --------
collection_name = "course_material"
chunk_size = 1000
overlap = 200
top_k = 8
min_importance = 1
max_total_ctx = 120
include_sample_papers = True
bloom_focus = "Mixed"

st.subheader("Upload")
st.markdown("<div class='hint'>Course outcomes are required. Materials and sample papers improve quality.</div>", unsafe_allow_html=True)
pii_consent = st.checkbox("I confirm I have consent to process personal data (if present).", value=False)

col_a, col_b, col_c = st.columns(3, gap="large")
with col_a:
    st.markdown("<div class='card'><div class='label'>Course outcomes</div></div>", unsafe_allow_html=True)
    outcomes_files = st.file_uploader(
        "Outcomes PDF",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
with col_b:
    st.markdown("<div class='card'><div class='label'>Course materials</div></div>", unsafe_allow_html=True)
    material_files = st.file_uploader(
        "Materials PDF",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
with col_c:
    st.markdown("<div class='card'><div class='label'>Sample papers</div></div>", unsafe_allow_html=True)
    sample_files = st.file_uploader(
        "Samples PDF",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.caption("Used for style only. Never copied.")

st.subheader("Scope")
course_name = st.text_input("Course name", value="", placeholder="e.g., Signals and Systems")
topic = st.text_input("Topic", value="", placeholder="e.g., Fourier series, Z-transform, sampling")

st.subheader("Output")
num_q = st.slider("Total questions", 5, 100, 20)
marks_each = st.selectbox("Marks per question", [2, 5, 10], index=0)
difficulty_mix = st.selectbox("Difficulty mix", ["Mostly Medium", "Easy/Medium", "Medium/Hard"], index=0)

generate_clicked = st.button("Generate Questions")

def _upload_signature(files: list) -> tuple:
    sig = []
    for f in files or []:
        sig.append((f.name, getattr(f, "size", None)))
    return tuple(sig)

if generate_clicked:
    if not outcomes_files and not material_files:
        st.error("Please upload course outcomes and/or course materials.")
    else:
        upload_sig = (
            _upload_signature(outcomes_files)
            + _upload_signature(material_files)
            + _upload_signature(sample_files)
        )

        with st.spinner("Preparing content..."):
            client = get_client("data/vector_db")
            embedder = get_embedder()
            collection = get_collection(client, collection_name, embedder)

            if (not st.session_state.ingested) or (st.session_state.last_upload_sig != upload_sig):
                pii_report = []
                total_chunks = 0

                for of in outcomes_files or []:
                    pdf_path = save_uploaded_pdf(of, save_dir="data/uploads")
                    pages = extract_pages_from_pdf(pdf_path)
                    pii_hits = scan_pages_for_pii(pages)
                    if pii_hits:
                        pii_report.append({"file": of.name, "findings": pii_hits})

                    chunks = chunk_pages(
                        pages,
                        source=of.name,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        source_type="outcomes",
                    )
                    total_chunks += len(chunks)
                    upsert_chunks(collection, chunks)

                for mf in material_files or []:
                    pdf_path = save_uploaded_pdf(mf, save_dir="data/uploads")
                    pages = extract_pages_from_pdf(pdf_path)
                    pii_hits = scan_pages_for_pii(pages)
                    if pii_hits:
                        pii_report.append({"file": mf.name, "findings": pii_hits})

                    chunks = chunk_pages(
                        pages,
                        source=mf.name,
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

                    chunks = chunk_pages(
                        pages,
                        source=sf.name,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        source_type="sample_paper",
                    )
                    total_chunks += len(chunks)
                    upsert_chunks(collection, chunks)

                if pii_report and not pii_consent:
                    st.error("PII detected in uploads. Please confirm consent to proceed.")
                    st.stop()
                if pii_report and pii_consent:
                    st.warning("PII detected in uploads. Proceeding with consent.")

                st.session_state.ingested = True
                st.session_state.last_upload_sig = upload_sig
                st.success(f"Ingested OK. chunks={total_chunks}")

            # Load syllabus context automatically
            auto_queries = ["Course outcomes", "Syllabus", "Module", "Unit"]
            syllabus_ctx = []
            for q in auto_queries:
                syllabus_ctx += retrieve_top_k_strict(
                    collection,
                    q,
                    k=3,
                    allowed_source_types=["material", "outcomes"],
                )

            # de-dup
            seen = set()
            cleaned = []
            for c in syllabus_ctx:
                key = (c.get("source"), c.get("page"), (c.get("text") or "")[:120])
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(c)
            st.session_state.syllabus_snippets = cleaned

            agent = GeneratorAgent(model="gpt-4o-mini")
            profile = agent.classify_subject(st.session_state.syllabus_snippets)
            st.session_state.subject_profile = profile.model_dump()
            if profile.recommended_mix:
                st.session_state.mix = profile.recommended_mix.model_dump()

            topics_list = [t.strip() for t in (topic or "").split(",") if t.strip()]
            topic_for_plan = " | ".join(topics_list) if topics_list else (course_name or "General")
            plan_obj = agent.plan(topic_for_plan, st.session_state.syllabus_snippets)
            st.session_state.planned = plan_obj.model_dump()

            planned_subs = st.session_state.planned.get("subtopics", []) or []
            planned_subs = [s for s in planned_subs if int(s.get("importance", 3)) >= int(min_importance)]

            all_ctx = []
            for sub in planned_subs:
                q = sub.get("query") or sub.get("name")
                allowed_types = ["material", "outcomes"]
                if include_sample_papers:
                    allowed_types.append("sample_paper")
                hits = retrieve_top_k_strict(
                    collection,
                    q,
                    k=top_k,
                    allowed_source_types=allowed_types,
                )
                for h in hits:
                    h["subtopic"] = sub.get("name")
                    h["importance"] = sub.get("importance", 3)
                all_ctx += hits

            all_ctx.sort(
                key=lambda x: (
                    -int(x.get("importance", 3)),
                    -int(x.get("kw_score", 0)),
                    float(x.get("distance", 9e9)),
                )
            )
            # Expand coverage with a generic retrieval pass to use more of the PDFs.
            if len(all_ctx) < max_total_ctx:
                generic = retrieve_top_k_strict(
                    collection,
                    topic_for_plan or "course content",
                    k=max_total_ctx,
                    allowed_source_types=["material", "outcomes"],
                )
                all_ctx += generic

            # de-dup and cap
            seen = set()
            deduped = []
            for c in all_ctx:
                key = (c.get("source"), c.get("page"), (c.get("text") or "")[:120])
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(c)
            all_ctx = deduped[:max_total_ctx]
            if not all_ctx:
                # Fallback: use top chunks from all material/outcome text.
                fallback = retrieve_top_k_strict(
                    collection,
                    topic or course_name or "Course content",
                    k=max_total_ctx,
                    allowed_source_types=["material", "outcomes"],
                )
                all_ctx = fallback[:max_total_ctx]
            st.session_state.ctx = all_ctx

        if not st.session_state.ctx:
            st.warning("No usable context found. Upload more materials.")
        else:
            difficulty_distribution_map = {
                "Mostly Medium": {"Easy": 20, "Medium": 60, "Hard": 20},
                "Easy/Medium": {"Easy": 40, "Medium": 40, "Hard": 20},
                "Medium/Hard": {"Easy": 20, "Medium": 40, "Hard": 40},
            }
            targets = {
                "topic": topic_for_plan or "General",
                "num_questions": num_q,
                "marks_each": marks_each,
                "bloom_focus": bloom_focus,
                "difficulty_mix": difficulty_mix,
                "difficulty_distribution": difficulty_distribution_map.get(difficulty_mix),
                "mark_distribution": None,
                "question_type_preferences": {"include_numerical": True, "include_diagram": False},
                "instruction": (
                    "Generate exam-relevant questions strictly from context. "
                    "Try to reach the requested count if context supports it. "
                    "If context is shallow, output fewer questions instead of inventing."
                ),
            }

            mix_sum = sum(st.session_state.mix.values()) or 1
            norm_mix = {k: int(round(v * 100 / mix_sum)) for k, v in st.session_state.mix.items()}

            with st.spinner("Generating questions..."):
                qb, audit, logs = run_generation_loop(
                    course_name=course_name or "Course",
                    targets=targets,
                    context_snippets=st.session_state.ctx,
                    subject_profile=st.session_state.get("subject_profile"),
                    question_mix=norm_mix,
                    max_iters=4,
                    model="gpt-4o-mini",
                )

            if not qb:
                st.error("Generation failed. Try again with more context.")
            else:
                st.session_state.last_qb = qb.model_dump()
                st.session_state.last_audit = audit.model_dump() if audit else None
                st.session_state.logs = logs

                if audit and not audit.passed:
                    st.warning("Review flagged issues below.")
                else:
                    st.success("Questions ready.")

                coverage = compute_coverage_report(st.session_state.last_qb.get("questions", []))
                st.session_state.coverage_report = coverage

                with st.expander("Coverage report"):
                    st.json(coverage)

                with st.expander("Improvement log"):
                    st.json(st.session_state.logs)

                csv_bytes = questions_to_csv_bytes(st.session_state.last_qb.get("questions", []))
                pdf_bytes = questions_to_pdf_bytes(st.session_state.last_qb.get("questions", []), coverage)
                st.download_button("Download CSV", data=csv_bytes, file_name="question_bank.csv")
                st.download_button("Download PDF", data=pdf_bytes, file_name="question_bank.pdf")

                st.subheader("Questions")
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
