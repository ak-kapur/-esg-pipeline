import warnings
warnings.filterwarnings("ignore")

import json
import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from langchain.schema import Document

from pdf_ingestion import ingest_pdf
from privacy_layer import mask_text
from vector_store import ESGVectorStore
from agents import run_esg_pipeline, _retrieve_chunks

st.set_page_config(
    page_title="ESG Insights Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.hero {
    background: linear-gradient(135deg, #0a2e1f 0%, #0d3b2a 50%, #092b1e 100%);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 24px;
    border: 1px solid #1a4d35;
}
.hero h1 {
    color: #4ade80;
    font-size: 28px;
    font-weight: 700;
    margin: 0 0 8px 0;
}
.hero p {
    color: #86efac;
    font-size: 14px;
    margin: 0;
    opacity: 0.85;
}

.kpi-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
.kpi-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.kpi-label {
    font-size: 11px;
    color: #6b7280;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 26px;
    font-weight: 700;
    color: #f9fafb;
}
.kpi-value.green { color: #4ade80; }
.kpi-value.yellow { color: #fbbf24; }
.kpi-value.red { color: #f87171; }

.section-title {
    font-size: 16px;
    font-weight: 700;
    color: #f9fafb;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1f2937;
}

.metric-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.metric-card.verified {
    border-left: 4px solid #22c55e;
}
.metric-card.hallucinated {
    border-left: 4px solid #ef4444;
    background: #1a0f0f;
}
.metric-card.unverifiable {
    border-left: 4px solid #f59e0b;
}
.metric-card.not-found {
    border-left: 4px solid #374151;
    opacity: 0.5;
}
.metric-name {
    font-size: 11px;
    font-weight: 600;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 17px;
    font-weight: 600;
    color: #f9fafb;
    margin-bottom: 8px;
}
.metric-value.not-found-val {
    color: #4b5563;
    font-style: italic;
    font-size: 14px;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.03em;
}
.badge-verified     { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.badge-hallucinated { background: #2d0a0a; color: #f87171; border: 1px solid #991b1b; }
.badge-unverifiable { background: #2d1f02; color: #fbbf24; border: 1px solid #92400e; }
.badge-notfound     { background: #111827; color: #4b5563; border: 1px solid #374151; }

.evidence-text {
    font-size: 11px;
    color: #6b7280;
    margin-top: 6px;
    line-height: 1.5;
    font-style: italic;
}

.redaction-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    margin: 3px;
}
.redaction-pii       { background: #1a0a1a; color: #c084fc; border: 1px solid #7e22ce; }
.redaction-financial { background: #1a1200; color: #fbbf24; border: 1px solid #92400e; }

.query-result-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
}
.query-result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.query-result-title {
    font-size: 12px;
    font-weight: 700;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.query-result-score {
    font-size: 11px;
    color: #4ade80;
    background: #052e16;
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid #166534;
}
.query-result-content {
    font-size: 13px;
    color: #d1d5db;
    line-height: 1.6;
    margin-bottom: 10px;
}
.query-meta-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.query-meta-pill {
    font-size: 11px;
    color: #6b7280;
    background: #1f2937;
    padding: 2px 8px;
    border-radius: 999px;
}
.query-meta-pill.redacted {
    color: #f87171;
    background: #2d0a0a;
}

.masked-text-box {
    background: #0d1117;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 14px;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    color: #9ca3af;
    line-height: 1.6;
    max-height: 200px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}

.divider {
    border: none;
    border-top: 1px solid #1f2937;
    margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ESG Pipeline")
    st.divider()

    uploaded = st.file_uploader("Upload Sustainability PDF", type=["pdf"])
    role     = st.selectbox("Access Role", ["guest", "admin"])
    run_btn  = st.button("Run Pipeline", type="primary", use_container_width=True)

    st.divider()
    st.markdown("""
**Role permissions:**
- `guest` — PII + financials masked
- `admin` — no masking, full access
    """)


# ── Session state ─────────────────────────────────────────────────────────────
for key in ["results", "chunks", "masking", "vs", "role_used"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_emission_value(val: str) -> float | None:
    if not val:
        return None
    cleaned = re.sub(r"[^\d.]", "", val.replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return None


def get_audit_score(results: dict) -> int:
    s = results["summary"]
    total = s["total_metrics"]
    if total == 0:
        return 0
    return round((s["verified"] / total) * 100)


def score_color(score: int) -> str:
    if score >= 80:
        return "green"
    elif score >= 50:
        return "yellow"
    return "red"


def convert_distance_to_similarity(distance: float) -> float:
    return round(max(0.0, 1 - (distance / 2)) * 100, 1)


# ── Pipeline execution ────────────────────────────────────────────────────────
# ── Pipeline execution ────────────────────────────────────────────────────────
if run_btn and uploaded:
    tmp = Path("sample_reports") / uploaded.name
    tmp.write_bytes(uploaded.read())
    st.session_state["uploaded_name"] = uploaded.name.replace(".pdf", "")

    with st.spinner("Ingesting PDF and applying Privacy Layer..."):
        chunks     = ingest_pdf(str(tmp), role="admin")
        full_text  = " ".join(c.page_content for c in chunks)
        masking    = mask_text(full_text, role=role)
        st.session_state.chunks  = chunks
        st.session_state.masking = masking

    with st.spinner("Building vector index..."):
        vs = ESGVectorStore()
        vs.add_documents(chunks)
        st.session_state.vs = vs

    with st.spinner("Running Agent A (Extractor) and Agent B (Auditor)..."):
        relevant_chunks = _retrieve_chunks(vs, role=role, k=8)   # ← uses 15 broad queries now
        results = run_esg_pipeline(relevant_chunks)
        st.session_state.results   = results
        st.session_state.role_used = role

    st.success("Pipeline complete!")

elif run_btn and not uploaded:
    st.warning("Please upload a PDF first.")

# ── Main dashboard ────────────────────────────────────────────────────────────
if st.session_state.results:
    res       = st.session_state.results
    masking   = st.session_state.masking
    chunks    = st.session_state.chunks
    vs        = st.session_state.vs
    role_used = st.session_state.role_used
    extracted = res["extracted_metrics"]
    audit_map = res["audit_map"]
    score     = get_audit_score(res)
    s_color   = score_color(score)

    # Hero
    st.markdown(f"""
<div class="hero">
    <h1>ESG Insights Dashboard</h1>
    <p>Privacy-preserving ESG extraction · Multi-agent hallucination detection · Role-gated vector search</p>
</div>
""", unsafe_allow_html=True)

    # KPI row
    pages    = len(set(c.metadata["page_number"] for c in chunks))
    redacted = masking.stats["total_redactions"] if masking else 0

    st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-label">Pages Processed</div>
        <div class="kpi-value">{pages}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Chunks Indexed</div>
        <div class="kpi-value">{len(chunks)}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Total Redactions</div>
        <div class="kpi-value">{redacted}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Metrics Extracted</div>
        <div class="kpi-value">{res['summary']['total_metrics']}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Audit Score</div>
        <div class="kpi-value {s_color}">{score}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Privacy + Emissions row
    col_privacy, col_emissions = st.columns([1, 1], gap="large")

    with col_privacy:
        st.markdown('<div class="section-title">Privacy Layer — Cleaned Data</div>', unsafe_allow_html=True)

        if masking:
            p1, p2 = st.columns(2)
            p1.metric("PII Redactions",       masking.stats["pii_count"])
            p2.metric("Financial Redactions",  masking.stats["financial_count"])

            if masking.redaction_log:
                from collections import Counter
                type_counts = Counter(e["type"] for e in masking.redaction_log)
                
                badges = ""
                for badge_type, count in sorted(type_counts.items()):
                    cls = "redaction-pii" if any(
                        e["type"] == badge_type and e["category"] == "PII"
                        for e in masking.redaction_log
                    ) else "redaction-financial"
                    badges += f'<span class="redaction-badge {cls}">{badge_type} &times;{count}</span> '
                
                st.markdown(badges, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

            with st.expander("View masked text sample"):
                st.markdown(
                    f'<div class="masked-text-box">{masking.masked_text[:1500]}</div>',
                    unsafe_allow_html=True,
                )
            
            report_name = st.session_state.get("uploaded_name", "ESG_Report")
            redacted_report = f"""ESG REPORT — REDACTED VERSION
            Source: {report_name}
            Generated for: {role_used.upper()} role
            Redactions Applied: {masking.stats['total_redactions']}
            PII Redactions: {masking.stats['pii_count']}
            Financial Redactions: {masking.stats['financial_count']}
            {'='*60}

            {masking.masked_text}
            """
            st.download_button(
                label="⬇️ Download Redacted Report",
                data=redacted_report,
                file_name=f"{report_name}_redacted_{role_used}.txt",
                mime="text/plain",
                use_container_width=True,
            )

            if masking.redaction_log:
                df_log = pd.DataFrame(masking.redaction_log)
                counts = df_log["type"].value_counts().reset_index()
                counts.columns = ["Type", "Count"]
                fig_bar = go.Figure(go.Bar(
                    x=counts["Count"],
                    y=counts["Type"],
                    orientation="h",
                    marker_color="#22c55e",
                    text=counts["Count"],
                    textposition="outside",
                ))
                fig_bar.update_layout(
                    height=250,
                    margin=dict(l=0, r=40, t=10, b=0),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9ca3af", size=12),
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No redactions found — PDF contains clean text with no PII.")

    with col_emissions:
        st.markdown('<div class="section-title">Emissions Breakdown</div>', unsafe_allow_html=True)

        s1 = parse_emission_value(extracted.get("carbon_footprint_scope1"))
        s2 = parse_emission_value(extracted.get("carbon_footprint_scope2"))
        s3 = parse_emission_value(extracted.get("carbon_footprint_scope3"))

        if s1 and s2 and s3:
            fig_pie = go.Figure(go.Pie(
                labels=["Scope 1", "Scope 2", "Scope 3"],
                values=[s1, s2, s3],
                hole=0.55,
                marker_colors=["#22c55e", "#4ade80", "#86efac"],
                textinfo="label+percent",
                textfont=dict(size=13, color="#fff"),
                hovertemplate="<b>%{label}</b><br>%{value:,.0f} tCO₂e<br>%{percent}<extra></extra>",
            ))
            fig_pie.update_layout(
                height=340,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(
                    font=dict(color="#9ca3af", size=12),
                    bgcolor="rgba(0,0,0,0)",
                ),
                annotations=[dict(
                    text=f"<b>Total</b><br>{s1+s2+s3:,.0f}",
                    x=0.5, y=0.5,
                    font=dict(size=13, color="#f9fafb"),
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            missing = []
            if not s1: missing.append("Scope 1")
            if not s2: missing.append("Scope 2")
            if not s3: missing.append("Scope 3")
            st.info(f"Emissions pie chart requires all 3 scopes. Missing: {', '.join(missing)}")

            # Audit donut as fallback
            fig_donut = go.Figure(go.Pie(
                labels=["Verified", "Hallucinated", "Unverifiable"],
                values=[
                    res["summary"]["verified"],
                    res["summary"]["hallucinated"],
                    res["summary"]["unverifiable"],
                ],
                hole=0.6,
                marker_colors=["#22c55e", "#ef4444", "#f59e0b"],
                textinfo="label+percent",
                textfont=dict(size=12, color="#fff"),
            ))
            fig_donut.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                annotations=[dict(
                    text=f"<b>{score}%</b><br>Verified",
                    x=0.5, y=0.5,
                    font=dict(size=14, color="#f9fafb"),
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Agent A + B metrics
    st.markdown('<div class="section-title">Agent A — Extracted Metrics · Agent B — Audit Verdict</div>', unsafe_allow_html=True)

    metric_cols = st.columns(2, gap="medium")
    items       = list(extracted.items())

    for i, (metric, value) in enumerate(items):
        if metric == "error":
            continue

        col         = metric_cols[i % 2]
        info        = audit_map.get(metric, {})
        status      = info.get("status", "UNVERIFIABLE")
        evidence    = info.get("evidence") or ""

        if not value:
            css         = "not-found"
            badge_cls   = "badge-notfound"
            badge_label = "NOT FOUND"
            val_cls     = "not-found-val"
            val_display = "Not reported"
        else:
            css         = status.lower()
            badge_map   = {
                "VERIFIED":     ("badge-verified",     "✓ VERIFIED"),
                "HALLUCINATED": ("badge-hallucinated", "✗ HALLUCINATED"),
                "UNVERIFIABLE": ("badge-unverifiable", "? UNVERIFIABLE"),
            }
            badge_cls, badge_label = badge_map.get(status, ("badge-unverifiable", status))
            val_cls     = ""
            val_display = value

        evidence_html = ""
        if evidence and value:
            evidence_html = f'<div class="evidence-text">"{evidence[:120]}..."</div>'

        col.markdown(f"""
<div class="metric-card {css}">
    <div class="metric-name">{metric.replace("_", " ").upper()}</div>
    <div class="metric-value {val_cls}">{val_display}</div>
    <span class="badge {badge_cls}">{badge_label}</span>
    {evidence_html}
</div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Custom query
    st.markdown('<div class="section-title">Ask Anything — Role-Gated Vector Search</div>', unsafe_allow_html=True)

   
    display_role = role_used if role_used else role
    role_label   = "financial metadata visible" if display_role == "admin" else "financial metadata redacted"
    badge_cls    = "badge-verified" if display_role == "admin" else "badge-hallucinated"

    st.markdown(f'<span class="badge {badge_cls}">{display_role.upper()}</span> &nbsp; <span style="font-size:13px;color:#6b7280">{role_label}</span>', unsafe_allow_html=True)

    query_input = st.text_input(
        "Ask a question about the report:",
        placeholder="e.g. What are the Scope 3 emission sources?",
        label_visibility="collapsed",
    )

    if query_input and vs:
        query_results = vs.query(query_input, role=role_used, k=3)

        # Mask chunks for guest role before sending to LLM
        if role_used == "guest":
            combined = "\n\n".join([
                mask_text(r["content"], role=role_used).masked_text
                for r in query_results
            ])
        else:
            combined = "\n\n".join([r["content"] for r in query_results])

        source_pages = list(set(str(r["metadata"].get("page_number", "?")) for r in query_results))
        is_redacted  = role_used == "guest"  # ← simpler: guest always has redacted view

        with st.spinner("Generating answer..."):
            from langchain.prompts import ChatPromptTemplate
            from agents import llm

            QA_PROMPT = ChatPromptTemplate.from_messages([
                ("system", """You are an ESG analyst assistant.
Answer the question directly and concisely using only the provided context.
If the answer is not in the context, say 'This information was not found in the report.'
Keep the answer under 4 sentences."""),
                ("human", "Question: {question}\n\nContext:\n{context}"),
            ])

            try:
                response = llm.invoke(
                    QA_PROMPT.format_messages(
                        question=query_input,
                        context=combined,
                    )
                )
                answer = response.content.strip()
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e):
                    answer = "Rate limit reached. Please wait and try again."
                else:
                    answer = f"Error: {str(e)}"

        redacted_pill = (
    '<span class="query-meta-pill redacted">Financial data redacted</span>'
    if is_redacted else ""
)
        pages_str = ", ".join(source_pages)
        role_str  = display_role.upper()
        html_answer = (
    '<div class="query-result-card">'
    '<div class="query-result-header">'
    '<span class="query-result-title">Answer</span>'
    f'<span class="query-result-score">Based on {len(query_results)} sources</span>'
    '</div>'
    f'<div style="font-size:14px;color:#d1d5db;line-height:1.8;margin-bottom:12px">{answer}</div>'
    '<div class="query-meta-row">'
    f'<span class="query-meta-pill">Pages referenced: {pages_str}</span>'
    f'<span class="query-meta-pill">{role_str} role</span>'
    + redacted_pill +
    '</div>'
    '</div>'
)

        st.markdown(html_answer, unsafe_allow_html=True)
        

else:
    st.markdown("""
<div class="hero">
    <h1>ESG Insights Dashboard</h1>
    <p>Privacy-preserving ESG extraction · Multi-agent hallucination detection · Role-gated vector search</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
### How it works
1. **Upload** any public Sustainability / ESG PDF report
2. **Privacy Layer** masks PII and financial figures using Microsoft Presidio before the LLM sees any text
3. **Agent A** extracts Environmental metrics — Carbon footprint, Water, Energy, Waste, Targets
4. **Agent B** cross-references every metric against the source text and flags hallucinations
5. **Emissions pie chart** auto-renders when Scope 1, 2 and 3 are all found
6. **Role-gated search** — Guest cannot access financial metadata, Admin sees everything
    """)