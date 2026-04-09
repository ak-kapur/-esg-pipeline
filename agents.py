"""
Multi-Agent ESG Reasoning using Groq (Llama 3.3)
Agent A: Extracts Environmental metrics from masked chunks
Agent B: Audits Agent A output for hallucinations against source chunks
"""

import json
import re
from typing import List, Dict, Any

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from config import GROQ_API_KEY, GROQ_MODEL


# ─────────────────────────────────────────────
#  LLM
# ─────────────────────────────────────────────
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL,
    temperature=0.1,
)


# ─────────────────────────────────────────────
#  AGENT A — EXTRACTOR PROMPT
# ─────────────────────────────────────────────
EXTRACTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an ESG data extraction specialist.
Extract ONLY Environmental metrics from the text chunks provided.
Return a valid JSON object with exactly these keys (use null if not found):
{{
    "carbon_footprint_scope1":    "<value and unit or null>",
    "carbon_footprint_scope2":    "<value and unit or null>",
    "carbon_footprint_scope3":    "<value and unit or null>",
    "water_consumption":          "<value and unit or null>",
    "renewable_energy_pct":       "<percentage or null>",
    "waste_generated":            "<value and unit or null>",
    "waste_recycled_pct":         "<percentage or null>",
    "energy_intensity":           "<value and unit or null>",
    "biodiversity_initiatives":   "<brief description or null>",
    "emission_reduction_target":  "<target and year or null>"
}}

Extraction Rules:
- Do NOT invent or estimate any numbers
- Only include values explicitly stated in the text
- For carbon_footprint_scope1: look for 'Scope 1', 'direct emissions',
  'stationary combustion', 'mobile combustion', 'refrigerants' values
- For carbon_footprint_scope2: look for 'Scope 2', 'market-based',
  'location-based', 'purchased electricity', 'Total Market-Based Scope 2' values
- For carbon_footprint_scope3: look for 'Scope 3', 'business travel',
  'downstream leased assets', 'value chain emissions', 'Category 6' values
- Units may appear as mtCO2e, tCO2e, MTCO2e, ktCO2e — treat all as valid
- For renewable_energy_pct: look for 'renewable', 'carbon neutral', '% renewable',
  'renewable electricity', 'clean energy percentage'
- For emission_reduction_target: look for 'net-zero', 'net zero', 'carbon neutral',
  'reduction target', '2030', '2050', 'baseline year'
- Return ONLY the JSON object, no explanation"""),
    ("human", "Text chunks:\n\n{chunks}"),
])


# ─────────────────────────────────────────────
#  AGENT B — AUDITOR PROMPT
# ─────────────────────────────────────────────
AUDITOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an ESG audit specialist checking for hallucinations.
You will receive extracted metrics from Agent A and the original source text chunks.

For each metric that is NOT null, check if it is supported by the source text.
Return a valid JSON array with one object per non-null metric:
[
    {{
        "metric":        "<metric name>",
        "extracted_val": "<what Agent A extracted>",
        "status":        "VERIFIED or HALLUCINATED or UNVERIFIABLE",
        "evidence":      "<direct quote from source text supporting it, or null>",
        "confidence":    <float between 0.0 and 1.0>
    }}
]

Audit Rules:
- VERIFIED:      value is explicitly stated in source text (even with slight formatting diff)
- HALLUCINATED:  value contradicts source text OR was clearly fabricated
- UNVERIFIABLE:  source text is ambiguous, incomplete, or mentions topic without exact value
- Only audit non-null metrics — skip null values entirely
- Be flexible with units: '14,201 mtCO2e' and '14201 mtCO2e' are the same value
- Return ONLY the JSON array, no explanation"""),
    ("human", "Extracted Metrics:\n{extracted}\n\nSource Chunks:\n{chunks}"),
])


# ─────────────────────────────────────────────
#  VECTOR SEARCH QUERIES
#  Broad vocabulary to catch different report styles
# ─────────────────────────────────────────────
ESG_RETRIEVAL_QUERIES = [
    # Emissions — broad
    "carbon emissions scope 1 scope 2 scope 3",
    "operational emissions GHG greenhouse gas mtCO2e",
    "market based location based emissions total",
    "financed emissions absolute tCO2e sector",
    "GHG emissions data base year direct indirect",
    "stationary combustion mobile combustion refrigerants",

    # Water
    "water consumption usage withdrawal cubic meters",
    "water intensity freshwater recycled wastewater",

    # Renewable energy
    "renewable energy percentage carbon neutral electricity",
    "clean energy solar wind power procurement",

    # Waste
    "waste generated recycled landfill hazardous",
    "waste diversion rate circular economy",

    # Targets & initiatives
    "emission reduction target net zero 2030 2050",
    "biodiversity nature conservation ecosystem initiative",
    "energy intensity per unit revenue employee",
]


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def _format_chunks(chunks: List[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[Page {d.metadata.get('page_number', '?')}]\n{d.page_content}"
        for d in chunks
    )


def _extract_json_object(text: str) -> Any:
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {"error": "No JSON object found", "raw": text}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return {"error": str(e), "raw": text}


def _extract_json_array(text: str) -> Any:
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return [{"error": "No JSON array found", "raw": text}]
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return [{"error": str(e), "raw": text}]


def _retrieve_chunks(vs, role: str = "guest", k: int = 8) -> List[Document]:
    """Run all ESG retrieval queries and deduplicate results."""
    seen: set = set()
    relevant_chunks: List[Document] = []

    for query in ESG_RETRIEVAL_QUERIES:
        results = vs.query(query, role=role, k=k)
        for r in results:
            key = r["content"][:120]
            if key not in seen:
                seen.add(key)
                relevant_chunks.append(
                    Document(
                        page_content=r["content"],
                        metadata=r["metadata"],
                    )
                )

    print(f"[Retrieval] Total unique chunks retrieved: {len(relevant_chunks)}")
    return relevant_chunks


# ─────────────────────────────────────────────
#  AGENT A — EXTRACTOR
# ─────────────────────────────────────────────
def run_extractor(chunks: List[Document]) -> Dict[str, Any]:
    print("[Agent A] Extracting ESG metrics...")
    formatted = _format_chunks(chunks)
    response = llm.invoke(
        EXTRACTOR_PROMPT.format_messages(chunks=formatted)
    )
    result = _extract_json_object(response.content.strip())

    non_null = [k for k, v in result.items() if v and k != "error"]
    print(f"[Agent A] Extracted {len(non_null)} non-null metrics: {non_null}")
    return result


# ─────────────────────────────────────────────
#  AGENT B — AUDITOR
# ─────────────────────────────────────────────
def run_auditor(extracted: Dict[str, Any], chunks: List[Document]) -> List[Dict[str, Any]]:
    print("[Agent B] Auditing extracted metrics...")

    # Only pass non-null metrics to auditor
    non_null_metrics = {k: v for k, v in extracted.items() if v and k != "error"}
    if not non_null_metrics:
        print("[Agent B] Nothing to audit — all metrics are null.")
        return []

    formatted = _format_chunks(chunks)
    response = llm.invoke(
        AUDITOR_PROMPT.format_messages(
            extracted=json.dumps(non_null_metrics, indent=2),
            chunks=formatted,
        )
    )
    result = _extract_json_array(response.content.strip())

    verified     = sum(1 for r in result if r.get("status") == "VERIFIED")
    hallucinated = sum(1 for r in result if r.get("status") == "HALLUCINATED")
    unverifiable = len(result) - verified - hallucinated
    print(f"[Agent B] Verified: {verified} | Hallucinated: {hallucinated} | Unverifiable: {unverifiable}")
    return result


# ─────────────────────────────────────────────
#  FULL PIPELINE
# ─────────────────────────────────────────────
def run_esg_pipeline(chunks: List[Document]) -> Dict[str, Any]:
    extracted = run_extractor(chunks)
    audit     = run_auditor(extracted, chunks)

    audit_map = {
        item["metric"]: item
        for item in audit
        if isinstance(item, dict) and "metric" in item
    }

    total   = len([v for v in extracted.values() if v and "error" not in extracted])
    verified     = sum(1 for a in audit if a.get("status") == "VERIFIED")
    hallucinated = sum(1 for a in audit if a.get("status") == "HALLUCINATED")
    unverifiable = sum(1 for a in audit if a.get("status") == "UNVERIFIABLE")

    return {
        "extracted_metrics": extracted,
        "audit_results":     audit,
        "audit_map":         audit_map,
        "summary": {
            "total_metrics": total,
            "verified":      verified,
            "hallucinated":  hallucinated,
            "unverifiable":  unverifiable,
        },
    }


# ─────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from pdf_ingestion import ingest_pdf
    from vector_store import ESGVectorStore

    import sys
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample_reports/your_report.pdf"
    role     = sys.argv[2] if len(sys.argv) > 2 else "guest"

    print(f"\n[Test] Running pipeline on: {pdf_path} | Role: {role}")

    docs = ingest_pdf(pdf_path, role=role)
    vs   = ESGVectorStore()
    vs.add_documents(docs)

    relevant_chunks = _retrieve_chunks(vs, role=role, k=8)
    print(f"[Test] Chunks fed to agents: {len(relevant_chunks)}")

    results = run_esg_pipeline(relevant_chunks)

    print("\n── Extracted Metrics ──")
    print(json.dumps(results["extracted_metrics"], indent=2))

    print("\n── Audit Results ──")
    for item in results["audit_results"]:
        print(f"  {item.get('metric'):35s} → {item.get('status')} (conf: {item.get('confidence', '?')})")

    print("\n── Summary ──")
    print(json.dumps(results["summary"], indent=2))