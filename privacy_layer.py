"""
Privacy Layer - Presidio-based PII and financial data masking
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerResult
from presidio_anonymizer import AnonymizerEngine

from config import PRESIDIO_SCORE_THRESHOLD


aadhaar_recognizer = PatternRecognizer(
    supported_entity="AADHAAR_NUMBER",
    patterns=[
        Pattern(
            name="aadhaar_spaced",
            regex=r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b",
            score=0.85
        ),
    
    ],
    context=["aadhaar", "aadhar", "uidai", "uid", "unique identification", "enrollment no"],
)

indian_phone_recognizer = PatternRecognizer(
    supported_entity="IN_PHONE_NUMBER",
    patterns=[
        Pattern(
            name="indian_mobile_intl",
            regex=r"\+91[\s-]?[6-9]\d{4}[\s-]?\d{5}",
            score=0.85
        ),
        Pattern(
            name="indian_mobile_plain",
            regex=r"\b[6-9]\d{9}\b",
            score=0.75
        ),
        # ← ADD THIS — catches landlines like +91 80 2852 0261
        Pattern(
            name="indian_landline",
            regex=r"\+91[\s-]?\d{2,4}[\s-]?\d{4}[\s-]?\d{4}",
            score=0.85
        ),
        # ← ADD THIS — catches 080-28520261 or 080 2852 0261
        Pattern(
            name="indian_landline_local",
            regex=r"\b0\d{2,4}[\s-]\d{4}[\s-]?\d{4}\b",
            score=0.80
        ),
    ],
)

FINANCIAL_PATTERNS = {
    "INVESTMENT_AMOUNT": (
        r"(?:USD|EUR|INR|\$|£|€)\s?[\d,]+(?:\.\d{1,2})?(?:\s?(?:million|billion|trillion|crore|M|B|Bn|Tn|Mn))?"
        r"|[\d,]+(?:\.\d{1,2})?\s?(?:million|billion|trillion|crore)\s?(?:USD|EUR|INR|dollars)?"
    ),
    "PERCENTAGE_TARGET": (
        r"\b\d{1,3}(?:\.\d{1,2})?%\s+(?:return|yield|IRR|ROI|CAGR|reduction|target|decrease|increase|lower|higher)"
    ),
    "INTERNAL_COST": (
        r"(?:cost|budget|capex|opex|CapEx|OpEx|revenue|income|earnings)\s*(?:of|:)?\s*(?:USD|INR|\$)?\s*[\d,]+(?:\.\d{1,2})?"
    ),
    "IRR_ROI": (
        r"\b\d{1,3}(?:\.\d{1,2})?%\s*(?:IRR|ROI|CAGR|return)"
    ),
    "PAN_NUMBER": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    # NEW — catches $620Bn, $1Tn, $85Bn style
    "DOLLAR_FIGURE": (
        r"\$[\d,]+(?:\.\d{1,2})?(?:\s?(?:Bn|Mn|Tn|billion|million|trillion))?"
    ),
}
PRESIDIO_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "AADHAAR_NUMBER",
    "IN_PHONE_NUMBER",

]
pan_recognizer = PatternRecognizer(
    supported_entity="IN_PAN",
    patterns=[
        Pattern(
            name="pan",
            regex=r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
            score=0.9
        ),
    ],
)

def _build_analyzer() -> AnalyzerEngine:
    engine = AnalyzerEngine()
    engine.registry.add_recognizer(aadhaar_recognizer)
    engine.registry.add_recognizer(pan_recognizer)
    engine.registry.add_recognizer(indian_phone_recognizer)
    return engine


_analyzer   = _build_analyzer()
_anonymizer = AnonymizerEngine()


@dataclass
class MaskingResult:
    masked_text:   str
    original_text: str
    redaction_log: List[Dict] = field(default_factory=list)
    stats:         Dict       = field(default_factory=dict)


def _make_token(label: str, value: str) -> str:
    h = hashlib.sha256(value.encode()).hexdigest()[:8].upper()
    return f"[{label}_{h}]"


def mask_text(text: str, role: str = "guest") -> MaskingResult:
    if role == "admin":
        return MaskingResult(
            masked_text=text,
            original_text=text,
            redaction_log=[],
            stats={"total_redactions": 0, "pii_count": 0, "financial_count": 0},
        )

    log:    List[Dict] = []
    masked: str        = text

    results: List[RecognizerResult] = _analyzer.analyze(
        text=masked,
        entities=PRESIDIO_ENTITIES,
        language="en",
        score_threshold=PRESIDIO_SCORE_THRESHOLD,
    )

    results = sorted(results, key=lambda r: r.start, reverse=True)

    for result in results:
        original = masked[result.start: result.end]
        token    = _make_token(result.entity_type, original)
        log.append({
            "type":       result.entity_type,
            "category":   "PII",
            "original":   original,
            "token":      token,
            "confidence": round(result.score, 2),
        })
        masked = masked[: result.start] + token + masked[result.end:]

    financial_count = 0
    if role == "guest":
        for label, pattern in FINANCIAL_PATTERNS.items():
            for match in re.finditer(pattern, masked, re.IGNORECASE):
                original = match.group(0)
                token    = _make_token(label, original)
                log.append({
                    "type":       label,
                    "category":   "SENSITIVE_FINANCIAL",
                    "original":   original,
                    "token":      token,
                    "confidence": 1.0,
                })
                masked = masked.replace(original, token, 1)
                financial_count += 1

    pii_count = sum(1 for e in log if e["category"] == "PII")

    return MaskingResult(
        masked_text=masked,
        original_text=text,
        redaction_log=log,
        stats={
            "total_redactions": len(log),
            "pii_count":        pii_count,
            "financial_count":  financial_count,
        },
    )


if __name__ == "__main__":
    sample = (
        "Contact Dr. Rajesh Kumar at rajesh.kumar@greenco.com or +91 98765 43210. "
        "Aadhaar: 2345 6789 0123. PAN: ABCDE1234F. "
        "Our ESG fund targets a 12.5% IRR with investment of $50 million. "
        "Capex of $2.3 billion allocated for Scope 1 emission reductions. "
        "Server IP: 192.168.1.100"
    )

    for role in ["guest", "analyst", "admin"]:
        print(f"\n{'='*50}\n{role.upper()} ROLE\n{'='*50}")
        result = mask_text(sample, role=role)
        print("Masked:\n", result.masked_text)
        print("Stats:", result.stats)
        for entry in result.redaction_log:
            print(f"  [{entry['category']}] {entry['type']} -> {entry['token']} (conf: {entry['confidence']})")