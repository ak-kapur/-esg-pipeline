"""
PDF Ingestion Pipeline
Extracts text from PDF using PyMuPDF, applies privacy masking, splits into chunks.
"""

import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import fitz

from privacy_layer import mask_text
from config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(pdf_path: str) -> List[dict]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "page_number": page_num,
                "text": text,
                "source": os.path.basename(pdf_path),
            })

    doc.close()
    print(f"[Ingestion] Extracted {len(pages)} pages from {os.path.basename(pdf_path)}")
    return pages


def ingest_pdf(pdf_path: str, role: str = "guest") -> List[Document]:
    pages = extract_text_from_pdf(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents:        List[Document] = []
    total_redactions: int            = 0

    for page in pages:
        masked = mask_text(page["text"], role="admin")
        total_redactions += masked.stats["total_redactions"]

        chunks = splitter.split_text(masked.masked_text)

        for idx, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source":             page["source"],
                    "page_number":        page["page_number"],
                    "chunk_index":        idx,
                    "total_redactions":   masked.stats["total_redactions"],
                    "pii_count":          masked.stats["pii_count"],
                    "financial_count":    masked.stats["financial_count"],
                    "is_sensitive":       masked.stats["financial_count"] > 0,
                },
            ))

    print(f"[Ingestion] {len(pages)} pages -> {len(documents)} chunks -> {total_redactions} total redactions")
    return documents


if __name__ == "__main__":
    import sys
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample_reports/test.pdf"
    role     = sys.argv[2] if len(sys.argv) > 2 else "guest"

    docs = ingest_pdf(pdf_path, role=role)

    print(f"\nFirst 3 chunks:\n")
    for doc in docs[:3]:
        print(f"Page {doc.metadata['page_number']} | Chunk {doc.metadata['chunk_index']}")
        print(f"Redactions: {doc.metadata['total_redactions']}")
        print(f"Text preview: {doc.page_content[:200]}")
        print("-" * 50)