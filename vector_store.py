"""
Encrypted Vector Storage with Role-Based Access Control.
Sensitive metadata fields are Fernet-encrypted before FAISS indexing.
Guest and Analyst roles cannot access financial metadata.
"""

import os
import json
from typing import List, Dict, Any, Optional

from cryptography.fernet import Fernet
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import ROLES, SENSITIVE_META_KEYS, FAISS_INDEX_PATH, EMBED_MODEL, KEY_FILE


def _load_or_create_key() -> bytes:
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            return f.read()
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    print(f"[VectorStore] New encryption key created at {KEY_FILE}")
    return key


def _cipher() -> Fernet:
    return Fernet(_load_or_create_key())


def encrypt_metadata(data: dict) -> str:
    return _cipher().encrypt(json.dumps(data).encode()).decode()


def decrypt_metadata(ciphertext: str) -> dict:
    return json.loads(_cipher().decrypt(ciphertext.encode()).decode())


def _apply_role_filter(metadata: dict, role: str) -> dict:
    perms = ROLES.get(role, ROLES["guest"])
    result = {}
    for k, v in metadata.items():
        if k == "_enc":
            continue
        result[k] = v

    enc = metadata.get("_enc")
    if enc:
        if perms["see_financial"]:
            decrypted = decrypt_metadata(enc)
            result.update(decrypted)
        else:
            result["sensitive_data"] = "REDACTED - insufficient privileges"

    return result


class ESGVectorStore:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.store: Optional[FAISS] = None

    def add_documents(self, documents: List[Document]) -> None:
        prepared = []
        for doc in documents:
            meta = dict(doc.metadata)
            sensitive = {k: meta.pop(k) for k in list(meta) if k in SENSITIVE_META_KEYS}
            if sensitive:
                meta["_enc"] = encrypt_metadata(sensitive)
            prepared.append(Document(page_content=doc.page_content, metadata=meta))

        if self.store is None:
            self.store = FAISS.from_documents(prepared, self.embeddings)
        else:
            self.store.add_documents(prepared)

        print(f"[VectorStore] Indexed {len(prepared)} documents.")

    def query(self, query: str, role: str = "guest", k: int = 5) -> List[Dict[str, Any]]:
        if role not in ROLES:
            raise PermissionError(f"Unknown role: {role}")
        if not ROLES[role]["can_query"]:
            raise PermissionError(f"Role '{role}' does not have query access.")
        if self.store is None:
            return []

        results = self.store.similarity_search_with_score(query, k=k)
        output  = []

        for doc, score in results:
            filtered = _apply_role_filter(dict(doc.metadata), role)
            output.append({
                "content":  doc.page_content,
                "metadata": filtered,
                "score":    round(float(score), 4),
            })

        return output

    def save(self) -> None:
        if self.store:
            self.store.save_local(FAISS_INDEX_PATH)
            print(f"[VectorStore] Saved index to {FAISS_INDEX_PATH}")

    def load(self) -> None:
        if os.path.exists(FAISS_INDEX_PATH):
            self.store = FAISS.load_local(
                FAISS_INDEX_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"[VectorStore] Loaded index from {FAISS_INDEX_PATH}")
        else:
            print(f"[VectorStore] No saved index found at {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    from pdf_ingestion import ingest_pdf

    print("Building vector store from Apple ESG report...")
    docs = ingest_pdf("sample_reports/your_report.pdf", role="guest")

    vs = ESGVectorStore()
    vs.add_documents(docs)
    vs.save()

    print("\n--- GUEST query ---")
    results = vs.query("carbon emissions and climate targets", role="guest", k=2)
    for r in results:
        print(f"Score: {r['score']}")
        print(f"Content: {r['content'][:200]}")
        print(f"Metadata: {r['metadata']}")
        print("-" * 40)

    print("\n--- ADMIN query ---")
    results = vs.query("carbon emissions and climate targets", role="admin", k=2)
    for r in results:
        print(f"Score: {r['score']}")
        print(f"Content: {r['content'][:200]}")
        print(f"Metadata: {r['metadata']}")
        print("-" * 40)