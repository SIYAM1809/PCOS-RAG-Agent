"""
Ingests clinical guideline PDFs into Weaviate vector store.

Pipeline:
1. Loads all PDF files from data/guidelines/
2. Splits documents into overlapping chunks with RecursiveCharacterTextSplitter (~500 tokens / 1200 chars)
3. Generates vector embeddings with sentence-transformers (all-MiniLM-L6-v2)
4. Indexes documents and metadata in Weaviate Cloud vector store.
"""

import os
import sys
import glob
from pathlib import Path
from dotenv import load_dotenv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import weaviate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore

# Load environment variables
load_dotenv()

GUIDELINES_DIR = Path(__file__).resolve().parent.parent / "data" / "guidelines"
INDEX_NAME = "PCOS_Guidelines"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def get_weaviate_client():
    """Connects to Weaviate Cloud cluster using credentials from .env."""
    cluster_url = os.getenv("WEAVIATE_URL", "").strip()
    api_key = os.getenv("WEAVIATE_API_KEY", "").strip()

    if not cluster_url or cluster_url == "your_cluster_url":
        raise ValueError("WEAVIATE_URL is not set in .env file.")
    if not api_key or api_key == "your_key_here":
        raise ValueError("WEAVIATE_API_KEY is not set in .env file.")

    if not cluster_url.startswith("http://") and not cluster_url.startswith("https://"):
        cluster_url = f"https://{cluster_url}"

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=cluster_url,
        auth_credentials=weaviate.auth.AuthApiKey(api_key),
    )
    return client


def load_pdf_documents(guidelines_dir: Path):
    """Loads all PDF files from the guidelines directory."""
    pdf_files = list(guidelines_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARN] No PDF files found in {guidelines_dir}")
        return []

    print(f"[INFO] Found {len(pdf_files)} PDF guideline(s):")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    all_docs = []
    for pdf_path in pdf_files:
        print(f"[INFO] Loading: {pdf_path.name} ...")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            # Add clean filename to metadata
            doc.metadata["source_file"] = pdf_path.name
        all_docs.extend(docs)
        print(f"       -> Loaded {len(docs)} pages.")

    print(f"[SUCCESS] Total pages loaded: {len(all_docs)}")
    return all_docs


def chunk_documents(docs, chunk_size=1200, chunk_overlap=200):
    """Splits documents into manageable chunks for vector search."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[SUCCESS] Split into {len(chunks)} chunks (avg ~{chunk_size} chars).")
    return chunks


def ingest():
    """Main ingestion execution pipeline."""
    print("=" * 60)
    print("PCOS Guideline Ingestion Pipeline")
    print("=" * 60)

    # 1. Check directory
    if not GUIDELINES_DIR.exists():
        os.makedirs(GUIDELINES_DIR, exist_ok=True)

    docs = load_pdf_documents(GUIDELINES_DIR)
    if not docs:
        print("\n[STOP] Please download guideline PDFs into data/guidelines/ and run again.")
        print("Recommended PDFs:")
        print("  1. monash_2023_international_guideline.pdf")
        print("  2. rotterdam_2003_consensus.pdf")
        print("  3. endocrine_society_2013_guideline.pdf")
        return

    # 2. Chunking
    chunks = chunk_documents(docs)

    # 3. Embeddings Model
    print(f"\n[INFO] Initializing embedding model ({EMBEDDING_MODEL_NAME})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 4. Weaviate Client
    print("[INFO] Connecting to Weaviate Cloud...")
    client = get_weaviate_client()

    try:
        if client.collections.exists(INDEX_NAME):
            print(f"[INFO] Replacing existing collection '{INDEX_NAME}' with fresh documents...")
            client.collections.delete(INDEX_NAME)

        print(f"[INFO] Ingesting {len(chunks)} chunks into Weaviate collection '{INDEX_NAME}'...")
        vector_store = WeaviateVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client,
            index_name=INDEX_NAME,
        )
        print(f"\n[SUCCESS] Successfully ingested {len(chunks)} chunks into Weaviate '{INDEX_NAME}'!")

        # Sanity search query
        print("\n[INFO] Running sanity similarity search ('Rotterdam diagnostic criteria')...")
        sample_results = vector_store.similarity_search("Rotterdam diagnostic criteria", k=2)
        for i, res in enumerate(sample_results, 1):
            src = res.metadata.get("source_file", "unknown")
            page = res.metadata.get("page", "?")
            snippet = res.page_content[:150].replace("\n", " ")
            print(f"  Result {i} [{src} - Page {page}]: {snippet}...")

    finally:
        client.close()
        print("\n[INFO] Weaviate connection closed.")


if __name__ == "__main__":
    ingest()
