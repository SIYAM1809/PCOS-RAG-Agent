"""
Standalone Guideline Agent (Week 1).

Retrieves clinical evidence from Weaviate vector store (PCOS_Guidelines)
and generates answers strictly grounded in the loaded guideline consensus documents.
"""

import os
import sys
from dotenv import load_dotenv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import weaviate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

INDEX_NAME = "PCOS_Guidelines"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"


def get_weaviate_client():
    """Connects to Weaviate Cloud cluster."""
    cluster_url = os.getenv("WEAVIATE_URL", "").strip()
    api_key = os.getenv("WEAVIATE_API_KEY", "").strip()

    if not cluster_url.startswith("http://") and not cluster_url.startswith("https://"):
        cluster_url = f"https://{cluster_url}"

    return weaviate.connect_to_weaviate_cloud(
        cluster_url=cluster_url,
        auth_credentials=weaviate.auth.AuthApiKey(api_key),
    )


def format_docs_with_citations(docs):
    """Formats retrieved document chunks with citation metadata."""
    formatted_chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "Guideline")
        page = doc.metadata.get("page", "?")
        formatted_chunks.append(
            f"--- [Source {i}: {source} | Page: {page}] ---\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted_chunks)


def create_guideline_chain(client, k: int = 4):
    """Builds the standalone LangChain RAG chain."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = WeaviateVectorStore(
        client=client,
        index_name=INDEX_NAME,
        embedding=embeddings,
        text_key="text",
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model=GROQ_MODEL_NAME, api_key=groq_api_key, temperature=0.1)

    system_prompt = (
        "You are an expert Clinical Guideline Agent specializing in Polycystic Ovary Syndrome (PCOS).\n"
        "Your task is to answer clinical queries accurately based STRICTLY on the retrieved clinical guideline context provided below.\n\n"
        "Guidelines for your response:\n"
        "1. Ground every claim directly in the provided context. Do NOT extrapolate or hallucinate unstated medical thresholds.\n"
        "2. Explicitly cite the source document and page number for key clinical statements (e.g., [rotterdam_2003_consensus.pdf, p. 1]).\n"
        "3. If the provided context does not contain sufficient details to answer all aspects of the query, explicitly state what is missing.\n"
        "4. Structure your response clearly with bullet points or concise sections where appropriate.\n\n"
        "--- CLINICAL GUIDELINE CONTEXT ---\n"
        "{context}\n"
        "--- END CONTEXT ---"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    return retriever, prompt, llm


def answer_question(question: str, client=None, k: int = 4):
    """Runs retrieval and answer generation for a given question."""
    owns_client = False
    if client is None:
        client = get_weaviate_client()
        owns_client = True

    try:
        retriever, prompt, llm = create_guideline_chain(client, k=k)
        docs = retriever.invoke(question)
        context = format_docs_with_citations(docs)

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"file": d.metadata.get("source_file", "unknown"), "page": d.metadata.get("page", "?")}
                for d in docs
            ],
            "raw_context": context,
        }
    finally:
        if owns_client:
            client.close()


def run_evaluation_suite():
    """Runs a 5-question clinical evaluation suite against the standalone agent."""
    eval_set = [
        {
            "id": 1,
            "question": "What are the 3 core diagnostic criteria established in the Rotterdam 2003 consensus for PCOS?",
            "focus": "Diagnostic Criteria Definition & 2-out-of-3 Rule"
        },
        {
            "id": 2,
            "question": "What are the specific ultrasound threshold criteria (follicle count or ovarian volume) for defining polycystic ovarian morphology?",
            "focus": "Ultrasound Thresholds"
        },
        {
            "id": 3,
            "question": "Why would an elevated LH to FSH ratio be observed in a patient suspected of PCOS?",
            "focus": "Biochemical / Hormonal Ratios"
        },
        {
            "id": 4,
            "question": "What lifestyle and behavioral interventions are recommended as the first-line management approach for PCOS?",
            "focus": "First-Line Clinical Management"
        },
        {
            "id": 5,
            "question": "How should diagnostic assessment for PCOS differ between adolescents and adult women?",
            "focus": "Adolescent vs Adult Nuance"
        }
    ]

    print("=" * 75)
    print("      STANDALONE GUIDELINE AGENT EVALUATION SUITE (WEEK 1)")
    print("=" * 75)

    client = get_weaviate_client()
    try:
        for item in eval_set:
            print(f"\n[{item['id']}/5] EVAL QUESTION: {item['question']}")
            print(f"Target Domain: {item['focus']}")
            print("-" * 75)

            res = answer_question(item['question'], client=client, k=4)
            print("\n[GENERATED CLINICAL ANSWER]:")
            print(res["answer"])

            print("\n[CITATIONS RETRIEVED]:")
            for s in res["sources"]:
                print(f"  • {s['file']} (Page {s['page']})")
            print("=" * 75)

    finally:
        client.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        run_evaluation_suite()
    else:
        test_q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the Rotterdam criteria for diagnosing PCOS?"
        print(f"\n[QUERY]: {test_q}\n" + "=" * 70)
        result = answer_question(test_q)
        print("\n[ANSWER]:\n")
        print(result["answer"])
        print("\n[CITATIONS]:")
        for src in result["sources"]:
            print(f" • {src['file']} (Page {src['page']})")
