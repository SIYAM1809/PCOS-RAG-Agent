import asyncio
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv()

import instructor
from groq import AsyncGroq
from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import Faithfulness, ContextRecall
from graph.build_graph import build_graph, _make_initial_state
from eval.ragas_dataset import eval_cases

JUDGE_MODEL = "qwen/qwen3.8-27b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

async def main():
    print("=" * 70)
    print("  DIAGNOSTIC ANALYSIS: Case 1 Faithfulness & Cases 2/4 Context Recall")
    print("=" * 70)

    # 1. Initialize LangGraph and run Case 1, 2, 4 to get live chunks and answers
    print("\n[1] Building graph...")
    app, graph_res = build_graph()
    
    target_ids = [1, 2, 4]
    data = {}
    try:
        for case in eval_cases:
            if case["id"] in target_ids:
                print(f"\n--- Running Case {case['id']} through LangGraph ---")
                res = app.invoke(_make_initial_state(query=case["question"]))
                ans = res.get("final_answer", "")
                chunks = res.get("retrieved_chunks", [])
                if not chunks and res.get("guideline_context"):
                    chunks = [res["guideline_context"]]
                data[case["id"]] = {
                    "case": case,
                    "answer": ans,
                    "chunks": chunks
                }
                print(f"Case {case['id']}: Answer len={len(ans)}, Chunks={len(chunks)}")
    finally:
        graph_res.close()

    # 2. Setup Judge LLM
    print("\n[2] Setting up Judge LLM (qwen/qwen3.8-27b)...")
    raw_client = AsyncGroq(timeout=60.0)
    patched = instructor.from_groq(raw_client, mode=instructor.Mode.TOOLS)
    judge_llm = InstructorLLM(
        client=patched,
        model=JUDGE_MODEL,
        provider="groq",
        model_args=InstructorModelArgs(temperature=0.0, max_tokens=2048),
    )

    # 3. Diagnose Case 1 Faithfulness
    print("\n" + "=" * 70)
    print("  CASE 1: FAITHFULNESS DIAGNOSTIC")
    print("=" * 70)
    c1 = data[1]
    print(f"Question: {c1['case']['question']}\n")
    print(f"Generated Answer:\n{c1['answer']}\n")
    print("Retrieved Chunks:")
    for idx, ch in enumerate(c1["chunks"], 1):
        print(f"--- Chunk {idx} (len={len(ch)}) ---")
        print(ch[:300] + ("..." if len(ch) > 300 else ""))
        print()

    faith_metric = Faithfulness(llm=judge_llm)
    try:
        f_res = await faith_metric.ascore(
            user_input=c1["case"]["question"],
            response=c1["answer"],
            retrieved_contexts=c1["chunks"]
        )
        print(f"\n[Case 1 Faithfulness Score]: {f_res.value}")
        for attr in dir(f_res):
            if not attr.startswith("_"):
                try:
                    val = getattr(f_res, attr)
                    if not callable(val):
                        print(f"  {attr}: {val}")
                except Exception:
                    pass
    except Exception as e:
        print(f"Faithfulness evaluation error: {e}")

    # 4. Diagnose Case 2 & Case 4 Context Recall
    cr_metric = ContextRecall(llm=judge_llm)
    for cid in [2, 4]:
        print("\n" + "=" * 70)
        print(f"  CASE {cid}: CONTEXT RECALL DIAGNOSTIC")
        print("=" * 70)
        c = data[cid]
        gt = c["case"]["ground_truth"]
        print(f"Question: {c['case']['question']}\n")
        print(f"Ground Truth:\n{gt}\n")
        print("Retrieved Chunks:")
        for idx, ch in enumerate(c["chunks"], 1):
            print(f"--- Chunk {idx} (len={len(ch)}) ---")
            print(ch[:300] + ("..." if len(ch) > 300 else ""))
            print()

        try:
            cr_res = await cr_metric.ascore(
                user_input=c["case"]["question"],
                retrieved_contexts=c["chunks"],
                reference=gt
            )
            print(f"\n[Case {cid} Context Recall Score]: {cr_res.value}")
            for attr in dir(cr_res):
                if not attr.startswith("_"):
                    try:
                        val = getattr(cr_res, attr)
                        if not callable(val):
                            print(f"  {attr}: {val}")
                    except Exception:
                        pass
        except Exception as e:
            print(f"Context Recall error for Case {cid}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
