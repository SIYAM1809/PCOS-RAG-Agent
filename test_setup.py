import os
import sys
from dotenv import load_dotenv

load_dotenv()

def test_groq():
    print("Testing Groq LLM...")
    try:
        from langchain_groq import ChatGroq
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key or groq_api_key == "your_key_here":
            print("[WARN] Skipping Groq test: GROQ_API_KEY not set.")
            return
        
        # Test with available models on this Groq key
        for model_name in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "qwen/qwen3.6-27b"]:
            try:
                llm = ChatGroq(model=model_name, api_key=groq_api_key)
                res = llm.invoke("Say hello in one word")
                print(f"[SUCCESS] Groq ({model_name}) response: {res.content.strip()}")
                break
            except Exception as e:
                print(f"Model {model_name} note: {e}")
    except Exception as e:
        print("[FAIL] Groq test failed:", e)

def test_weaviate():
    print("Testing Weaviate...")
    try:
        import weaviate
        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        if not weaviate_url or weaviate_url == "your_cluster_url":
            print("[WARN] Skipping Weaviate test: WEAVIATE_URL not set.")
            return
        
        # Clean URL format if needed
        clean_url = weaviate_url.strip()
        if not clean_url.startswith("http://") and not clean_url.startswith("https://"):
            clean_url = f"https://{clean_url}"
            
        if hasattr(weaviate, "connect_to_weaviate_cloud"):
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=clean_url,
                auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key)
            )
        else:
            client = weaviate.connect_to_wcs(
                cluster_url=clean_url,
                auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key)
            )
        print("[SUCCESS] Weaviate connected! is_ready:", client.is_ready())
        client.close()
    except Exception as e:
        print("[FAIL] Weaviate test failed:", e)

def test_neo4j():
    print("Testing Neo4j...")
    try:
        from neo4j import GraphDatabase
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        if not neo4j_uri or neo4j_uri == "your_uri_here":
            print("[WARN] Skipping Neo4j test: NEO4J_URI not set.")
            return
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        driver.verify_connectivity()
        print("[SUCCESS] Neo4j connected!")
        driver.close()
    except Exception as e:
        print("[FAIL] Neo4j test failed:", e)

if __name__ == "__main__":
    print("Running setup sanity checks...\n")
    test_groq()
    print("-" * 40)
    test_weaviate()
    print("-" * 40)
    test_neo4j()
    print("\nSanity check completed.")


