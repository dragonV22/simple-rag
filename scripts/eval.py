import requests
import json

BASE = "http://127.0.0.1:8000"

TESTS = [
    {
        "namespace": "default",
        "question": "What is RAG?",
        "must_contain": ["retrieval", "generation"],
    },
    {
        "namespace": "default",
        "question": "If the context doesn't contain an answer, what should you say?",
        "must_contain": ["I don't know"],
    },
]

def run():
    ok = 0
    for t in TESTS:
        r = requests.post(f"{BASE}/rag/ask", json={
            "namespace": t["namespace"],
            "question": t["question"],
            "top_k": 5,
            "temperature": 0.2
        })
        r.raise_for_status()
        data = r.json()
        text = data.get("text", "")

        passed = all(s.lower() in text.lower() for s in t["must_contain"])
        print("\nQ:", t["question"])
        print("A:", text[:300].replace("\n", " "))
        print("Pass:", passed)

        ok += 1 if passed else 0

    print(f"\nSCORE: {ok}/{len(TESTS)}")

if __name__ == "__main__":
    run()