# src/llm_tags.py
from __future__ import annotations
import argparse, os, json
from collections import Counter
from typing import Tuple
from utils import load_ml100k

# Try to use Ollama; fall back to a deterministic summary if unavailable
USE_OLLAMA = True
try:
    from ollama import chat
except Exception:
    USE_OLLAMA = False

def build_prompt(user_id: int, base_dir: str = "data/ml-100k") -> Tuple[str, str]:
    ratings, movies = load_ml100k(base_dir)
    df = ratings.merge(movies, on="item", how="left")
    u = df[df["user"] == user_id]
    if u.empty:
        raise ValueError(f"user {user_id} has no ratings")

    # genre counts for quick context
    genres = []
    for g in u["genres"].fillna(""):
        for gg in g.split("|"):
            if gg:
                genres.append(gg)
    counts = Counter(genres)
    top = ", ".join([f"{g}({c})" for g, c in counts.most_common(5)])

    # recent titles for flavor
    recent = u.sort_values("timestamp", ascending=False).head(12)
    recent_txt = "; ".join(f"{t}={r}" for t, r in zip(recent["title"], recent["rating"]))

    prompt = (
        "You are a movie recommendation assistant.\n"
        f"User ID: {user_id}\n"
        f"Top genres by count: {top}\n"
        f"Recent ratings: {recent_txt}\n\n"
        "Task: Write 1–2 concise sentences describing this user's taste, "
        "then list exactly 5 tags (2–3 words each). Avoid movie titles."
    )
    return prompt, top

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_id", type=int, default=196)
    ap.add_argument("--data_dir", type=str, default="data/ml-100k")
    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    prompt, top = build_prompt(args.user_id, args.data_dir)

    if USE_OLLAMA:
        # <-- This is the line that needs the import above
        res = chat(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": "Be concise and specific."},
                {"role": "user", "content": prompt},
            ],
        )["message"]["content"].strip()
    else:
        # Fallback if Ollama not installed or import failed
        res = (
            f"Profile: likely prefers the listed top genres. "
            f"Tags: {top}. (Fallback summary)"
        )

    out_path = os.path.join(args.out_dir, f"tags_user_{args.user_id}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(res)

    print(res)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()