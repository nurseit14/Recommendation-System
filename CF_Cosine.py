import argparse, json, os
from utils import load_ml100k, user_item_matrix, topk_neighbors, score_items, explain_item

def recommend(user_id: int, top_n=5, k_neighbors=30, base_dir="data/ml-100k"):
    ratings, movies = load_ml100k(base_dir)
    M = user_item_matrix(ratings)

    if user_id not in M.index:
        raise ValueError(f"user_id {user_id} not found (1–943)")

    neighbors = topk_neighbors(M, user_id, k_neighbors)
    scores = score_items(M, neighbors, user_id)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    id2title = movies.set_index("item")["title"].to_dict()
    results = []
    for item_id, wscore in ranked:
        contrib = explain_item(M, neighbors, item_id)
        reasons = [f"U{nb} rated {r:.1f} (sim {s:.2f})" for nb, r, s, _ in contrib]
        results.append({
            "movie": id2title.get(item_id, str(item_id)),
            "score": round(wscore, 3),
            "reasons": reasons
        })
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_id", type=int, default=196)
    ap.add_argument("--top_n", type=int, default=5)
    args = ap.parse_args()

    os.makedirs("outputs", exist_ok=True)
    recs = recommend(args.user_id, args.top_n)

    print(f"\nTop-{args.top_n} recommendations for User {args.user_id}:\n")
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r['movie']} (score={r['score']})")
        for rr in r["reasons"]:
            print(f"   {rr}")
    json.dump(recs, open(f"outputs/recs_user_{args.user_id}.json","w"), indent=2, ensure_ascii=False)