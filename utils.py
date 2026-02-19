import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

def load_ml100k(base_dir: str = "data/ml-100k") -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(f"{base_dir}/u.data", sep="\t",
                          names=["user","item","rating","timestamp"], engine="python")

    movie_cols = ["item","title","release_date","video_release_date","imdb_url"] + [f"g{i}" for i in range(19)]
    movies_raw = pd.read_csv(f"{base_dir}/u.item", sep="|", names=movie_cols,
                             encoding="latin-1", engine="python")

    genre_names = [
        "Unknown","Action","Adventure","Animation","Children's","Comedy","Crime",
        "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery",
        "Romance","Sci-Fi","Thriller","War","Western"
    ]
    movies_raw["genres"] = movies_raw[[f"g{i}" for i in range(19)]].apply(
        lambda r: "|".join([genre_names[i] for i,flag in enumerate(r) if flag==1]), axis=1)
    movies = movies_raw[["item","title","genres"]]
    return ratings, movies

def user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.pivot_table(index="user", columns="item", values="rating", aggfunc="mean")

def cosine_on_overlap(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 2: return 0.0
    va, vb = a[mask], b[mask]
    num = np.dot(va, vb)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(num/denom) if denom != 0 else 0.0

def topk_neighbors(M: pd.DataFrame, user_id: int, k: int = 30) -> List[Tuple[int,float]]:
    target = M.loc[user_id].to_numpy()
    sims = []
    for u in M.index:
        if u == user_id: continue
        s = cosine_on_overlap(target, M.loc[u].to_numpy())
        if s > 0:
            sims.append((u, s))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]

def score_items(M: pd.DataFrame, neighbors: List[Tuple[int,float]], user_id: int) -> Dict[int,float]:
    seen = set(M.loc[user_id].dropna().index)
    scores = {}
    for nb, sim in neighbors:
        row = M.loc[nb].dropna()
        for item, r in row.items():
            if item in seen: continue
            scores[item] = scores.get(item, 0.0) + sim * r
    return scores

def explain_item(M: pd.DataFrame, neighbors: List[Tuple[int,float]], item: int, topn=3):
    contrib = []
    for nb, sim in neighbors:
        if item not in M.columns: continue
        r = M.loc[nb, item]
        if not np.isnan(r):
            contrib.append((nb, float(r), sim, sim*r))
    contrib.sort(key=lambda x: x[3], reverse=True)
    return contrib[:topn]