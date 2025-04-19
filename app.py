#!/usr/bin/env python3
import argparse
import os
import re
import faiss
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

# Configuration
EXCEL_PATH = r"E:\AI Model\elastic_search_testing\synthetic_candidates_with_skills.xlsx"
MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "candidate_index.faiss"
METADATA_PATH = "metadata.pkl"
N_LIST = 100     # HNSW connectivity
K_SEARCH = 20    # number of neighbors to search

# Load models
nlp = spacy.load("en_core_web_lg")
sbert = SentenceTransformer(MODEL_NAME)

# Helper: regex for exp range
def extract_range(text):
    m = re.search(r"(\d+)\s*(?:to|–|-)\s*(\d+).*", text)
    if not m:
        return None, None
    lo, hi = int(m.group(1)), int(m.group(2))
    if "year" in text.lower():
        lo, hi = lo*12, hi*12
    return lo, hi

# Parse filters
def parse_filters(query):
    doc = nlp(query)
    params = {"grade": None, "min_exp": None, "max_exp": None, "skills": []}
    # grade
    for tok in doc:
        if tok.text.upper() in {"A","B","C","TA","PA","SA","M","SM"}:
            params["grade"] = tok.text.upper(); break
    # exp
    lo, hi = extract_range(query)
    if lo is not None:
        params["min_exp"], params["max_exp"] = lo, hi
    # skills
    known = [s.lower() for s in ["python","java","docker","kubernetes","aws"]]
    ql = query.lower()
    for sk in known:
        if sk in ql:
            params["skills"].append(sk)
    return params

# Build and optionally save FAISS index
def build_index(save=True):
    # load data
    df = pd.read_excel(EXCEL_PATH)
    df = df.dropna(how="all")
    df["exp_in_company_months"] = pd.to_numeric(df["exp_in_company_months"], errors="coerce").fillna(0).astype(int)
    df["total_exp_months"] = pd.to_numeric(df["total_exp_months"], errors="coerce").fillna(0).astype(int)
    texts = df.apply(lambda r: f"{r.associate_name} Grade {r.grade} Exp {r.exp_in_company_months} mo Skills: {r.skills}", axis=1).tolist()
    embs = sbert.encode(texts, show_progress_bar=True).astype("float32")
    dim = embs.shape[1]
    # create HNSW index
    index = faiss.IndexHNSWFlat(dim, N_LIST)
    index.hnsw.efSearch = 50
    index.add(embs)
    # save index and metadata
    if save:
        faiss.write_index(index, FAISS_INDEX_PATH)
        df.to_pickle(METADATA_PATH)
        print(f"Saved FAISS index to {FAISS_INDEX_PATH} and metadata to {METADATA_PATH}")
    return index, df

# Load index and metadata
def load_index():
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Index or metadata not found. Run 'index' command first.")
    index = faiss.read_index(FAISS_INDEX_PATH)
    df = pd.read_pickle(METADATA_PATH)
    return index, df

# Search function
def search(index, df, query):
    params = parse_filters(query)
    print("Filters:", params)
    qemb = sbert.encode([query]).astype("float32")
    D, I = index.search(qemb, K_SEARCH)
    results = []
    for dist, idx in zip(D[0], I[0]):
        row = df.iloc[idx]
        # apply metadata filters
        if params["grade"] and row.grade != params["grade"]:
            continue
        if params["min_exp"] is not None and (row.exp_in_company_months < params["min_exp"] or row.exp_in_company_months > params["max_exp"]):
            continue
        if params["skills"]:
            skills_l = [s.strip().lower() for s in row.skills.split(",")]
            if not all(sk in skills_l for sk in params["skills"]):
                continue
        results.append([row.associate_name, row.grade, f"{row.exp_in_company_months} mo", f"{row.total_exp_months} mo", row.skills, round(float(dist), 4)])
    if results:
        print(tabulate(results, headers=["Name","Grade","InComp","Total","Skills","Score"], tablefmt="grid"))
    else:
        print("No results after filtering.")

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAISS-based Candidate Search CLI")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("index", help="Build and save FAISS index")
    sp = sub.add_parser("search", help="Search candidates")
    sp.add_argument("query", type=str, help="Natural language search query")
    args = parser.parse_args()

    if args.cmd == "index":
        build_index(save=True)
    elif args.cmd == "search":
        try:
            idx, df_meta = load_index()
        except FileNotFoundError as e:
            print(e)
            exit()
        search(idx, df_meta, args.query)
    else:
        parser.print_help()
