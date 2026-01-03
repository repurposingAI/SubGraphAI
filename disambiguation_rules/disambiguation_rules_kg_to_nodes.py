import json
import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
from functools import lru_cache

# ======================
# HuggingFace Login
# ======================
login("")

# ======================
# Device + Model
# ======================
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ======================
# Embedding utilities
# ======================
@lru_cache(maxsize=50_000)
def get_embedding_cached(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]
        emb = F.normalize(emb, dim=1)

    return emb.cpu()


def embed_batch(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]
        emb = F.normalize(emb, dim=1)

    return emb.cpu()

# ======================
# Alignment statistics
# ======================
alignment_stats = {
    "nodes_total": 0,
    "nodes_aligned": 0,
    "nodes_unaligned": 0,
    "nodes_mismatched": 0,
    "rels_total": 0,
    "rels_aligned": 0,
    "rels_unaligned": 0,
    "rels_mismatched": 0
}

# ======================
# Cached OLS API
# ======================
@lru_cache(maxsize=20_000)
def ols_search(label, ontology, max_results):
    url = "https://www.ebi.ac.uk/ols/api/search"
    params = {
        "q": label,
        "ontology": ontology,
        "exact": "false",
        "rows": max_results
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()

# ======================
# Ontology lookup + scoring
# ======================
def lookup_term_scored(label, ontology, max_results=10, threshold=0.7):
    try:
        if ontology is None:
            return (label, "error", [])

        data = ols_search(label, ontology, max_results)
        docs = data.get("response", {}).get("docs", [])

        if not docs:
            return (label, "no_match", [])

        # Exact lexical match
        for d in docs:
            if d.get("label", "").lower() == label.lower():
                iri = d.get("iri")
                return iri, "exact", [{
                    "iri": iri,
                    "label": d["label"],
                    "semantic_sim": 1.0
                }]

        # Semantic similarity (batched)
        query_vec = get_embedding_cached(label.lower())

        cand_labels = [d.get("label", "") for d in docs]
        cand_iris = [d.get("iri") for d in docs]

        cand_vecs = embed_batch([lbl.lower() for lbl in cand_labels])
        sims = F.cosine_similarity(query_vec, cand_vecs).tolist()

        candidates = [
            {"iri": iri, "label": lbl, "semantic_sim": sim}
            for iri, lbl, sim in zip(cand_iris, cand_labels, sims)
        ]

        candidates.sort(key=lambda x: x["semantic_sim"], reverse=True)
        best = candidates[0]

        # Threshold filtering
        if best["semantic_sim"] < threshold:
            return (label, "no_match", candidates)

        return best["iri"], "scored", candidates

    except Exception as e:
        print(f"Error looking up '{label}' in ontology '{ontology}': {e}")
        return (label, "error", [])

# ======================
# Node mapping
# ======================
def map_node(label, node_type, threshold=0.7):
    alignment_stats["nodes_total"] += 1

    ontology_mapping = {
        "chemical": "chebi",
        "drug": "chebi",
        "compound": "chebi",
        "molecule": "chebi",
        "protein": "pr",
        "gene": "pr",
        "disease": "doid",
        "anatomy": "uberon",
        "cellular component": "go",
        "pathway": "pw",
        "symptom": "hp",
        "side effect": "hp"
    }

    ontology = ontology_mapping.get(node_type.lower())
    iri, quality, _ = lookup_term_scored(label, ontology, threshold=threshold)

    if quality in ["exact", "scored"]:
        alignment_stats["nodes_aligned"] += 1
    elif quality == "no_match":
        alignment_stats["nodes_unaligned"] += 1
    else:
        alignment_stats["nodes_mismatched"] += 1
    print(quality)

    return iri

# ======================
# Relationship mapping
# ======================
def map_relationship(label, threshold=0.7):
    alignment_stats["rels_total"] += 1

    if label.lower() in ["binds to", "binds"]:
        alignment_stats["rels_aligned"] += 1
        return "http://purl.obolibrary.org/obo/RO_000243"

    iri, quality, _ = lookup_term_scored(label, "ro", threshold=threshold)

    if quality in ["exact", "scored"]:
        alignment_stats["rels_aligned"] += 1
    elif quality == "no_match":
        alignment_stats["rels_unaligned"] += 1
    else:
        alignment_stats["rels_mismatched"] += 1

    return iri

# ======================
# Transform LLM output
# ======================
def transform_graph(llm_output, threshold=0.7):
    transformed = []

    for edge in llm_output:
        try:
            node1_label = edge["node_1"].get("name", "")
            node2_label = edge["node_2"].get("name", "")
            node1_type = edge["node_1"].get("label", "")
            node2_type = edge["node_2"].get("label", "")
            rel_label = edge.get("relationship", "")
            descr = edge.get("description", "")

            if not (node1_label and node2_label and rel_label):
                continue

            transformed.append({
                "node1": map_node(node1_label, node1_type, threshold),
                "relationship_label": rel_label,
                "relationship_iri": map_relationship(rel_label, threshold),
                "node2": map_node(node2_label, node2_type, threshold),
                "descrptrelllm": descr
            })

        except KeyError as e:
            print(f"Missing key: {e}")

    return transformed

# ======================
# Main execution
# ======================
input_file = "data/kg_llm_relationships_all.json"
output_file = "data/standardized_graph_embeddings.json"
stats_file = "data/alignment_stats.json"

with open(input_file, "r", encoding="utf-8") as f:
    llm_output = json.load(f)

standardized_graph = transform_graph(llm_output, threshold=0.7)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(standardized_graph, f, indent=2)

with open(stats_file, "w", encoding="utf-8") as f:
    json.dump(alignment_stats, f, indent=2)

print("Standardized graph and alignment statistics saved")
