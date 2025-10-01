import json
import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ======================
# Load BioBERT model
# ======================
from huggingface_hub import login

# Collez ici votre clé d'accès Hugging Face
login("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")

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


def get_embedding(text: str):
    """Return the embedding vector for a given text using BioBERT."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use [CLS] token representation
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding


def cosine_similarity(vec1, vec2):
    return F.cosine_similarity(vec1, vec2).item()


# ======================
# Alignment statistics
# ======================
alignment_stats = {
    "nodes_total": 0,
    "nodes_aligned": 0,
    "nodes_mismatched": 0,
    "rels_total": 0,
    "rels_aligned": 0,
    "rels_mismatched": 0
}


def lookup_term_scored(label, ontology, max_results=10, threshold=0.7):
    """
    Lookup ontology term using OLS API + BioBERT embeddings.
    Règle de désambiguïsation :
      - Si correspondance exacte trouvée dans OLS → on retourne directement (pas de BioBERT).
      - Sinon → on calcule la similarité sémantique via BioBERT.
      - Si meilleur score < threshold → no_match.
    """
    url = "https://www.ebi.ac.uk/ols/api/search"
    params = {"q": label, "ontology": ontology, "exact": "false", "rows": max_results}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        docs = data.get("response", {}).get("docs", [])
        if not docs:
            return (label, "no_match", [])

        # ✅ Étape 1 : Vérifier si OLS retourne un match exact (sans BioBERT)
        for d in docs:
            cand_label = d.get("label", "")
            iri = d.get("iri", "")
            if cand_label and cand_label.lower() == label.lower():
                return iri, "exact", [
                    {"iri": iri, "label": cand_label, "api_score": 1.0, "semantic_sim": 1.0, "combined_score": 1.0}]

        # ✅ Étape 2 : Sinon, utiliser BioBERT pour comparer sémantiquement
        query_vec = get_embedding(label.lower())
        candidates = []
        for d in docs[:max_results]:
            iri = d.get("iri")
            cand_label = d.get("label", "")
            api_score = float(d.get("score", 0.5))  # score OLS
            if (api_score != float(0.5)):
                print(api_score)

            cand_vec = get_embedding(cand_label.lower())
            semantic_sim = cosine_similarity(query_vec, cand_vec)

            # semantic_sim si on veut
            combined_score = semantic_sim

            candidates.append({
                "iri": iri,
                "label": cand_label,
                "api_score": api_score,
                "semantic_sim": semantic_sim,
                "combined_score": combined_score
            })

        # Trier les candidats par score combiné
        candidates_sorted = sorted(candidates, key=lambda x: x["combined_score"], reverse=True)
        best = candidates_sorted[0]

        # ✅ Étape 3 : appliquer le seuil
        if best["combined_score"] < threshold:
            return (label, "no_match", candidates_sorted)

        return best["iri"], "scored", candidates_sorted

    except Exception as e:
        print(f"Error looking up '{label}' in ontology '{ontology}': {e}")
        return (label, "error", [])


# ======================
# Map node / relationship
# ======================
def map_node(label, node_type):
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
    iri, quality, _ = lookup_term_scored(label, ontology)
    if iri.startswith("http"):
        alignment_stats["nodes_aligned"] += 1
    else:
        alignment_stats["nodes_mismatched"] += 1
    return iri


def map_relationship(label):
    alignment_stats["rels_total"] += 1
    # Force "Binds to" as RO_000243
    if label.lower() in ["binds to", "binds"]:
        iri = "http://purl.obolibrary.org/obo/RO_000243"
        alignment_stats["rels_aligned"] += 1
        return iri
    iri, _, _ = lookup_term_scored(label, "ro")
    if iri.startswith("http"):
        alignment_stats["rels_aligned"] += 1
    else:
        alignment_stats["rels_mismatched"] += 1
    return iri


# ======================
# Transform LLM output
# ======================
def transform_graph(llm_output):
    transformed = []
    for edge in llm_output:
        try:
            node1_label = edge["node_1"].get("name", "")
            node2_label = edge["node_2"].get("name", "")
            node1_type = edge["node_1"].get("label", "")
            node2_type = edge["node_2"].get("label", "")
            rel_label = edge.get("relationship", "")
            descrptrelllm = edge.get("description", "")

            if not (node1_label and node2_label and rel_label):
                continue

            mapped_node1 = map_node(node1_label, node1_type)
            mapped_node2 = map_node(node2_label, node2_type)
            mapped_rel = map_relationship(rel_label)

            transformed.append({
                "node1": mapped_node1,
                "relationship_label": rel_label,
                "relationship_iri": mapped_rel,
                "node2": mapped_node2,
                "descrptrelllm": descrptrelllm
            })

        except KeyError as e:
            print(f"Missing key in input: {e}")

    return transformed


# ======================
# Main Execution
# ======================
input_file = "data/kg_llm_relationships_all.json"
output_file = "data/standardized_graph_embeddings.json"
stats_file = "data/alignment_stats.json"

with open(input_file, "r", encoding="utf-8") as f:
    llm_output = json.load(f)

standardized_graph = transform_graph(llm_output)

# Save the standardized graph
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(standardized_graph, f, indent=2)
print(f"Saved standardized graph: {output_file}")

# Save alignment statistics separately
with open(stats_file, "w", encoding="utf-8") as f:
    json.dump(alignment_stats, f, indent=2)
print(f"Saved alignment statistics: {stats_file}")
