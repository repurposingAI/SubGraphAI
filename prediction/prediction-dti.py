import csv
import json

import pandas as pd
import timeout_decorator
from langchain.memory import ConversationBufferMemory
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from langchain.schema import HumanMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import display, Image
from bioservices import KEGG  # For KEGG
from dotenv import load_dotenv


# G-Retriever -  GraphRAG with GAT
import pdb
from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.nn import GAT
from torch_geometric.nn.models import GRetriever
from torch_geometric.nn.nlp import LLM
import warnings

from huggingface_hub import login

# Collez ici votre clé d'accès Hugging Face
login("")

# Check if GPU is available and set device accordingly
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f"Using device: {device}")

def load_policy(file_path="policy.txt"):
    args = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                args[key.strip()] = value.strip()
    return args

# Example usage
config = load_policy()

# Access values
node_descriptions_file = config["node_descriptions_file"]
source_nodes_file = config["source_nodes_file"]
target_nodes_file = config["target_nodes_file"]
attribut_file = config["attribut_file"]
drug_embedding_file = config["drug_embedding_file"]
target_embedding_file = config["target_embedding_file"]
missing_embedding_file = config["missing_embedding_file"]
input_file = config["input_file"]
file2_path = config["file2_path"]
file_path = config["file_path"]
ontology_file = config["ontology_file"]
validation_dataset = config["validation_dataset"]
strategies_workflow = config["strategies_workflow"]



# Initialize Sentence Transformer for embeddings
model = SentenceTransformer('all-MPNet-base-v2')

# Load node_descriptions_raw, source_nodes, and target_nodes from files
with open(node_descriptions_file, "r") as f:
    node_descriptions_raw = json.load(f)

with open(source_nodes_file, "r") as f:
    source_nodes = json.load(f)

with open(target_nodes_file, "r") as f:
    target_nodes = json.load(f)
# print(node_descriptions_raw[0])

with open(attribut_file, "r") as f:
    edge_attributes_raw = json.load(f)

relation_types = sorted(list(set(attr["relation"] for attr in edge_attributes_raw)))
relation_to_idx = {rel: idx for idx, rel in enumerate(relation_types)}

# Parse node descriptions and create a mapping of id -> index
node_descriptions = {
    node["id"]: node["properties"].get("source", "no source information")
    for node in node_descriptions_raw
}
print(len(node_descriptions))
# pdb.set_trace()


import json

##source_nodes=source_nodes[:1000]
# target_nodes=target_nodes[:1000]
# Ensure all IDs in the edges exist in the node_descriptions
unique_ids = set(source_nodes) | set(target_nodes)
assert unique_ids.issubset(node_descriptions.keys()), "Some IDs in edges are missing in node_descriptions!"
node_descriptions_name = {
    node["id"]: node["properties"].get("name", "no source information")
    for node in node_descriptions_raw
}
node_descriptions = {
    node["id"]: node["properties"].get("source", "no source information")
    for node in node_descriptions_raw
}
print(len(node_descriptions))
# Map node IDs to indices
id_to_index = {node_id: idx for idx, node_id in enumerate(node_descriptions.keys())}

# Map node IDs to indices
name_to_id = {name: node_id for node_id, name in node_descriptions_name.items()}
# Map node IDs to indices


# Generate embeddings for node descriptions
node_features = torch.tensor(
    model.encode([node_descriptions[node_id] for node_id in node_descriptions.keys()]),
    dtype=torch.float
)

# Build edge_index using the index mapping
edge_index = torch.tensor([
    [id_to_index[src] for src in source_nodes],  # Convert source IDs to indices
    [id_to_index[tgt] for tgt in target_nodes]  # Convert target IDs to indices
], dtype=torch.long)

edge_attr = torch.zeros(len(source_nodes), len(relation_types), dtype=torch.float)

# Batch vector for a single graph
batch = torch.zeros(len(node_descriptions), dtype=torch.long)



# Move data to the correct device
node_features = node_features.to(device)
edge_index = edge_index.to(device)
batch = batch.to(device)

for attr in edge_attributes_raw:
    src = attr["source"]
    tgt = attr["target"]
    rel = attr["relation"]

    # Trouver toutes les positions où cette paire (src,tgt) apparaît
    for i, (s, t) in enumerate(zip(source_nodes, target_nodes)):
        if s == src and t == tgt:
            edge_attr[i, relation_to_idx[rel]] = 1.0
# pdb.set_trace()

# Déplacer sur le bon device
edge_attr = edge_attr.to(device)
# Debugging output
print("Node Features (Embeddings):", node_features.shape)
print("Edge Index:", edge_index)

# Configuration du GAT
gnn = GAT(
    in_channels=768,  # Première ligne
    hidden_channels=1024,  # Alignée avec la première
    out_channels=768,  # Même indentation
    num_layers=4,  # Même indentation
    heads=8,  # Même indentation + virgule
    edge_dim=len(relation_types)  # Dernier paramètre sans virgule
).to(device)  # Méthode .to() alignée

# Initialize LLM
llm = LLM(model_name='meta-llama/Llama-2-7b-chat-hf', num_params=7).to(device)  # **Move LLM to device**


from typing import List, Optional

from torch import Tensor

from torch_geometric.nn.nlp.llm import BOS, LLM, MAX_NEW_TOKENS
from torch_geometric.utils import scatter
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F






# Suppress specific warnings (e.g., CPU usage warnings)
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.nn.nlp.llm")



# Example node_descriptions in JSON-like format
# node_descriptions_raw = [
# {"id": 2, "properties": {"source": "METHODS Three chemometric approaches were applied for the stability indicating determination of SUM in presence of NAP Classical least squares CLS Partial least squares PLS and Princip"}},
# {"id": 6, "properties": {"source": "Another text for embedding generation"}},
# {"id": 7, "properties": {"source": "Yet another example of node description"}}
# ]

# Example source and target node lists (unordered IDs)
# source_nodes = [2, 7, 6, 2]
# target_nodes = [6, 2, 7, 7]

import json



# Define GRetriever class
from typing import List, Optional

from torch import Tensor

from torch_geometric.nn.nlp.llm import BOS, LLM, MAX_NEW_TOKENS






# Initialize G-Retriever with the real GNN
g_retriever = GRetriever(llm=llm, gnn=gnn, mlp_out_channels=4096).to(device)

# Complex query without additional context
# query = [
#     "Do Biperiden and Muscarinic acetylcholine receptor M3 interact?"
# ]


query = [
    "Do Pentobarbital and Amyloid-beta precursor protein interact?"
]

# Perform inference without additional context
with torch.no_grad():
    g_retriever.gnn = g_retriever.gnn.to(device)
    # Étape 1: Propagation dans le GNN pour obtenir les embeddings des nœuds
    node_embeddings = g_retriever.gnn(
        x=node_features,  # Features des nœuds
        edge_index=edge_index,  # Connexions des arêtes
        batch=batch  # Batch d'exemples
    )
    subgraph_context = g_retriever.inference(
        question=query,
        x=node_features,  # Node embeddings
        edge_index=edge_index,  # Complex edge connections
        edge_attr=edge_attr,
        batch=batch,
        max_out_tokens=1024
    )


# Print results
print(subgraph_context)
######################Extract_embedding###########
import os

# Load node_embeddings and id_to_index
# node_embeddings is assumed to be a dictionary {index: embedding_vector}
# id_to_index is assumed to map drug/protein names to their index in node_embeddings
# Parse node descriptions and create a mapping of id -> index

def process_file(input_file):

    # Sets to track processed drugs and proteins
    processed_drugs = set()
    processed_proteins = set()

    # Open output files
    with open(drug_embedding_file, 'w') as drug_file, \
            open(target_embedding_file, 'w') as target_file, \
            open(missing_embedding_file, 'w') as missing_file:

        # Process the input file
        with open(input_file, 'r') as infile:
            for line in infile:
                drug, protein, name = line.strip().split('\t')

                # Retrieve embedding for drug
                if drug not in processed_drugs:
                    if drug in name_to_id:
                        try:
                            drug_embedding = node_embeddings[id_to_index[name_to_id[drug]]]
                            drug_file.write(f"{drug}\t{drug_embedding}\n")
                            processed_drugs.add(drug)
                        except KeyError:
                            missing_file.write(f"{drug}\n")
                    else:
                        missing_file.write(f"{drug}\n")

                # Retrieve embedding for protein
                if protein not in processed_proteins:
                    if protein in name_to_id:
                        try:
                            protein_embedding = node_embeddings[id_to_index[name_to_id[protein]]]
                            target_file.write(f"{protein}\t{protein_embedding}\n")
                            processed_proteins.add(protein)
                        except KeyError:
                            missing_file.write(f"{protein}\n")
                    else:
                        missing_file.write(f"{protein}\n")


process_file(input_file)

####################################################

# Afficher les embeddings générés par le GNN
print("Embeddings générés par le GNN (taille):", node_embeddings.shape)

# Utilisez la moyenne comme méthode d'agrégation (vous pouvez changer selon vos besoins)
aggregated_gnn_embedding = node_embeddings.mean(dim=0)  # Taille : [out_channels]
print(aggregated_gnn_embedding.shape)

###################G-RETRIEVER + react######################################

import requests


def biosearch_tool(entity_name: str) -> str:
    """
    Search for biological information about a gene, protein, or drug using the UniProt API.
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    query_params = {
        "query": entity_name,
        "fields": "accession,protein_name,gene_names,organism,comment,function",
        "format": "json",
        "size": 1  # Limit results to 1 for simplicity
    }

    try:
        # Send GET request to UniProt API
        response = requests.get(base_url, params=query_params)
        response.raise_for_status()  # Raise an error if the request fails

        # Parse response JSON
        data = response.json()
        if 'results' in data and len(data['results']) > 0:
            entry = data['results'][0]  # Take the first result
            protein_name = entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get(
                "value", "N/A")
            gene_names = ", ".join(
                [gene['value'] for gene in entry.get("genes", {}).get("gene", [])]) if "genes" in entry else "N/A"
            organism = entry.get("organism", {}).get("scientificName", "N/A")
            function = next(
                (comment['texts'][0]['value'] for comment in entry.get("comments", [])
                 if comment.get('type') == 'FUNCTION'), "No function description available."
            )

            # Format result string
            result = (
                f"**Protein Name**: {protein_name}\n"
                f"**Gene Name(s)**: {gene_names}\n"
                f"**Organism**: {organism}\n"
                f"**Function**: {function}"
            )
        else:
            result = "No information found in UniProt for the given query."

    except Exception as e:
        result = f"An error occurred while querying UniProt: {str(e)}"

    return result


from langchain.agents import initialize_agent, Tool
from langchain.tools import tool
from langchain.llms import HuggingFacePipeline
#from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentType
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import GAT
from torch_geometric.nn.models import GRetriever
from torch_geometric.nn.nlp import LLM
import torch
import warnings
from chembl_webresource_client.new_client import new_client  # For ChEMBL
def check_drugbank_interaction(drug_name, target_protein, dictdtref2):
    if drug_name in dictdtref2:
        targets = dictdtref2[drug_name]
        for trg in targets:
            if trg == target_protein:
                return True
    return False


def check_chembl_interaction(drug_name, target_protein):
    """
    Check if the given drug-target interaction is available in ChEMBL.
    """
    molecule = new_client.molecule
    target = new_client.target

    mols = molecule.filter(pref_name__iexact=drug_name).only('molecule_chembl_id')
    df_mols = pd.DataFrame(mols)
    if df_mols.empty:
        return False
    drug_chembl_id = df_mols.iloc[0]['molecule_chembl_id']

    res = target.filter(target_components__accession=target_protein)
    df_target = pd.DataFrame(res)
    if df_target.empty:
        return False
    target_chembl_ids = df_target['target_chembl_id'].tolist()

    activity = new_client.activity.filter(molecule_chembl_id=drug_chembl_id,
                                          target_chembl_id__in=target_chembl_ids).only(['pchembl_value'])
    df_activity = pd.DataFrame(activity)
    return not df_activity.empty

# -------------------------------------------------------------------
# External Interaction Checks
# -------------------------------------------------------------------
def read_csv_and_focus_on_drug_target(file_path, dictdtref2):
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)
        for row in reader:
            if str(row[1]) not in dictdtref2:
                dictdtref2[str(row[1])] = set()
            dictdtref2[str(row[1])].add(str(row[4]))
    return dictdtref2
def read_csv_name_to_uniprot_id(file_path, dictdtref):
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)
        for row in reader:
            if str(row[1]) not in dictdtref:
                dictdtref[str(row[1])] = set()
            dictdtref[str(row[1])].add(str(row[5]))
    return dictdtref


def check_kegg_interaction(drug_name, target_protein):
    """
    Check if the given drug-target interaction is available in KEGG.
    """
    kegg = KEGG()  # Uses default KEGG REST endpoint.
    results = kegg.find("drug", drug_name)
    if results:
        drug_entry = results.split(';')[0]
        parts = drug_entry.split()
        if parts:
            drug_id = parts[0].split(":")[1]
            try:
                pathways_info = kegg.get(drug_id).split("PATHWAY")
                if len(pathways_info) > 1:
                    pathways_section = pathways_info[1]
                    subpathw = pathways_section.split("INTERACTION")[0].split("\n")
                    for z in subpathw:
                        l = z.strip()
                        if l:
                            str1 = "path:" + l.split("(")[0].strip()
                            pathway_entries = kegg.parse(kegg.get(str1))
                            if pathway_entries and "GENE" in pathway_entries:
                                for gene_id in pathway_entries["GENE"]:
                                    gn = pathway_entries["GENE"][gene_id].split(";")[0].strip()
                                    if target_protein == gn:
                                        return True
            except Exception as e:
                print("KEGG error:", e)
    return False


def uniprot_to_gene_name(uniprot_id):
    """
    Retrieve the gene name corresponding to a given UniProt ID.
    """
    api_url = f'https://www.uniprot.org/uniprot/{uniprot_id}.json'
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            gene_name_v = data['genes']
            gene_name = gene_name_v[0]['geneName']['value']
            return gene_name
        else:
            print(f"Error: Unable to fetch data from UniProt API. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

dictdtref = {}
dictdtref2 = {}
dictdtref = read_csv_name_to_uniprot_id(file_path, dictdtref)
dictdtref2 = read_csv_and_focus_on_drug_target(file2_path, dictdtref2)
def ask_ontology_DT(entities_uri1: str, entities_uri2: str):
    import rdflib

    # Load your ontology into an rdflib graph.
    g = rdflib.Graph()
    g.parse(ontology_file)

    # Define the URIs for the drug and the target.
    # These URIs should match those present in your ontology.
    # drug_uri = "http://purl.obolibrary.org/obo/CHEBI_34107"      # Example: dimenhydrinate
    # target_uri = "http://purl.obolibrary.org/obo/PR_000006122"    # Example: cytochrome P450 2B6

    drug_uri =entities_uri1
    target_uri = entities_uri2
    # The SPARQL query uses:
    # - the prefix "my" for classes defined in your custom ontology.
    # - the prefix "ro" for the “Binds to” relationship property.
    # We use an ASK query to verify if an interaction exists.

    # Define the predicate URIs representing different binding or interaction relationships.
    binding_predicates = """
        <http://purl.obolibrary.org/obo/RO_000243>,
        <http://purl.obolibrary.org/obo/RO_0002434>,
        <http://purl.obolibrary.org/obo/RO_0002437>
    """

    # Use an ASK query to verify if an interaction exists via any of the binding predicates.
    sparql_query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    ASK {{
      <{drug_uri}> ?relation <{target_uri}> .
      FILTER (?relation IN ({binding_predicates}))
    }}
    """

    # Execute the query.
    result = g.query(sparql_query)
    print(result)
    db_valid=False
    chembl_valid=False
    kegg_valid=False

    db_valid = check_drugbank_interaction(drug_uri, target_uri, dictdtref2)
    chembl_valid = check_chembl_interaction(drug_uri, target_uri)
    nameprot_uniprot = dictdtref.get(target_uri, set())
    # If there is at least one UniProt ID for the target protein, get one.
    if nameprot_uniprot:
        text_val = nameprot_uniprot.pop()
        kegg_valid = check_kegg_interaction(drug_uri, uniprot_to_gene_name(text_val))
    else:
        kegg_valid = False
        # If any external validation is positive, add a validatedBy triple.

    # ASK queries return a boolean value.
    # In rdflib, iterating over the result gives us a row where the boolean is the first element.
    print("Proteins that interact with the drug:")
    for row in result:
        print (str(row))
        return("Ask Ontology",str(row),"drugbank_valid",str(db_valid),"chembl_valid",str(chembl_valid),"kegg_valid",str(kegg_valid) )

### NCBO BioPortal API Configuration ###
### NCBO BioPortal API Configuration ###
API_KEY = "bd2d2269-59a8-4559-adc5-cd5aa4bcb181"
import urllib.parse
def get_annotations_from_iri_bioportal(iri, ontology_acronym):
    """
    Retrieve annotation details for the given IRI from the NCBO BioPortal API.
    Returns a dictionary with keys: 'label', 'definition', and 'synonyms'.
    """
    """
        Retrieve annotation details for the given IRI from the NCBO BioPortal API.
        Returns a dictionary with keys: 'label', 'definition', and 'synonyms'.
        """
    encoded_iri = urllib.parse.quote(iri, safe='')
    url = f"https://data.bioontology.org/ontologies/{ontology_acronym}/classes/{encoded_iri}"
    headers = {
        "Authorization": f"apikey token={API_KEY}",
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        print("inside get_annotations_from_iri")
        annotations = {
            "label": data.get("prefLabel", iri.split("/")[-1]),
            "definition": None,
            "synonyms": []
        }
        if "definition" in data:
            definitions = data["definition"]
            if isinstance(definitions, list) and definitions:
                annotations["definition"] = definitions[0]
            else:
                annotations["definition"] = definitions
        if "synonym" in data:
            synonyms = data["synonym"]
            if isinstance(synonyms, list):
                annotations["synonyms"] = synonyms
            else:
                annotations["synonyms"] = [synonyms]
        annotlist= list(annotations.items())
        return annotlist
    except requests.exceptions.RequestException as e:
        print(f"BioPortal API error for '{iri}': {e}")
    annotlist= list({"label": iri.split("/")[-1], "definition": None, "synonyms": []}.items())
    return annotlist


import requests
import urllib.parse


def get_annotations_from_iri(iri, ontology_acronym):
    """
    Retrieve annotation details for the given IRI from the OLS API.
    Returns a list of tuples with keys: 'label', 'description', and 'synonyms'.
    """
    encoded_iri = urllib.parse.quote(iri, safe='')
    url = f"https://www.ebi.ac.uk/ols/api/ontologies/{ontology_acronym}/terms?iri={encoded_iri}"
    headers = {"Accept": "application/json"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # OLS returns the term details under _embedded -> terms (a list)
        terms = data.get('_embedded', {}).get('terms', [])
        if not terms:
            raise ValueError("No terms found")
        term = terms[0]

        # Extract label (use the IRI fragment as a fallback)
        label = term.get("label", iri.split("/")[-1])

        # Extract description: OLS returns this as a list. Use the first item if available.
        desc_list = term.get("description", [])
        description = desc_list[0] if isinstance(desc_list, list) and desc_list else None

        # Extract synonyms: try the "synonyms" key; if empty, check "obo_synonym" entries for the "name" field.
        synonyms = term.get("synonyms", [])
        if not synonyms:
            obo_syn = term.get("obo_synonym", [])
            try:
                synonyms = [syn.get("name") for syn in obo_syn if "name" in syn]
            except:
                synonyms=[]

        annotations = {
            "label": label,
            "description": description,
            "synonyms": synonyms
        }
        return list(annotations.items())

    except requests.exceptions.RequestException as e:
        print(f"OLS API error for '{iri}': {e}")
        return list({"label": iri.split("/")[-1], "description": None, "synonyms": []}.items())


def retrieve_node_subontology(entities_uri: str) -> str:
    import rdflib

    # Load your ontology into an rdflib graph.
    g = rdflib.Graph()
    g.parse(ontology_file)

    # SPARQL query that extracts:
    # - The entity's descriptive information (label, comment, description, alternative labels)
    # - For each triple where the entity (a drug) is connected via one of the selected object properties,
    #   it extracts detailed information about the property (label, comment, description) and also for the object.
    # Define the drug URI as a variable.
    print(entities_uri)

    # SPARQL query that extracts:
    # - The entity's descriptive information (label, comment, description, alternative labels)
    # - For each triple where the entity (a drug) is connected via one of the selected object properties,
    #   it extracts detailed information about the property (label, comment, description) and also for the object.
    sparql_query = f"""
    PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dc:   <http://purl.org/dc/elements/1.1/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT ?entity 
                  (SAMPLE(?entityLabel) AS ?Label) 
                  (SAMPLE(?entityDesc) AS ?Description) 
                  (SAMPLE(?entityComment) AS ?Comment) 
                  (GROUP_CONCAT(DISTINCT ?entityAltLabel; separator=", ") AS ?AltLabels)
                  (GROUP_CONCAT(DISTINCT CONCAT(
                      "Property:(Label: ", STR(COALESCE(?propLabel, "")),
                      " ; Object: Label: ", STR(COALESCE(?objLabel, "")),
                      IF(BOUND(?objAltLabel), CONCAT(", AltLabels: ", STR(?objAltLabel)), ""),
                      ")"
                  ); separator=" || ") AS ?ObjectProperties)
    WHERE {{
      # Use the drug_uri value as a parameter.
      VALUES ?entity {{ <{entities_uri}> }}

      # Descriptive info about the entity.
      OPTIONAL {{ ?entity rdfs:label ?entityLabel . }}
      OPTIONAL {{ ?entity dc:description ?entityDesc . }}
      OPTIONAL {{ ?entity rdfs:comment ?entityComment . }}
      OPTIONAL {{ ?entity skos:altLabel ?entityAltLabel . }}

      # Retrieve object property triples for selected properties.
      OPTIONAL {{
        ?entity ?prop ?obj .
        FILTER(?prop IN (
               <http://purl.obolibrary.org/obo/RO_000243>,
          <http://purl.obolibrary.org/obo/RO_0002434>,
          <http://purl.obolibrary.org/obo/RO_0002437>,
          <http://purl.obolibrary.org/obo/RO_0002302>,
          <http://purl.obolibrary.org/obo/RO_0012001>,
    <http://purl.obolibrary.org/obo/RO_0002407>,
    <http://purl.obolibrary.org/obo/RO_0011002>,
    <http://purl.obolibrary.org/obo/RO_0002264>,
    <http://purl.obolibrary.org/obo/RO_0002325>,
    <http://purl.obolibrary.org/obo/RO_000243>,
    <http://purl.obolibrary.org/obo/RO_HOM0000012>,
    <http://purl.obolibrary.org/obo/RO_0018002>,
    <http://purl.obolibrary.org/obo/RO_HOM0000008>,
    <http://purl.obolibrary.org/obo/RO_0004025>,
    <http://purl.obolibrary.org/obo/RO_HOM0000065>,
    <http://purl.obolibrary.org/obo/RO_0002326>,
    <http://purl.obolibrary.org/obo/RO_HOM0000061>,
    <http://purl.obolibrary.org/obo/ENVO_02000200>,
    <http://purl.obolibrary.org/obo/RO_0002524>,
    <http://purl.obolibrary.org/obo/RO_0002610>,
    <http://purl.obolibrary.org/obo/RO_HOM0000010>,
    <http://purl.obolibrary.org/obo/RO_0002508>,
    <http://purl.obolibrary.org/obo/RO_0002327>,
    <http://purl.obolibrary.org/obo/RO_0004024>,
    <http://purl.obolibrary.org/obo/RO_0002529>,
    <http://purl.obolibrary.org/obo/GO_0008150>,
    <http://purl.obolibrary.org/obo/RO_0002206>,
    <http://purl.obolibrary.org/obo/RO_0002505>,
    <http://purl.obolibrary.org/obo/OAE_0001000>,
    <http://purl.obolibrary.org/obo/RO_0002224>,
    <http://purl.obolibrary.org/obo/RO_0002452>,
    <http://purl.obolibrary.org/obo/IAO_0000428>,
    <http://purl.obolibrary.org/obo/RO_0004012>,
    <http://purl.obolibrary.org/obo/RO_0017004>,
    <http://purl.obolibrary.org/obo/RO_0004048>,
    <http://purl.obolibrary.org/obo/RO_0002386>,
    <http://purl.obolibrary.org/obo/RO_0002500>,
    <http://purl.obolibrary.org/obo/RO_0012002>,
    <http://purl.obolibrary.org/obo/RO_0002409>,
    <http://purl.obolibrary.org/obo/RO_0002437>,
    <http://purl.obolibrary.org/obo/RO_0002434>,
    <http://purl.obolibrary.org/obo/RO_0002331>,
    <http://purl.obolibrary.org/obo/RO_HOM0000004>,
    <http://purl.obolibrary.org/obo/RO_0002470>,
    <http://purl.obolibrary.org/obo/RO_0002586>,
    <http://purl.obolibrary.org/obo/RO_0002494>,
    <http://purl.obolibrary.org/obo/RO_0012007>,
    <http://purl.obolibrary.org/obo/RO_0001025>,
    <http://purl.obolibrary.org/obo/RO_0017005>,
    <http://purl.obolibrary.org/obo/RO_0002447>,
    <http://purl.obolibrary.org/obo/RO_0003307>,
    <http://purl.obolibrary.org/obo/RO_0002211>,
    <http://purl.obolibrary.org/obo/RO_HOM0000000>,
    <http://purl.obolibrary.org/obo/RO_0002285>,
    <http://purl.obolibrary.org/obo/RO_0018037>,
    <http://purl.obolibrary.org/obo/BFO_0000050>,
    <http://purl.obolibrary.org/obo/RO_0002400>,
    <http://purl.obolibrary.org/obo/RO_0002134>,
    <http://purl.obolibrary.org/obo/RO_0002302>
        ))
        OPTIONAL {{ ?prop rdfs:label ?propLabel . }}
        OPTIONAL {{ ?prop rdfs:comment ?propComment . }}
        OPTIONAL {{ ?prop dc:description ?propDesc . }}

        OPTIONAL {{ ?obj rdfs:label ?objLabel . }}
        OPTIONAL {{ ?obj rdfs:comment ?objComment . }}
        OPTIONAL {{ ?obj dc:description ?objDesc . }}
        OPTIONAL {{ ?obj skos:altLabel ?objAltLabel . }}
      }}
    }}
    GROUP BY ?entity
    """

    # Execute the query.
    results = g.query(sparql_query)

    # Helper function: safely converts a value to a string (or returns an empty string if not bound).
    def safe_str(val):
        return str(val) if val is not None else ""

    # Create an empty list to hold the aggregated results.
    results_list = []

    # Iterate over the SPARQL query results and aggregate them into the list.
    for row in results:
        row_dict = {
            "entity": safe_str(row.entity),
            "Label": safe_str(row.Label),
            "Description": safe_str(row.Description),
            "Comment": safe_str(row.Comment),
            "AltLabels": safe_str(row.AltLabels),
            "ObjectProperties": safe_str(row.ObjectProperties)
        }
        results_list.append(row_dict)
    return(str(results_list))



# Tool 1: Retrieve Subgraph Context


def retrieve_subgraph_context(question: str) -> str:
    # Run G-Retriever Inference
    with torch.no_grad():
        subgraph_context = g_retriever.inference(
            question=[question],
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            max_out_tokens=1024
        )
    return subgraph_context


# Tool 2: Perform Math Calculation
@tool
def math_tool(expression: str) -> str:
    """Solve a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}."
    except Exception as e:
        return f"Error: {e}"


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline

#meta-llama/Llama-2-7b-chat-hf
#BioMistral/BioMistral-7B
# Load model and tokenizer
model_name = "BioMistral/BioMistral-7B"
device_map = 'auto'

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_cache=True,
    device_map=device_map
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Create generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.6,
    top_p=0.9,
    do_sample=True
)

# Wrap with LangChain
llm_chain = HuggingFacePipeline(pipeline=pipe)
"""


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

# Load model and tokenizer
model_name = "BioMistral-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically allocate to GPU if available
    torch_dtype="auto",  # Use appropriate precision
    load_in_8bit=True  # Optional: Load model in 8-bit precision to save memory
)

# Define HuggingFace pipeline with text generation
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"

)

# Wrap pipeline for LangChain compatibility
llm_chain = HuggingFacePipeline(pipeline=text_generation_pipeline, model_kwargs={"temperature":0.0, "max_new_token":40000})
"""
def local_determine_interaction(source_drug, target_protein):
    """
    Determine whether a given drug and target interact based on graph embeddings and hierarchical analysis.
    """
    try:
        # Retrieve embeddings for the source and target
        source_embedding = node_embeddings[id_to_index[name_to_id[source_drug]]]
        # print(source_embedding)
        target_embedding = node_embeddings[id_to_index[name_to_id[target_protein]]]
        # print(target_embedding)

        # Compute similarity score (e.g., cosine similarity or a learned classifier)
        similarity_score = torch.nn.functional.cosine_similarity(
            source_embedding.unsqueeze(0),
            target_embedding.unsqueeze(0)
        ).item()

        # Convert similarity score to probability (example using sigmoid)
        probability = torch.sigmoid(torch.tensor(similarity_score)).item()

        # Define threshold for interaction
        interaction_threshold = 0.5
        interaction_result = "Yes" if probability >= interaction_threshold else "No"
        print(f"Interaction: {interaction_result} (Probability: {probability:.2f})")

        return f"Interaction: {interaction_result} (Probability: {probability:.2f})"

    except Exception as e:
        return f"Error determining interaction: {str(e)}"


# Tool: Determine Drug-Target Interaction
@tool
def determine_interaction(source_drug: str, target_protein: str) -> str:
    """
    Determine whether a given drug and target interact based on graph embeddings and hierarchical analysis.
    """
    try:
        # Retrieve embeddings for the source and target
        source_embedding = node_embeddings[id_to_index[source_drug]]
        target_embedding = node_embeddings[id_to_index[target_protein]]

        # Compute similarity score (e.g., cosine similarity or a learned classifier)
        similarity_score = torch.nn.functional.cosine_similarity(
            source_embedding.unsqueeze(0),
            target_embedding.unsqueeze(0)
        ).item()

        # Convert similarity score to probability (example using sigmoid)
        probability = torch.sigmoid(torch.tensor(similarity_score)).item()

        # Define threshold for interaction
        interaction_threshold = 0.5
        interaction_result = "Yes" if probability >= interaction_threshold else "No"
        print(f"Interaction: {interaction_result} (Probability: {probability:.2f})")

        return f"Interaction: {interaction_result} (Probability: {probability:.2f})"

    except Exception as e:
        return f"Error determining interaction: {str(e)}"


#
# Integrate with ReAct Agent
tools = [
    Tool(
        name="DetermineInteraction",
        func=lambda input_dict: determine_interaction(input_dict['source_drug'], input_dict['target_protein']),
        description="Determine whether a drug and target interact based on embeddings and hierarchical relationships."
    ),
    Tool(
        name="RetrieveSubgraph",
        func=retrieve_subgraph_context,
        description="Retrieve relevant graph sub-context for a query."
    ),
    Tool(
        name="BioSearchTool",
        func=biosearch_tool,
        description="Search biological information about genes, proteins, or drugs."
    )
]
# Setting a memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# ReAct Agent Initialization
agent = initialize_agent(
    tools=tools,
    llm=llm_chain,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parse_error=True,
    memory=memory,
    max_iterations=6
)


template = "Context: {context}\n\nQuestion: {question}\nAnswer:"
prompt = PromptTemplate(input_variables=["context", "question"], template=template)


#########################################LangGraph#################################
class State(TypedDict):
    source_drug: str
    target_protein: str
    subgraph_context: str
    subonto_drug:str
    subonto_protein: str
    verif_interaction_protein_drug_from_custom_onto:str
    interaction: str
    interactionOnto:str
    biosearch_info: List[str]
    summary: str




def retrieve_subgraph_node(state: State):
    ''' Retrieve relevant subgraph context for the query. '''
    prompt = PromptTemplate(
        input_variables=["source_drug", "target_protein"],
        template=(
            "Retrieve the relevant subgraph context for the interaction between the drug {source_drug} "
            "and the protein {target_protein}. Return a detailed explanation."
        ),
    )
    message = HumanMessage(content=prompt.format(
        source_drug=state["source_drug"], target_protein=state["target_protein"]
    ))
    subgraph_context = retrieve_subgraph_context([state["source_drug"],state["target_protein"] ])
    return {"subgraph_context": str(subgraph_context)}
def lookup_term(label, ontology):
    """
    Lookup an ontology term using the EBI OLS API and return the complete IRI.
    """
    url = "https://www.ebi.ac.uk/ols/api/search"
    params = {
        "q": label,
        "ontology": ontology,
        "exact": "true"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data["response"]["numFound"] > 0:
            result = data["response"]["docs"][0]
            iri = result.get("iri")
            if iri:
                return iri
            else:
                obo_id = result.get("obo_id", label)
                return f"http://purl.obolibrary.org/obo/{obo_id.replace(':', '_')}"
        else:
            return label
    except Exception as e:
        print(f"Error looking up '{label}' in ontology '{ontology}': {e}")
        return label

def searchinformation_aboutdrug_node(state: State):
    ''' Determine if a drug and protein interact based on embeddings and relationships. '''
    iri_drug = lookup_term(str(state["source_drug"]), "chebi")
    try:
        res = retrieve_node_subontology(iri_drug)
        if len(res) == 2:
            res = get_annotations_from_iri(iri_drug, "chebi")
        return {"subonto_drug": res}
    except:
        res = get_annotations_from_iri(iri_drug, "chebi")
        return {"subonto_drug": res}





def searchinformation_aboutprotein_node(state: State):
    ''' Determine if a drug and protein interact based on embeddings and relationships. '''
    iri_protein = lookup_term(str(state["target_protein"]), "pr")
    res = []
    try:
        res= retrieve_node_subontology(iri_protein)
        print("res",res)
        print("len(res)", len(res))
        if len(res)==2:
            res= get_annotations_from_iri(iri_protein, "pr")
            print("line 1000")
            return {"subonto_protein": res}
        else:
            print("line 1002")
            return {"subonto_protein": res}
    except:
        res = get_annotations_from_iri(iri_protein, "pr")
        print("line 1006")
        return {"subonto_protein": res}


def ask_interaction_based_on_ontologies_node(state: State):
    ''' Determine if a drug and protein interact based on embeddings and relationships. '''
    iri_drug = lookup_term(str(state["source_drug"]), "chebi")
    iri_protein = lookup_term(str(state["target_protein"]), "pr")
    res=ask_ontology_DT(iri_drug, iri_protein)
    return {"interactionOnto": res}

def determine_interaction_node_based_only_on_gretriever(state: State):
    ''' Determine if a drug and protein interact based on embeddings and relationships. '''
    prompt = PromptTemplate(
        input_variables=["subgraph_context", "source_drug", "target_protein"],
        template=(
            "Based on the retrieved subgraph context, determine if the drug {source_drug} and "
            "the protein {target_protein} interact. Provide a rationale. Context: {subgraph_context}"
        ),
    )
    message = HumanMessage(content=prompt.format(
        subgraph_context=state["subgraph_context"],
        source_drug=state["source_drug"],
        target_protein=state["target_protein"]
    ))
    interaction = llm_chain.invoke([message])
    return {"interaction": interaction}


def determine_interaction_node_based_only_on_onto_and_gretriever_without_interactiononto(state: State):
    ''' Determine if a drug and protein interact based on ontology-derived evidence and subgraph context. '''
    prompt = PromptTemplate(
        input_variables=[
            "source_drug",
            "target_protein",
            "subonto_drug",
            "subonto_protein",
            "subgraph_context"
        ],
        template=(
            "Detailed ontology information for the drug is provided as follows: {subonto_drug} "
            "(including label, description, and object properties). "
            "Similarly, detailed ontology information for the protein is provided as: {subonto_protein} "
            "(including label, description, and object properties). "
            "Additionally, the following subgraph context has been retrieved: {subgraph_context}. "
            "Based on these factors, please determine if the drug {source_drug} and the protein {target_protein} interact. "
            "Respond with 'Yes' or 'No' and provide a brief, clear rationale explaining how the ontology evidence supports your conclusion."
        )
    )
    message = HumanMessage(content=prompt.format(
        source_drug=state["source_drug"],
        target_protein=state["target_protein"],
        subonto_drug=state["subonto_drug"],
        subonto_protein=state["subonto_protein"],
        subgraph_context=state["subgraph_context"]
    ))
    try:
        interaction = llm_chain.invoke([message])
    except Exception as e:
        truncated_text = message.content[:4000] + "..."
        interaction = llm_chain.invoke([HumanMessage(content=truncated_text)])

    return {"interaction": interaction}


def determine_interaction_node_based_on_onto_and_gretriever(state: State):
    ''' Determine if a drug and protein interact based on ontology-derived evidence. '''
    prompt = PromptTemplate(
        input_variables=[
            "interactionOnto",
            "source_drug",
            "target_protein",
            "subonto_drug",
            "subonto_protein",
            "subgraph_context"
        ],
        template=(
            "If any part of 'interactionOnto' contains the word 'True', please respond with 'Yes', which indicates that there is an interaction between the drug and the target as identified by querying ontologies or databases (i.e., DrugBank, KEGG, or ChEMBL). "
            "The ontology interaction query returned the following information: {interactionOnto}. "
            "Detailed ontology information for the drug is provided as follows: {subonto_drug} "
            "(including label, description, and object properties). "
            "Similarly, detailed ontology information for the protein is provided as: {subonto_protein} "
            "(including label, description, and object properties). "
            "Based on these factors, please determine if the drug {source_drug} and the protein {target_protein} interact. "
            "Otherwise, analyze the ontology information to determine the interaction and respond with 'Yes' or 'No' accordingly. "
            "Provide a brief and clear rationale that explains how the ontology evidence supports your conclusion."
        ),

    )
    message = HumanMessage(content=prompt.format(
        interactionOnto=state["interactionOnto"],
        source_drug=state["source_drug"],
        target_protein=state["target_protein"],
        subonto_drug=state["subonto_drug"],
        subonto_protein=state["subonto_protein"],
        subgraph_context=state["subgraph_context"]
    ))
    interaction = llm_chain.invoke([message])
    return {"interaction": interaction}

def determine_interaction_node_based_only_on_onto(state: State):
    ''' Determine if a drug and protein interact based on ontology-derived evidence (without using subgraph_context). '''
    prompt = PromptTemplate(
        input_variables=[
            "interactionOnto",
            "source_drug",
            "target_protein",
            "subonto_drug",
            "subonto_protein"
        ],
        template=(
            "The ontology interaction query returned the following information: {interactionOnto}. "
            "Detailed ontology information for the drug is provided as follows: {subonto_drug} "
            "(including label, description, and object properties). "
            "Similarly, detailed ontology information for the protein is provided as: {subonto_protein} "
            "(including label, description, and object properties). "
            "Based on these factors, please determine if the drug {source_drug} and the protein {target_protein} interact. "
            "If any part of 'interactionOnto' contains the word 'True', respond with 'Yes'. "
            "Otherwise, analyze the ontology information to determine the interaction and respond with 'Yes' or 'No' accordingly. "
            "Provide a clear rationale that explains how the ontology evidence supports your conclusion."
        )
    )
    message = HumanMessage(content=prompt.format(
        interactionOnto=state["interactionOnto"],
        source_drug=state["source_drug"],
        target_protein=state["target_protein"],
        subonto_drug=state["subonto_drug"],
        subonto_protein=state["subonto_protein"]
    ))
    try:
        interaction = llm_chain.invoke([message])
    except Exception as e:
        truncated_text = message.content[:4000] + "..."
        interaction = llm_chain.invoke([HumanMessage(content=truncated_text)])

    return {"interaction": interaction}




def biosearch_tool_node(state: State):
    ''' Search for biological information related to the drug and protein. '''
    prompt = PromptTemplate(
        input_variables=["source_drug", "target_protein"],
        template=(
            "Search biological databases and retrieve detailed information about the drug {source_drug} "
            "and the protein {target_protein}. Include known interactions and relevant studies."
        ),
    )
    message = HumanMessage(content=prompt.format(
        source_drug=state["source_drug"], target_protein=state["target_protein"]
    ))
    biosearch_info = llm_chain.invoke([message]).split("\n")
    return {"biosearch_info": biosearch_info}
def summarization_node_v3(state: State):
    '''
    Summarize the provided context about the interaction between the drug and protein.
    The response should be a single short sentence that begins with either "Yes" or "No" and does not include any extra explanation.
    '''
    prompt = PromptTemplate(
        input_variables=["interaction", "source_drug", "target_protein"],
        template=(
            "Based on the following context: {interaction}\n\n"
            "Determine if the drug {source_drug} and the protein {target_protein} interact. "
            "Your answer must be a single short sentence containing only the word 'Yes' or 'No' (and optionally a brief rationale that starts with 'Yes' or 'No'). "
            "Please do not include any additional content.\n\nSummary:"
        ),
    )
    message = HumanMessage(content=prompt.format(
        interaction=state["interaction"],
        source_drug=state["source_drug"],
        target_protein=state["target_protein"]
    ))
    summary = llm_chain.invoke([message]).strip()
    return {"summary": summary}


def summarization_node_v2(state: State):
    ''' Summarize the text in one short sentence using the Word "Yes" or "No"'''

    prompt = PromptTemplate(
        input_variables=["interaction", "source_drug", "target_protein"],
        template=(
            "Based on the retrieved information from this context: {interaction}. Determine if the drug {source_drug} and "
            "the protein {target_protein} interact. Summarize the information about the drug {source_drug} and the protein {target_protein} in one short sentence, and please tell me if there is an interaction between them using the word Yes or No.\n\nSummary:"
        ),
    )
    message = HumanMessage(content=prompt.format(
        interaction=state["interaction"],
        source_drug=state["source_drug"],
        target_protein=state["target_protein"]
    ))
    summary = llm_chain.invoke([message]).strip()
    return {"summary": summary}

def summarization_node(state: State):
    ''' Summarize the text in one short sentence using the Word "Yes" or "No"'''

    prompt = PromptTemplate(
        input_variables=["subgraph_context", "source_drug", "target_protein"],
        template=(
            "Determine if the drug {source_drug} and the protein {target_protein} interact. "
            "Additionally, we have the following subgraph context: {subgraph_context}.\n\n"
            "Your answer must be a single short sentence containing only the word 'Yes' or 'No' (and optionally a brief rationale that starts with 'Yes' or 'No'). "
            "Please do not include any additional content.\n\nSummary:"
        ),
    )
    message = HumanMessage(content=prompt.format(
        subgraph_context=state["subgraph_context"],
        source_drug=state["source_drug"],
        target_protein=state["target_protein"]
    ))
    summary = llm_chain.invoke([message]).strip()
    return {"summary": summary}

def summarization_node_ontology(state: State):
    ''' Summarize based on ontology-derived evidence by indicating with "Yes" or "No" if the drug and protein interact. '''
    prompt = PromptTemplate(
        input_variables=["interactionOnto", "source_drug", "target_protein", "subonto_drug", "subonto_protein"],
        template=(
            "The ontology interaction query returned a boolean value: {interactionOnto}. "
            "Detailed ontology information for the drug is provided as follows: {subonto_drug} "
            "(including labels, descriptions, and object properties). "
            "Similarly, detailed ontology information for the protein is provided as: {subonto_protein} "
            "(including labels, descriptions, and object properties). "
            "Based on this ontology evidence, determine if the drug {source_drug} and the protein {target_protein} interact. "
            "Summarize your findings in one short sentence using 'Yes' or 'No'.\n\nSummary:"
        ),
    )
    message = HumanMessage(content=prompt.format(
        interactionOnto=state["interactionOnto"],
        source_drug=state["source_drug"],
        target_protein=state["target_protein"],
        subonto_drug=state["subonto_drug"],
        subonto_protein=state["subonto_protein"]
    ))
    summary = llm_chain.invoke([message]).strip()
    return {"summary": summary}

def summarization_node_v4(state: State):
    '''
    Summarize the provided context about the interaction between the drug and protein.
    The response should be a single short sentence that begins with either "Yes" or "No" and does not include any extra explanation.
    '''
    prompt = PromptTemplate(
        input_variables=["interaction", "source_drug", "target_protein", "subgraph_context"],
        template=(
            "Based on the following context: {interaction}\n\n"
            "Determine if the drug {source_drug} and the protein {target_protein} interact. "
            "Additionally, we have the following subgraph context: {subgraph_context}.\n\n"
            "Your answer must be a single short sentence containing only the word 'Yes' or 'No' (and optionally a brief rationale that starts with 'Yes' or 'No'). "
            "Please do not include any additional content.\n\nSummary:"
        ),
    )
    message = HumanMessage(content=prompt.format(
        interaction=state["interaction"],
        source_drug=state["source_drug"],
        target_protein=state["target_protein"],
    subgraph_context = state["subgraph_context"]
    ))
    try:
        summary = llm_chain.invoke([message]).strip()
    except Exception as e:
        truncated_text = message.content[:4000] + "..."
        summary = llm_chain.invoke([HumanMessage(content=truncated_text)])
    return {"summary": summary}
def summarization_node_onto_gt(state: State):
    ''' Summarize the evidence using both ontology-derived information and subgraph context. '''
    prompt = PromptTemplate(
        input_variables=[
            "interactionOnto",
            "source_drug",
            "target_protein",
            "subonto_drug",
            "subonto_protein",
            "subgraph_context"
        ],
        template=(
            "The ontology interaction query returned a boolean value: {interactionOnto}.\n"
            "Detailed ontology information for the drug is provided as: {subonto_drug} (including labels, "
            "descriptions, and object properties), and for the protein as: {subonto_protein} (including labels, "
            "descriptions, and object properties).\n"
            "Additionally, we have the following subgraph context: {subgraph_context}.\n\n"
            "Based on this combined evidence, determine if the drug {source_drug} interacts with the protein {target_protein}. "
            "Summarize your findings in one short sentence using either 'Yes' or 'No', and provide a brief rationale."
        )
    )
    message = HumanMessage(content=prompt.format(
        interactionOnto=state["interactionOnto"],
        source_drug=state["source_drug"],
        target_protein=state["target_protein"],
        subonto_drug=state["subonto_drug"],
        subonto_protein=state["subonto_protein"],
        subgraph_context=state["subgraph_context"]
    ))
    summary = llm_chain.invoke([message]).strip()
    return {"summary": summary}


def summarization_node_onto(state: State):
    ''' Summarize the evidence using both ontology-derived information and subgraph context. '''
    prompt = PromptTemplate(
        input_variables=[
            "interaction",
            "source_drug",
            "target_protein",
            "subonto_drug",
            "subonto_protein",
        ],
        template=(
             "Based on the following context: {interaction}\n\n"
            "Detailed ontology information for the drug is provided as: {subonto_drug} (including labels, "
            "descriptions, and object properties), and for the protein as: {subonto_protein} (including labels, "
            "descriptions, and object properties).\n"
            "Based on this combined evidence, determine if the drug {source_drug} interacts with the protein {target_protein}. "
            "Summarize your findings in one short sentence using either 'Yes' or 'No', and provide a brief rationale."
        )
    )
    message = HumanMessage(content=prompt.format(
        interaction=state["interaction"],
        source_drug=state["source_drug"],
        target_protein=state["target_protein"],
        subonto_drug=state["subonto_drug"],
        subonto_protein=state["subonto_protein"],
    ))
    summary = llm_chain.invoke([message]).strip()
    return {"summary": summary}


# Define the workflow graph
workflow = StateGraph(State)

#####################Workflow_based_only_on_gretriver##########################


if (strategies_workflow=="Simple"):
    workflow.add_node("retrieve_subgraph", retrieve_subgraph_node)
    workflow.add_node("determine_interaction", determine_interaction_node_based_only_on_gretriever)
    workflow.add_node("biosearch_tool", biosearch_tool_node)
    workflow.add_node("summarization", summarization_node)
    workflow.set_entry_point("retrieve_subgraph")  # Entry point of the graph
    workflow.add_edge("retrieve_subgraph", "determine_interaction")
    workflow.add_edge("determine_interaction", "biosearch_tool")
    workflow.add_edge("biosearch_tool", "summarization")
    workflow.add_edge("summarization", END)  # Mark the end of the workflow
elif(strategies_workflow=="GT+ONTO"):
    ################################################End Complete workflow##########################################
    #####################Workflow_based_only_on_gretriver and ontologies##########################

    workflow.add_node("retrieve_subgraph", retrieve_subgraph_node)
    workflow.add_node("searchinformation_aboutprotein_node", searchinformation_aboutprotein_node)
    workflow.add_node("searchinformation_aboutdrug_node", searchinformation_aboutdrug_node)
    workflow.add_node("ask_interaction_based_on_ontologies_node", ask_interaction_based_on_ontologies_node)
    workflow.add_node("determine_interaction_node_based_only_on_onto_and_gretriever_without_interactiononto",
                      determine_interaction_node_based_only_on_onto_and_gretriever_without_interactiononto)

    workflow.add_node("biosearch_tool", biosearch_tool_node)
    workflow.add_node("summarization_node_v4", summarization_node_v4)

    workflow.set_entry_point("retrieve_subgraph")  # Entry point of the graph
    workflow.add_edge("retrieve_subgraph", "searchinformation_aboutprotein_node")
    workflow.add_edge("searchinformation_aboutprotein_node", "searchinformation_aboutdrug_node")
    workflow.add_edge("searchinformation_aboutdrug_node", "determine_interaction_node_based_only_on_onto_and_gretriever_without_interactiononto")
    workflow.add_edge("determine_interaction_node_based_only_on_onto_and_gretriever_without_interactiononto", "biosearch_tool")
    workflow.add_edge("biosearch_tool", "summarization_node_v4")
    workflow.add_edge("summarization_node_v4", END)  # Mark the end of the workflow
    ################################################End Complete workflow##########################################
elif(strategies_workflow=="ONTO"):
    workflow.add_node("searchinformation_aboutprotein_node", searchinformation_aboutprotein_node)
    workflow.add_node("searchinformation_aboutdrug_node", searchinformation_aboutdrug_node)
    workflow.add_node("ask_interaction_based_on_ontologies_node", ask_interaction_based_on_ontologies_node)
    workflow.add_node("determine_interaction_node_based_only_on_onto",
                      determine_interaction_node_based_only_on_onto)

    workflow.add_node("biosearch_tool", biosearch_tool_node)
    workflow.add_node("summarization_node_onto", summarization_node_onto)

    workflow.set_entry_point("searchinformation_aboutprotein_node")  # Entry point of the graph
    workflow.add_edge("searchinformation_aboutprotein_node", "searchinformation_aboutdrug_node")
    workflow.add_edge("searchinformation_aboutdrug_node", "ask_interaction_based_on_ontologies_node")
    workflow.add_edge("ask_interaction_based_on_ontologies_node",
                      "determine_interaction_node_based_only_on_onto")
    workflow.add_edge("determine_interaction_node_based_only_on_onto", "biosearch_tool")
    workflow.add_edge("biosearch_tool", "summarization_node_onto")
    workflow.add_edge("summarization_node_onto", END)  # Mark the end of the workflow
elif(strategies_workflow=="ONTO_light"):
    workflow.add_node("searchinformation_aboutprotein_node", searchinformation_aboutprotein_node)
    workflow.add_node("searchinformation_aboutdrug_node", searchinformation_aboutdrug_node)
    workflow.add_node("ask_interaction_based_on_ontologies_node", ask_interaction_based_on_ontologies_node)
    workflow.add_node("determine_interaction_node_based_only_on_onto",
                      determine_interaction_node_based_only_on_onto)
    workflow.add_node("summarization_node_v3", summarization_node_v3)
    workflow.set_entry_point("searchinformation_aboutprotein_node")  # Entry point of the graph
    workflow.add_edge("searchinformation_aboutprotein_node", "searchinformation_aboutdrug_node")
    workflow.add_edge("searchinformation_aboutdrug_node", "ask_interaction_based_on_ontologies_node")
    workflow.add_edge("ask_interaction_based_on_ontologies_node",
                      "determine_interaction_node_based_only_on_onto")
    workflow.add_edge("determine_interaction_node_based_only_on_onto", "summarization_node_v3")
    workflow.add_edge("summarization_node_v3", END)  # Mark the end of the workflow




# Compile the workflow
app = workflow.compile()

import os
import json
from sklearn.metrics import precision_recall_fscore_support


# File paths
output_dir = "summaries"
os.makedirs(output_dir, exist_ok=True)

# Parse the bipartite graph file
true_labels = []
predicted_labels = []
summary_file = os.path.join(output_dir, "DTI-PREDICTION.txt")
explainability_file = os.path.join(output_dir, "explainability.txt")
with open(summary_file, "w") as summary_f, open(explainability_file, "w") as explainability_f:
    with open(input_file, "r") as file:
        for line in file:
            drug, target, interaction = line.strip().split("\t")
            interaction = int(float(interaction))
            true_labels.append(interaction)
            # Query the LLM
            state_input = {
                "source_drug": drug,
                "target_protein": target,
            }
            result = app.invoke(state_input)
            try:
                interaction = "Yes" if "Yes" in str(result["summary"].split("Summary:")[1]) else "No"
            except:
                continue

            # Save the explainability data
            explainability_f.write(f"======================================================================\n")
            if(strategies_workflow=="Simple"):
                explainability_f.write(f"Subgraph Context: {result['subgraph_context']}\n")
                explainability_f.write(f"\nInteraction Determination: {result['interaction']}\n")
                explainability_f.write(f"\nBiological Information: {result['biosearch_info']}\n\n")
                explainability_f.write(f"\nsummarization: {result['summary']}\n\n")
            elif(strategies_workflow == "GT+ONTO"):
                explainability_f.write(f"Subgraph Context: {result['subgraph_context']}\n")
                explainability_f.write(f"\nInteraction Determination: {result['interaction']}\n")
                explainability_f.write(f"\nsubonto_drug: {result['subonto_drug']}\n\n")
                explainability_f.write(f"\nsubonto_protein: {result['subonto_protein']}\n\n")
                #explainability_f.write(f"\ninteractionOnto: {result['interactionOnto']}\n\n")
                explainability_f.write(f"\nBiological Information: {result['biosearch_info']}\n\n")
                explainability_f.write(f"\nsummarization: {result['summary']}\n\n")
            elif (strategies_workflow == "ONTO"):
                explainability_f.write(f"\nInteraction Determination: {result['interaction']}\n")
                explainability_f.write(f"\nsubonto_drug: {result['subonto_drug']}\n\n")
                explainability_f.write(f"\nsubonto_protein: {result['subonto_protein']}\n\n")
                #explainability_f.write(f"\ninteractionOnto: {result['interactionOnto']}\n\n")
                explainability_f.write(f"\nBiological Information: {result['biosearch_info']}\n\n")
                explainability_f.write(f"\nsummarization: {result['summary']}\n\n")
            elif (strategies_workflow == "ONTO_light"):
                explainability_f.write(f"\n*Interaction Determination*: {result['interaction']}\n")
                explainability_f.write(f"\n*subonto_drug*: {result['subonto_drug']}\n\n")
                explainability_f.write(f"\n*subonto_protein*: {result['subonto_protein']}\n\n")
                #explainability_f.write(f"\n*interactionOnto*: {result['interactionOnto']}\n\n")
                explainability_f.write(f"\n*summarization*: {result['summary']}\n\n")

            # Save the summary
            print(f"{drug}\t{target}\t{1 if interaction.lower() == 'yes' else 0}\n")
            stringres = f"{drug}\t{target}\t{1 if interaction.lower() == 'yes' else 0}\n"
            summary_f.write(stringres)
            explainability_f.write(f"\n---> Final results: {stringres}\n\n")
            predicted_labels.append(1 if interaction.lower() == "yes" else 0)

# Calculate precision, recall, and F-measure
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="binary")

# Display results
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F-measure: {f1:.4f}")

# import networkx as nx
# import matplotlib.pyplot as plt
#
# # Create a directed graph
# G = nx.DiGraph()
#
# # Add nodes
# G.add_node("Start", label="Start")
# G.add_node("RetrieveSubgraph", label="Retrieve Subgraph Context")
# G.add_node("BioSearchTool", label="BioSearch Tool")
# G.add_node("Agent", label="Agent Node")
# G.add_node("End", label="End")
#
# # Add edges
# G.add_edges_from([
#     ("Start", "RetrieveSubgraph"),
#     ("Start", "BioSearchTool"),
#     ("RetrieveSubgraph", "Agent"),
#     ("BioSearchTool", "Agent"),
#     ("Agent", "End"),
# ])
#
# # Draw the graph
# pos = nx.spring_layout(G)
# plt.figure(figsize=(10, 8))
# nx.draw(
#     G, pos, with_labels=True, node_color='lightgreen', node_size=3000,
#     font_size=10, font_color='black', edge_color='blue', arrowsize=20
# )
# plt.title("Workflow Visualization", fontsize=14)
# plt.savefig("pipeline_workflow_start_end.png")
# plt.show()
#
# print("Pipeline saved as pipeline_workflow_start_end.png")
#
#
# print("Final Result:")
#
#
