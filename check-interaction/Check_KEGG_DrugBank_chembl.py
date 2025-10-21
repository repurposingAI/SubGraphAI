import json
import requests
import csv
from rdflib import Graph, Namespace, URIRef, BNode, Literal
from rdflib.namespace import RDF, XSD, RDFS, OWL, SKOS
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from chembl_webresource_client.new_client import new_client  # For ChEMBL
from bioservices import KEGG  # For KEGG


# -------------------------------------------------------------------
# External Interaction Check Functions (unchanged)
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

def check_drugbank_interaction(drug_name, target_protein, dictdtref2):
    if drug_name in dictdtref2:
        targets = dictdtref2[drug_name]
        for trg in targets:
            if trg == target_protein:
                return True
    return False

def check_chembl_interaction(drug_name, target_protein):
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


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
API_KEY = "bd2d2269-59a8-4559-adc5-cd5aa4bcb181"  # Adjust if needed
HEADERS = {"Authorization": "apikey token=" + API_KEY}
file2_path = 'data/drug_target_all.csv'
file_path = 'data/uniprot_data.csv'

drug_name = "Sorafenib"
target_protein = "Vascular endothelial growth factor receptor 1"
dictdtref = {}
dictdtref2 = {}
dictdtref = read_csv_name_to_uniprot_id(file_path, dictdtref)
dictdtref2 = read_csv_and_focus_on_drug_target(file2_path, dictdtref2)
db_valid = check_drugbank_interaction(drug_name, target_protein, dictdtref2)
chembl_valid = check_chembl_interaction(drug_name, target_protein)
nameprot_uniprot = dictdtref.get(target_protein, set())
print("line 120")
if nameprot_uniprot:
    uniprot_id = nameprot_uniprot.pop()
    kegg_valid = check_kegg_interaction(drug_name, uniprot_to_gene_name(uniprot_id))
else:
    kegg_valid = False
if db_valid:
    print("Interaction is reported in DrugBank.")
if chembl_valid:
    print("Interaction is reported in Chembl.")
if kegg_valid:
    print("Interaction is reported in KEGG.")
print("line 132")
