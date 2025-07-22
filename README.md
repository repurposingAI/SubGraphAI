# Ontology‐Driven Graph‐Augmented Language Models for Interpretable Drug-Target Interaction Prediction
## Introduction
This repository contains the code for our proposed approach, which builds a semantic biomedical question-answering system by combining large language models (LLMs), ontology-guided extraction, and knowledge graphs. First, drug and protein keywords are used to retrieve relevant abstracts from PubMed. These abstracts are then processed by a large language model (LLM), guided by a prompt and a biomedical ontology, to extract entities and relationships. The extracted information is used to construct a knowledge graph (KG) in JSON format. Next, relevant subgraphs are retrieved based on embedding similarity and refined using the Prize-Collecting Steiner Tree algorithm to focus on the most informative context. These subgraphs are encoded both structurally (via GAT and MLP) and textually, then integrated as soft prompts into a frozen LLM. The KG is further enriched with ontology metadata from resources such as BioPortal and the Ontology Lookup Service, linking entities to standardized identifiers and preserving provenance. The result of this enrichment process is the Biomedical Generated Ontology (BioGenOnt), which serves as a structured and interoperable foundation for downstream reasoning. Finally, the LLM leverages the enriched subgraph and BioGenOnt context to generate accurate and interpretable predictions for drug-target interaction tasks.

## Requirements

### For Running `prediction-dti.py`

To run the DTI prediction pipeline, please ensure the following Python dependencies are installed:

```
pip install timeout_decorator
pip install langchain
pip install langgraph
pip install python-dotenv
pip install sentence_transformers
pip install torch_geometric
pip install sentencepiece
pip install protobuf
pip install langchain_community
pip install 'accelerate>=0.26.0'
pip install -U bitsandbytes
pip install bioservices
pip install chembl_webresource_client
```

### For Running `extract-kg.py`
To run the knowledge graph extraction pipeline from PubMed abstracts using LLMs, install the following:

```
pip install groq
pip install yachalk
pip install neomodel
```


## Repository Structure

This repository is organized into the following main components:

### Abstract Retrieval from PubMed


The pubmed-extraction.ipynb notebook, located in the pubmed-extraction folder, is responsible for extracting abstracts from PubMed. The retrieval is performed using keyword queries composed of drug and protein names. It enables the automatic collection of relevant biomedical literature for downstream knowledge graph construction. 

### Knowledge Graph Extraction from PubMed Abstracts Using LLMs


The kg-extraction folder contains the codebase for extracting knowledge graphs (KGs) using a large language model (LLM) guided by a structured ontology and custom prompts.

The main script is located in extraction-kg.py, which orchestrates the entity and relation extraction process.

Abstracts must first be segmented and stored in the file Abstract_chunks.py.

Due to input size limitations of the LLM, The code was executed on 200 abstracts per run. As a result, the knowledge graph was generated piece by piece, with Each part was saved individually in the results folder. Finally, all partial graphs were merged to produce the complete knowledge graph, available in JSON format under the llm-biokg directory.

### Drug-Target Interaction Prediction


The prediction folder contains the Python source code and all necessary files for Drug-Target Interaction (DTI) prediction.

The main script, prediction-dti.py, takes as input the generated KG (Located in the llm-biokg folder under the extraction-kg directory, in JSON format) along with the BioGenOnt ontology (located in the ontology folder under dataset directory, in owl format), and uses both to perform Drug-Target Interaction prediction through LLM-based reasoning.

Configuration Files (policy.txt) : 


The file policy.txt defines the paths and configuration settings used to run the DTI prediction pipeline. It specifies the location of all required input files such as knowledge graph elements, embeddings, datasets, and ontology. Here's a breakdown of what each path points to:

```
node_descriptions_file: JSON file containing textual descriptions of each node in the KG.

source_nodes_file: List of source node names involved in the KG edges.

target_nodes_file: List of target node names.

attribut_file: File containing edge attributes associated with each interaction.

drug_embedding_file: File with embedding vectors for drugs.

target_embedding_file: File with embedding vectors for protein targets.

missing_embedding_file: List of entities with missing embeddings.

input_file: Bipartite graph file with drug-target pairs to evaluate.

file2_path: CSV file listing all drug-target pairs used.

file_path: UniProt-based protein annotation file.

ontology_file: The OWL ontology file (custom_ontology-ro.owl) used for semantic reasoning.

validation_dataset: Ground-truth file for evaluating predictions (negative examples).

strategies_workflow: Specifies the prediction strategy (e.g., Simple).
```

### Output Files (Located in the prediction-results Folder)

Our approach generates two key output files (See the prediction-results folder) that provide detailed predictions and explainability for Drug-Target Interactions (DTIs):

1. dti-prediction.txt 

This file contains the final binary predictions for each drug–protein pair. Each line follows the format:

```
<Drug Name>    <Protein Name>    <Interaction (1 = Yes, 0 = No)>
```

2. explainability.txt

This file offers a detailed trace of the reasoning process for each prediction. It includes:

```
Subgraph Context: Textualized form of the relevant subgraph retrieved for the given query, highlighting biological mechanisms (e.g., enzyme pathways).

Ontology Metadata: Information from the BioGenOnt ontology including: Labels, descriptions, synonyms, and other metadata for the selected drug and protein from the dataset, used to predict their interaction.

Biological Context: Additional biological data supporting the prediction (e.g., enzyme functions, known roles).

Interaction Reasoning: Final interaction decision, accompanied by a concise rationale based on ontology and subgraph evidence.
```

This file serves as a transparent explanation of the model’s decision-making process and facilitates interpretability in biomedical applications.

### The process of fine-tuning our large language model using Alpaca

The alpaca directory contains the script alpaca-llama.py, which provides the code for performing Alpaca-style fine-tuning on the model `meta-llama/Llama-2-7b-chat-hf`. This fine-tuning process follows the self-instruct methodology to adapt the base model to biomedical question-answering tasks.


### Installation Guide

To get started with our approach SubGraphAI, follow the steps below. The execution was tested on a uCloud server running Linux and requires at least 10 GB of available RAM.

1. Clone the repository

```
git clone https://github.com/YourUsername/SubGraphAI.git
cd SubGraphAI
```

2. Install Python dependencies

Before running the code, make sure to install all required libraries listed in the Requirements section. Once these are installed, proceed with the main execution.

3. Run the prediction script

After setting up the environment and dependencies, you can directly launch the DTI prediction task using:

```
python3 prediction-dti.py
```

⚠️ Make sure you have added your Groq API key, your Hugging Face access token, and obtained permission to use specific models (e.g., meta-llama/Llama-2-7b-chat-hf, BioMistral/BioMistral-7B). See the section Authentication & Model Access Setup for more details.
## Datasets

The dataset folder within the prediction directory is organized into several subfolders containing the data required for drug-target interaction (DTI) prediction.

### The LLM-BioKG knowledge graph

The folder kg-llm contains the knowledge graph (KG) extracted from PubMed abstracts using an LLM guided by a biomedical ontology and a structured prompt. The LLM-BioKG knowledge graph was adapted to be compatible as input to G-Retriever, which is part of the DTI prediction pipeline, and was therefore saved in four separate JSON files following a specific structure :

```
descriptions.json: Textual descriptions of each KG node.

source.json: Source entities for each edge in the KG.

target.json: Target entities for each edge.

attribut.json: Attributes or relationship types between source and target entities.
```

### The BioGenOnt ontology

This folder ontology includes the file:

custom_ontology-ro.owl: This is the BioGenOnt ontology used for semantic enrichment, including curated biomedical concepts and relationships.

## Authentication & Model Access Setup

To run the Knowledge Graph Extraction script, follow these steps to authenticate and gain access to the required LLMs:

1. ChatGroq API Key (for deepseek-r1-distill-llama-70b, mixtral-8x7b-32768, etc.)
To use models hosted on ChatGroq such as deepseek-coder, mixtral, and others:

Sign up or log in to https://console.groq.com

Navigate to your API settings and generate a ChatGroq API key

In the file groq_client.py, insert your API key in the following line:

```
client = Groq(api_key="your_api_key")
```

2. Hugging Face Token Access (for Meta & BioMistral models)
To access meta-llama/Llama-2-7b-chat-hf and BioMistral/BioMistral-7B:

Create a Hugging Face account at https://huggingface.co

Go to your settings → Access Tokens and generate a token

Important: You must request access to the following models:

meta-llama/Llama-2-7b-chat-hf

BioMistral/BioMistral-7B

Once access is granted, you must insert your Hugging Face token into the file prediction-dti.py (located in the prediction folder) by modifying the following code:

```
from huggingface_hub import login
login("your_token")
```

## Ontology Visualization of Drug and Protein Entities in Protégé

This image shows how the drug (R)-lipoic acid is represented as an individual within the BioGenOnt ontology, loaded in Protégé. The ontology captures rich semantic information through annotations (such as labels, descriptions, and synonyms) and property assertions that link the drug to related biomedical entities :

<p align="center">
<img src="https://github.com/compbiolabucf/DTI-LM/blob/main/overall-1.png" width="450" height="450">
</p>


This image shows how a specific biomedical entity—the protein 6-phosphofructo-2-kinase/fructose-2,6-bisphosphatase 4—is represented as an individual within the BioGenOnt ontology using the Protégé editor. The ontology includes annotations that capture metadata such as labels, descriptions, and synonyms, as well as property assertions that define semantic relationships between this protein and other biomedical concepts:

