from graph_maker import GraphMaker, Ontology, GroqClient
from graph_maker import Document
import pdb




ontology = Ontology(
    labels=[
        {"Drug": "Represents medications, drugs, or pharmaceutical substances used for treatment."},
        {"Protein": "Represents biological macromolecules consisting of amino acid chains that perform various functions in organisms."},
        {"Disease": "Represents medical conditions or disorders affecting organisms."},  
        {"Gene": "Represents a unit of heredity in organisms, responsible for specific traits or biological functions."},
        {"Pathway": "Represents a series of actions among molecules that lead to a specific product or change in a cell."},
        {"Side Effect": "Represents unintended or adverse effects of a medication or treatment."},
        {"Symptom": "Represents an observable or subjective indication of a disease or medical condition."},
        {"Anatomy": "Represents parts or structures of an organism's body."},
        {"Cellular Component": "Represents a part or structure within a cell where a gene product is active."},
    ],
    relationships=[
        "General relationships represent interactions or associations between Drugs, Proteins, Diseases, Genes, Side Effects, Symptoms, Anatomy, and Cellular Components, encompassing Binds, Interacts, Inhibits, Activates, Regulates, Expressed in, Catalyzes, Catalyzed by, Involved in, Located in, Modulates, Treats, Causes, Repurposed, Affects, Induces, Requires, Enhances, Suppresses, Increases, Decreases, Detected in, Involves, Has side effect, Binds to, Inhibitor of, Substrate of, Cofactor of, Upregulates, Downregulates, and Metabolized by.",
    ],
)






from abstract_chunks import Abstracts as example_text_list
from langchain_openai import ChatOpenAI

len(example_text_list)

#model = "mixtral-8x7b-32768"
#model ="llama3-8b-8192"
#model = "llama3-70b-8192"
# model="gemma-7b-it"
model ="deepseek-r1-distill-llama-70b"

import datetime
current_time = str(datetime.datetime.now())
llm = GroqClient(model=model, temperature=0.6, top_p=0.5)
graph_maker = GraphMaker(ontology=ontology, llm_client=llm, verbose=False)

def generate_summary(text):
    SYS_PROMPT = (
        "Succintly summarise the text provided by the user. "
        "Respond only with the summary and no other comments"
    )
    try:
        summary = llm.generate(user_message=text, system_message=SYS_PROMPT)
    except:
        summary = ""
    finally:
        return summary


docs = map(
    lambda t: Document(text=t, metadata={"summary": generate_summary(t), 'generated_at': current_time}),
    example_text_list
)

graph = graph_maker.from_documents(
    list(docs),
    delay_s_between=10 
    )

print("Total number of Edges:", len(graph))

import json

graph_data = []

for edge in graph:
    node_1 = edge.node_1  
    node_2 = edge.node_2  
    
    graph_data.append({
        "node_1": {
            "label": node_1.label,
            "name": node_1.name,
            "source": node_1.source,
            "link": node_1.link
        },
        "node_2": {
            "label": node_2.label,
            "name": node_2.name,
            "source": node_2.source,
            "link": node_2.link
        },
        "relationship": edge.relationship,
        "description": edge.description

    })

with open('Result-ENZ-sentence-21.json', 'w', encoding="utf-8") as file:
    json.dump(graph_data, file, ensure_ascii=False, indent=4)