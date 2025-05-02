# Automating Mathematical Proof

Official repository for the research paper: "Automating Mathematical Proof Generation Using Large Language Model
Agents and Knowledge Graphs"

## Table of Contents
- [Getting Started](#getting-started)
- [Extracting Proofs to Neo4j AuraDB](#extracting-proofs-to-your-own-neo4j-auradb-optional)
- [Running Tests](#running-tests)

## Getting Started

### Setting up the mathlib repository
1. First download [elan](https://github.com/leanprover/elan)
2. Clone and build the repository:
```bash
git clone --recurse-submodules https://github.com/vinli0921/LLM-proof.git
cd LLM-proof
cd mathlib4
lake build
```

### Create a virtual environment
Create a virtual environment in the root directory of the repository:
```bash
python -m venv venv
```

### Activate the virtual environment

**On macOS and Linux:**
```bash
source myenv/bin/activate
```

**On Windows:**
```bash
myenv\Scripts\activate
```

### Install Dependencies
Install all required packages and dependencies:
```bash
pip install -r requirements.txt
```

### Set Environment Variables
1. Create a `.env` file in the root directory of the repository
2. Copy the content from `.env.example` into the newly created `.env` file
3. Configure the variables with appropriate values

## Extracting Proofs to Your Own Neo4j AuraDB (OPTIONAL)

### Retrieving the XML Dump
1. Download the latest XML dump from: https://proofwiki.org/xmldump/latest.xml.gz
2. Extract the file to obtain `latest.xml`
3. Move `latest.xml` to the root directory of the repository

### Extracting to Nodes and Relationships from the XML file to CSV files
Run the extraction script:
```bash
python Graph_Creation/extract_proofs_XML.py latest.xml
```

### Uploading CSV files to Neo4j AuraDB
Run the database upload script:
```bash
python neo4j_kg.py
```
This should successfully upload approximately 60,000 nodes and 300,000 relationships.

### Building the Graph from Scratch (Optional)
If you're creating the graph from scratch:

1. Delete the existing graph (optional):

   ```bash
   python Knowledge_Graph/neo4j_delete.py
   ```

2. Initialize constraints and load data in Python:

   ```python
   from Knowledge_Graph.neo4j_kg import gds, create_constraints, load_nodes, load_relationships, append_embeddings

   with gds.session() as session:
       create_constraints(session)
       load_nodes(session, 'nodes.csv')
       load_relationships(session, 'relationships.csv')
       append_embeddings(session, 'embeddings.csv')
   ```

### Generating Embeddings (Optional)
There are two ways to create node embeddings:

- From the `nodes.csv` file:

   ```python
   from Graph_Creation.embedding_chunks import generate_embeddings
   generate_embeddings('nodes.csv', 'embeddings.csv')
   ```

- From Neo4j:

   ```python
   from Graph_Creation.embedding_chunks import generate_neo4j_embeddings
   generate_neo4j_embeddings('embeddings.csv')
   ```

After generating embeddings with either method, upload them in Python:

```python
from Knowledge_Graph.neo4j_kg import gds, append_embeddings

with gds.session() as session:
    append_embeddings(session, 'embeddings.csv')
```

### Creating a Vector Index (Optional)
Create a vector index for fast cosine similarity in Python:

```python
from Graph_Creation.vector_functions import gds, create_vector_index

with gds.session() as session:
    create_vector_index(session)
```

## Running Tests

### Configuring the LLMs
- Edit `retrieval_agent_RAG.py` and `retrieval_agent.py` to configure:
  - The LLM model for the Proof Generation agent (default is GPT-4o)
  - The dataset by changing the dataset name in the `load_test_data` function

### Datasets
Three datasets are currently available in this repository:
- `datasets/minif2f.jsonl`
- `datasets/proofnet.jsonl`
- `datasets/mustard_short.jsonl`

Remember to rename both the logging and results files to match your test configuration.

### Frameworks
We provide three LLM testing frameworks:

1. **Default RAG + Knowledge Graph**
   - Location: `algos_retrieval`
   - Base: `retrieval_agent.py`
   - RAG: `retrieval_agent_RAG.py`
   - Graph + RAG: `retrieval_agent_graph_RAG.py`

2. **Best-of-N**
   - Built on top of the RAG + Knowledge Graph setup
   - Location: `algos_best_of_n`

3. **Tree Search**
   - Built on top of the Best-of-N framework
   - Location: `algos_tree_search`

Each folder contains two versions: one adapted for the OpenAI API and one for the TogetherAI API (for open-source models). Feel free to adapt them to other inference APIs such as Anthropic or Hugging Face.

### Running the files
**Before running the file:** ensure that your system has the `killall` command.
You can check by running:
```bash
which killall
```

If the command doesn't exist, install it:
```bash
sudo apt-get update
sudo apt-get install psmisc
```

Now you can run the retrieval agent:
```bash
python retrieval_agent.py
```
or
```bash
python3 retrieval_agent.py
```

To use multiple GPUs (NVIDIA):
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 retrieval_agent.py
```