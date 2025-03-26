# LLM-proof
Repo for Algoverse Research

# 1. Getting Started
https://github.com/vinli0921/LLM-proof/blob/main/README.md
## 1.1 Setting up the repository
- First download [elan](https://github.com/leanprover/elan)
- Run these commands
```
git clone --recurse-submodules https://github.com/vinli0921/LLM-proof.git
cd LLM-proof
cd mathlib4
lake build
```

## 1.2 Create a virtual environment 
- Create a virtual enviornment in the root directory of the repository.
- You may replace "myenv" with a name of your choice
```
python -m venv myenv
```

## 1.3 Activate the virtual environment

### On macOS and Linux
```
source myenv/bin/activate
```

### On Windows
```
myenv\Scripts\activate
```

## 1.4 Install Dependencies
To install all packages and code dependencies please run:
```
pip install -r requirements.txt
```

## 1.5 Set Enviornment Variables
1. In the root directory of the repository, make a new file named **.env**
2. Copy + Paste the content from the file named **.env.example** into the newly created **.env** file
3. Set the variables to the necessary values

# 2. Extracting Proofs to Your Own Neo4j AuraDB (OPTIONAL)

## 2.1 Retrieving the XML Dump
1. Press this link: https://proofwiki.org/xmldump/latest.xml.gz which will download a zip file
2. Unzip the file to extract **latest.xml**
3. Move **latest.xml** to the root directory of the repository

## 2.2 Extracting to Nodes and Relationships from the XML file to CSV files
run the command 
```
python Graph_Creation/extractProofsXML.py latest.xml
```

## 2.3 Uploading CSV files to Neo4j AuraDB
run the file **neo4j_kg.py**
- this should successfully upload around 20k nodes and 80k relationships (give or take)

# 3. Running Tests
## 3.1 Configuring the LLMS
- Go to retrieval_agent_RAG.py and retrieval_agent.py
- You can change the LLM model of the Proof Generation agent by specifiying the model name string in the constructor located in the main method. Default model is GPT-4o.

## 3.2 Configuring the datasets
- Go to retrieval_agent_RAG.py and retrieval_agent.py
- You can change the datasets by changing the dataset name string in the load_test_data function located in the main method.
- Currently, there are two datasets:
  - datasets/minif2f.jsonl
  - datasets/proofnet.jsonl
- Please also rename both the logging and results files to fit the context.

## 3.3 Running the files
**Before running the file:** ensure that the system you are on has the command ```killall```.\
You can check by running the following in a terminal:
```
which killall
```
If this command doesn't exist in the system, please install it using the folowing commands:
```
sudo apt-get update
sudo apt-get install psmisc
```
You should now be able to run the files:
```
python retrieval_agent.py
```
or
```
python3 retrieval_agent.py
```
If you want to use multiple GPUS (nvidia):
```
CUDA_VISIBLE_DEVICES=0,1 python3 retrieval_agent.py
```
