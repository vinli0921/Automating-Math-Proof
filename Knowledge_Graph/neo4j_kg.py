import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd
import json
import re
from typing import Dict

load_dotenv()

# Neo4j config and constraints
neo4j_uri = os.environ.get("NEO4J_URI")
neo4j_user = os.environ.get("NEO4J_USERNAME")
neo4j_password = os.environ.get("NEO4J_PASSWORD")

gds = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

def create_constraints(session):
    """
    Creates a uniqueness constraint Cypher query on :Node(id).

    Args:
        session (Session): Neo4j Session object from the Neo4j driver.
    """   
     
    session.run("""
    CREATE CONSTRAINT node_id_unique IF NOT EXISTS
    FOR (n:Node)
    REQUIRE n.id IS UNIQUE
    """)

def load_nodes(session, filepath):
    """
    Loads nodes from a CSV file into the Neo4j database through batching.

    Args:
        session (Session): Neo4j session object from the Neo4j driver.
        filepath (str): The file path to the CSV file containing node data.
                        The CSV file should have columns: 'id', 'type', 'title', 'name', 'content'.
    """    
    
    df_nodes = pd.read_csv(filepath)
    # convert 'id' to integer
    df_nodes['id'] = df_nodes['id'].astype(int)
    
    # Convert math_expressions to proper JSON strings if they exist
    if 'math_expressions' in df_nodes.columns:
        df_nodes['math_expressions'] = df_nodes['math_expressions'].apply(
            lambda x: json.dumps(x) if isinstance(x, list) else x
        )

    batch_size = 1000
    total_rows = len(df_nodes)
    for i in range(0, total_rows, batch_size):
        batch = df_nodes.iloc[i:i+batch_size]
        nodes = batch.to_dict('records')
        session.execute_write(create_nodes_batch, nodes)
        print(f"Processed {min(i+batch_size, total_rows)} out of {total_rows} nodes")

def create_nodes_batch(tx, nodes):
    """
    Neo4j Cypher query to create nodes in the database.

    Args:
        tx (tx): The transaction object provided by the Neo4j driver.
        nodes (list): A list of node dictionaries to be created.
                      Each dictionary should have keys: 'id', 'type', 'title', 'name', 'content'.
    """    
    
    tx.run("""
    UNWIND $nodes AS node
    MERGE (n:Node {id: node.id})
    SET n.type = node.type,
        n.title = node.title,
        n.name = node.name,
        n.content = node.content,
        n.theorem = node.theorem,
        n.proof = node.proof,
        n.math_expressions = node.math_expressions
    """, nodes=nodes)

def load_relationships(session, filepath):
    """
    Loads relationships from a CSV file into the Neo4j database using batching.

    Args:
        session (Session): Neo4j Session object from the Neo4j driver.
        filepath (str): The file path to the CSV file containing relationships.
                        The CSV file should have columns: 'from_id', 'to_id', and 'type'.
    """    
    
    df_rels = pd.read_csv(filepath)
    # convert IDs to integers
    df_rels['from_id'] = df_rels['from_id'].astype(int)
    df_rels['to_id'] = df_rels['to_id'].astype(int)
    
    # Ensure type and context exist
    if 'context' not in df_rels.columns:
        df_rels['context'] = ''
    if 'type' not in df_rels.columns:
        df_rels['type'] = 'LINK'

    batch_size = 1000
    total_rows = len(df_rels)
    for i in range(0, total_rows, batch_size):
        batch = df_rels.iloc[i:i+batch_size]
        relationships = batch.to_dict('records')
        session.execute_write(create_relationships_batch, relationships)
        print(f"Processed {min(i+batch_size, total_rows)} out of {total_rows} relationships")

def create_relationships_batch(tx, relationships):
    """
    Executes a Cypher query to create relationships in the Neo4j database for a batch of data.

    Args:
        tx (tx): The transaction object provided by the Neo4j driver.
        relationships (list): A list of relationship dictionaries to be created.
                              Each dictionary should have keys: 'from_id', 'to_id', and 'type'.
    """    
    
    tx.run("""
    UNWIND $relationships AS rel
    MATCH (from:Node {id: rel.from_id})
    MATCH (to:Node {id: rel.to_id})
     MERGE (from)-[r:LINK {type: rel.type, context: rel.context}]->(to)
    """, relationships=relationships)
    
def append_embeddings(session, filepath, batch_size=100):
    """
    Reads embeddings from a CSV file and updates the 'plotEmbedding' property of nodes in the database.
    Uses chunked reading to manage memory usage.

    Args:
        session (Session): Neo4j Session object from the Neo4j driver.
        filepath (str): The file path to the CSV file containing embeddings.
        batch_size (int): Number of rows to process at once. Defaults to 100.
    """
    # Get total number of rows first
    total_rows = sum(1 for _ in open(filepath)) - 1  # subtract 1 for header
    print(f"Total embeddings to process: {total_rows}")
    
    # Process the CSV in chunks
    for chunk_number, chunk in enumerate(pd.read_csv(filepath, chunksize=batch_size)):
        # Convert IDs to integers
        chunk['id'] = chunk['id'].astype(int)
        
        # Process embeddings one at a time in this chunk
        embeddings = []
        for _, row in chunk.iterrows():
            try:
                embedding_vector = json.loads(row['embedding'])
                embeddings.append({
                    'id': row['id'],
                    'embedding': embedding_vector
                })
            except json.JSONDecodeError as e:
                print(f"Error processing embedding for ID {row['id']}: {e}")
                continue
        
        # Update database with this batch
        if embeddings:
            session.execute_write(update_embeddings_batch, embeddings)
        
        # Calculate progress
        processed = min((chunk_number + 1) * batch_size, total_rows)
        print(f"Processed {processed} out of {total_rows} embeddings")
        
def update_embeddings_batch(tx, embeddings):
    """
    Updates the 'plotEmbedding' property of nodes in the database for a batch of embeddings.

    Args:
        tx (Transaction): The transaction object provided by the Neo4j driver.
        embeddings (list): A list of dictionaries with 'id' and 'embedding' keys.
    """
    tx.run("""
    UNWIND $embeddings AS embedding
    MATCH (n:Node {id: embedding.id})
    SET n.plotEmbedding = embedding.embedding
    """, embeddings=embeddings)
    
def clean_content(content: str) -> str:
    """Clean retrieved content by removing unwanted sections and formatting while preserving proofs."""
    # Define filter patterns - removed patterns that would remove proofs and equations
    # Handle non-string content
    if not isinstance(content, str):
        return ""
    
    filter_patterns = [
        r'\[\[Category:.?\]\]',  # Category links
        r'== Also see ==.?(?=\n\n|\Z)',  # Also see sections
        r'== Historical Note ==.?(?=\n\n|\Z)',  # Historical notes
        r'== Notation ==.?(?=\n\n|\Z)',  # Notation sections
        r'== Examples ==.?(?=\n\n|\Z)',  # Examples sections
        r'\* \[\[.?\]\].?\n',  # Bullet points with wiki links
        r'{{Link-to-category\|.?}}',  # Link to category templates
        r'{{:.*?}}',  # Template inclusions (but not equation templates)
    ]

    # Extract content between <onlyinclude> tags while keeping the content inside
    content = re.sub(r'<onlyinclude>(.*?)</onlyinclude>', r'\1', content, flags=re.DOTALL)

    # Apply all filter patterns
    for pattern in filter_patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL)

    # Remove multiple newlines and clean up spacing
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = re.sub(r'^\s+|\s+$', '', content)

    # Remove wiki formatting but preserve section headers for proofs
    content = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', content)

    return content

def clean_mathematical_content(node: Dict) -> Dict:
    """Clean mathematical content while preserving structure."""
    cleaned = {
        "id": node["id"],
        "cleaned_theorem": "",
        "cleaned_proof": "",
        "math_expressions": []
    }
    
    # Extract and clean theorem content
    theorem = node.get("theorem", "")
    if theorem is not None:
        cleaned["cleaned_theorem"] = clean_content(str(theorem))
    
    # Extract and clean proof content
    proof = node.get("proof", "")
    if proof is not None:
        cleaned["cleaned_proof"] = clean_content(str(proof))
    
    # Extract mathematical expressions from both theorem and proof
    content = ""
    if theorem is not None:
        content += str(theorem) + " "
    if proof is not None:
        content += str(proof)
        
    math_patterns = [
        r'\$.*?\$',  # Inline math
        r'\\ds.*?(?=\\ds|$)',  # Display style math
        r'\\begin\{.*?\}.*?\\end\{.*?\}'  # Environment math
    ]
    
    math_expressions = []
    for pattern in math_patterns:
        matches = re.finditer(pattern, content, re.DOTALL)
        math_expressions.extend(match.group(0) for match in matches)
    
    cleaned["math_expressions"] = json.dumps(math_expressions)
    
    return cleaned
    
def get_batch_nodes(tx, skip, limit):
    """Retrieve nodes with mathematical content."""
    result = tx.run("""
        MATCH (n:Node)
        RETURN n.id as id, 
            n.title as title,
            n.theorem as theorem,
            n.proof as proof,
            n.math_expressions as math_expressions
        SKIP $skip
        LIMIT $limit
    """, skip=skip, limit=limit)
    return [record for record in result]

def update_nodes_batch(tx, nodes):
    tx.run("""
    UNWIND $nodes as node
    MATCH (n:Node {id: node.id})
    SET n.theorem = node.cleaned_theorem,
        n.proof = node.cleaned_proof,
        n.math_expressions = node.math_expressions
    """, nodes=nodes)
        
def update_node_content(session, batch_size=1000):
    """
    Updates all node content in the Neo4j database using the content cleaning function.
    Processes nodes in batches to manage memory and performance.

    Args:
        session (Session): Neo4j Session object from the Neo4j driver.
        batch_size (int): Number of nodes to process in each batch. Defaults to 1000.
    """
    def get_total_nodes(tx):
        result = tx.run("MATCH (n:Node) RETURN count(n) as count")
        return result.single()["count"]

    def get_batch_nodes(tx, skip, limit):
        result = tx.run("""
            MATCH (n:Node)
            RETURN n.id as id, 
                   n.theorem as theorem,
                   n.proof as proof
            SKIP $skip
            LIMIT $limit
        """, skip=skip, limit=limit)
        return [dict(record) for record in result]

    total_nodes = session.execute_read(get_total_nodes)
    print(f"Total nodes to process: {total_nodes}")

    for skip in range(0, total_nodes, batch_size):
        nodes = session.execute_read(get_batch_nodes, skip, batch_size)
        cleaned_nodes = [clean_mathematical_content(node) for node in nodes]
        
        session.execute_write(update_nodes_batch, cleaned_nodes)
        
        processed = min(skip + batch_size, total_nodes)
        print(f"Processed {processed} out of {total_nodes} nodes")

def main():
    with gds.session() as session:
        #create_constraints(session)
        #load_nodes(session, 'nodes.csv') 
        append_embeddings(session, 'embeddings.csv')
        #load_relationships(session, 'relationships.csv') 
        update_node_content(session)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        gds.close()
