import openai
import csv
import os
import json
from dotenv import load_dotenv
from tiktoken import get_encoding
from neo4j import GraphDatabase

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")
neo4j_uri = os.environ.get("NEO4J_URI")
neo4j_user = os.environ.get("NEO4J_USERNAME")
neo4j_password = os.environ.get("NEO4J_PASSWORD")

gds = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))


# Initialize encoding for token counting
encoding = get_encoding("cl100k_base")  # Use appropriate encoding for your model

# Define maximum tokens allowed by the model
MAX_TOKENS = 8191  # Adjust based on the specific model's limit
EMBEDDING_MODEL = "text-embedding-3-large"  # Replace with your embedding model (3072)
BATCH_SIZE = 100  # Number of nodes to process at once

def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def split_text(text, max_tokens):
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = encoding.decode(tokens[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

def structure_mathematical_text(title, theorem, proof, math_expressions):
    """Structure the text to prioritize mathematical components."""
    structured_text = f"Title: {title}\n"
    
    if theorem:
        structured_text += f"Theorem: {theorem}\n"
    
    if proof:
        structured_text += f"Proof: {proof}\n"
    
    if math_expressions:
        structured_text += f"Mathematical Expressions: {' '.join(math_expressions)}"
        
    return structured_text.strip()

def generate_embeddings(input_csv, output_csv):
    with open(input_csv, mode='r', encoding='utf-8') as infile, \
         open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        writer.writerow(['id', 'embedding'])
        
        count = 1
        for row in reader:
            node_id = row['id']
            title = row['title']
            theorem = row['theorem'] if 'theorem' in row else ''
            proof = row['proof'] if 'proof' in row else ''
            
            try:
                math_expressions = json.loads(row['math_expressions']) if 'math_expressions' in row and row['math_expressions'] else []
            except (json.JSONDecodeError, TypeError):
                math_expressions = []
            
            structured_text = structure_mathematical_text(
                title, theorem, proof, math_expressions
            )
            
            if not structured_text:
                print(f"{count}. Skipping empty Node {node_id}")
                continue
            
            token_count = len(encoding.encode(structured_text))
            if token_count > MAX_TOKENS:
                print(f"{count}. Text too long for Node {node_id}, splitting into chunks.")
                chunks = split_text(structured_text, MAX_TOKENS // 2)
                embeddings = []
                for chunk in chunks:
                    embedding = get_embedding(chunk)
                    embeddings.append(embedding)
                aggregated_embedding = [sum(values)/len(values) for values in zip(*embeddings)]
                writer.writerow([node_id, aggregated_embedding])
            else:
                print(f"{count}. Generating embedding for Node {node_id}")
                embedding = get_embedding(structured_text)
                writer.writerow([node_id, embedding])
            
            count += 1
            
def generate_neo4j_embeddings(output_csv):
    """
    Generate embeddings for nodes in Neo4j database and save to CSV.
    
    Args:
        output_csv (str): Path to output CSV file
    """
    def get_total_nodes(tx):
        result = tx.run("MATCH (n:Node) RETURN count(n) as count")
        return result.single()["count"]
    
    def get_batch_nodes(tx, skip, limit):
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

    with gds.session() as session:
        # Get total number of nodes
        total_nodes = session.execute_read(get_total_nodes)
        print(f"Total nodes to process: {total_nodes}")

        # Open CSV file for writing
        with open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['id', 'embedding'])
            
            # Process nodes in batches
            count = 1
            for skip in range(0, total_nodes, BATCH_SIZE):
                # Get batch of nodes
                nodes = session.execute_read(get_batch_nodes, skip, BATCH_SIZE)
                
                for node in nodes:
                    node_id = node["id"]
                    title = node["title"] or ""
                    theorem = node["theorem"] or ""
                    proof = node["proof"] or ""
                    
                    # Parse math_expressions from string if it exists
                    try:
                        math_expressions = json.loads(node["math_expressions"]) if node["math_expressions"] else []
                    except json.JSONDecodeError:
                        math_expressions = []
                    
                    # Structure the text with mathematical components
                    structured_text = structure_mathematical_text(
                        title, theorem, proof, math_expressions
                    )
                    
                    if not structured_text:
                        print(f"{count}. Skipping empty Node {node_id}")
                        continue

                    # Check token length and generate embedding
                    token_count = len(encoding.encode(structured_text))
                    if token_count > MAX_TOKENS:
                        print(f"{count}. Text too long for Node {node_id}, splitting into chunks.")
                        chunks = split_text(structured_text, MAX_TOKENS // 2)
                        embeddings = []
                        for chunk in chunks:
                            embedding = get_embedding(chunk)
                            embeddings.append(embedding)
                        aggregated_embedding = [sum(values)/len(values) for values in zip(*embeddings)]
                        writer.writerow([node_id, aggregated_embedding])
                    else:
                        print(f"{count}. Generating embedding for Node {node_id}")
                        embedding = get_embedding(structured_text)
                        writer.writerow([node_id, embedding])
                    
                    count += 1
                
                processed = min(skip + BATCH_SIZE, total_nodes)
                print(f"Processed {processed} out of {total_nodes} nodes")

# Example function to get embedding size
def embedding_size():
    response = openai.embeddings.create(
        input="Hello, World!",
        model=EMBEDDING_MODEL
    )
    return len(response.data[0].embedding)

if __name__ == "__main__":
    # input_csv = '../nodes.csv'  # path to input CSV file
    # output_csv = 'embeddings.csv'  # path to output CSV file
    # # print(f"Embedding size: {embedding_size()}")  # debugging statement
    # generate_embeddings(input_csv, output_csv)
    output_csv = 'embeddings.csv'
    try:
        print(f"Starting embedding generation...")
        generate_neo4j_embeddings(output_csv)
        print(f"Embedding generation completed. Results saved to {output_csv}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        gds.close()
