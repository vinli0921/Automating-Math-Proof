import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd

load_dotenv()

neo4j_uri = os.environ.get("NEO4J_URI")
neo4j_user = os.environ.get("NEO4J_USERNAME")
neo4j_password = os.environ.get("NEO4J_PASSWORD")

gds = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

def create_vector_index(session, dimensions=3072, similarity_function='cosine'):
    """
    Creates a vector index on the 'plotEmbedding' property of nodes labeled 'Node'.
    """
    query = f"""
    CREATE VECTOR INDEX nodeContent IF NOT EXISTS
    FOR (m:Node)
    ON (m.plotEmbedding)
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dimensions},
            `vector.similarity_function`: '{similarity_function}'
        }}
    }}
    """
    session.run(query)
    print("Vector index 'nodeContent' created or already exists.")

def query_vector_index(session, query_embedding, top_k=10):
    """
    Queries the vector index to find the top K most similar nodes to the query embedding.
    """
    result = session.run("""
    WITH $embedding AS query_embedding
    CALL db.index.vector.queryNodes(
        'nodeContent',
        query_embedding,
        $top_k
    ) YIELD nodeId, score
    RETURN nodeId, score
    ORDER BY score DESC
    """, embedding=query_embedding, top_k=top_k)
    
    nodes = [(record["nodeId"], record["score"]) for record in result]
    return nodes

def main():
    with gds.session() as session:
        # Create the vector index
        create_vector_index(session, dimensions=3072, similarity_function='cosine')
        
        # Replace with your actual query embedding vector
        # query_embedding = [0.1] * 3072  # Example embedding
        
        # # Query the vector index
        # top_k = 10
        # results = query_vector_index(session, query_embedding, top_k)
        
        # # Print the results
        # for node_id, score in results:
        #     print(f"Node ID: {node_id}, Similarity Score: {score}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        gds.close()
