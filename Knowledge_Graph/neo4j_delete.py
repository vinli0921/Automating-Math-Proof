from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

def delete_all_nodes(uri, user, password):
    """
    Connects to the Neo4j database and deletes all nodes and relationships.

    Args:
        uri (str): The URI of the Neo4j database.
        user (str): The username for authentication (default is usually "neo4j").
        password (str): The password for authentication.
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()  # Check if this raises any error
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("All nodes and relationships have been deleted.")

    driver.close()

if __name__ == "__main__":
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")

    delete_all_nodes(uri, user, password)
