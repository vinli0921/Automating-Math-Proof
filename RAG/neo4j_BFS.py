import os
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import difflib
import tiktoken

api_key = os.environ.get('OPENAI_API_KEY')

uri = os.environ.get('NEO4J_URI')
user = os.environ.get('NEO4J_USERNAME')
password = os.environ.get('NEO4J_PASSWORD')

driver = GraphDatabase.driver(uri, auth=(user, password))

chat_llm = ChatOpenAI(openai_api_key=api_key, model_name='gpt-4o-mini', temperature=0)

# starting node prompt
starting_node_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert in mathematical concepts represented in a knowledge graph. The node names are the names of the mathematical concepts, such as Odd Integer or Prime Number."
    ),
    (
        "human",
        'Given the following user query:\n\n"{query}"\n\nIdentify the most relevant node title in the knowledge graph that should be used as a starting point for information retrieval.\n\nProvide only the node name, exactly as it appears in the knowledge graph.'
    )
])

starting_node_chain = starting_node_prompt | chat_llm | StrOutputParser()

def find_closest_node_title(driver, node_title):
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN n.name AS name")
        titles = [record["name"] for record in result]
    
    closest_matches = difflib.get_close_matches(node_title, titles, n=1, cutoff=0.5)
    if closest_matches:
        return closest_matches[0]
    else:
        return None
    
def node_exists(driver, title):
    with driver.session() as session:
        result = session.run("MATCH (n {title: $title}) RETURN n LIMIT 1", title=title)
        return result.single() is not None

def get_starting_node_name(query):
    response = starting_node_chain.invoke({"query": query})
    node_title = response.strip()
    print(f"LLM suggested starting node: {node_title}")
    
    # check if the node exists
    if not node_exists(driver, node_title):
        print(f"Node '{node_title}' does not exist. Finding closest match...")
        closest_title = find_closest_node_title(driver, node_title)
        if closest_title:
            print(f"Using closest matching node: {closest_title}")
            node_title = closest_title
        else:
            print("No close match found.")
            node_title = None
    
    return node_title


# BFS
def bfs_traversal(driver, start_name, max_depth=2, max_nodes=50):
    visited = set()
    queue = [(start_name, 0)]
    nodes_content = []
    
    while queue and len(visited) < max_nodes:
        current_name, depth = queue.pop(0)
        if depth > max_depth:
            continue
        if current_name in visited:
            continue
        visited.add(current_name)
        
        # get node content and connected nodes
        with driver.session() as session:
            result = session.execute_read(get_node_and_neighbors, current_name)
        
        if not result:
            continue
        
        content = result['content']
        nodes_content.append(content)
        
        neighbors = result['neighbors']
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, depth + 1))
    
    return nodes_content

def get_node_and_neighbors(tx, name):
    query = """
    MATCH (n:Node {name: $name})
    OPTIONAL MATCH (n)-[:LINK]->(m:Node)
    RETURN n.content AS content, collect(CASE WHEN m IS NOT NULL THEN m.name END) AS neighbors
    """
    result = tx.run(query, name=name)
    record = result.single()
    if record:
        return {
            'content': record['content'],
            'neighbors': record['neighbors']
        }
    else:
        return None

def compile_context(nodes_content, max_tokens=100000):
    # combine and truncate context to fit token limit
    context = "\n\n".join(nodes_content)
    encoding = tiktoken.encoding_for_model('gpt-4')
    tokens = encoding.encode(context)
    if len(tokens) <= max_tokens:
        return context
    else:
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)

# Answer Generation
answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a knowledgeable assistant who uses provided context to answer questions."
    ),
    (
        "system",
        "Context:\n{context}"
    ),
    (
        "human",
        "{question}"
    )
])

answer_chain = answer_prompt | chat_llm | StrOutputParser()

def generate_answer(context, question):
    response = answer_chain.invoke({"context": context, "question": question})
    return response.strip()

# main
def answer_user_query(query, driver, max_depth=2, max_nodes=50):
    # 1: LLM determines starting node
    start_name = get_starting_node_name(query)
    if not start_name:
        return "I'm sorry, I couldn't determine a starting point for your query."
    
    # 2: perform BFS
    nodes_content = bfs_traversal(driver, start_name, max_depth=max_depth, max_nodes=max_nodes)
    
    if not nodes_content:
        return f"I'm sorry, I couldn't find information on '{start_name}' to answer your question."
    
    # 3: compile context
    context = compile_context(nodes_content)
    
    # 4: generate answer
    answer = generate_answer(context, query)
    return answer


user_query = "Prove this statement: Suppose that a, b, and n are whole numbers. If n does not divide a times b, then n does not divide a and b."

answer = answer_user_query(user_query, driver, max_depth=3, max_nodes=50)

print("Answer:")
print(answer)
driver.close()

