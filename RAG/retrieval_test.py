import os
import openai
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import tiktoken

from Neo4jVectorStoreRetriever import Neo4jVectorStoreRetriever

# Initialize OpenAI API key
api_key = os.environ.get('OPENAI_API_KEY')
openai.api_key = api_key

# Initialize Neo4j driver
uri = os.environ.get('NEO4J_URI')
user = os.environ.get('NEO4J_USERNAME')
password = os.environ.get('NEO4J_PASSWORD')
driver = GraphDatabase.driver(uri, auth=(user, password))

# Initialize LLM
llm = ChatOpenAI(openai_api_key=api_key, model_name='gpt-4o-mini', temperature=0)

# Answer Generation Prompt
answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert mathematician who uses provided context to answer questions."
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
answer_chain = answer_prompt | llm | StrOutputParser()

def generate_answer(context, question):
    response = answer_chain.invoke({"context": context, "question": question})
    return response.strip()

def compile_context(documents, max_tokens=100000):
    # Combine and truncate context to fit token limit
    context = "\n\n".join([doc.page_content for doc in documents])
    encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    tokens = encoding.encode(context)
    if len(tokens) <= max_tokens:
        return context
    else:
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)

def answer_user_query(query, driver, max_depth=3, top_k=5):
    # Initialize the retriever
    retriever = Neo4jVectorStoreRetriever(driver=driver, top_k=top_k)

    for depth in range(0, max_depth+1):
        if depth == 0:
            # Zero-shot prompting without context
            context = ""
        else:
            # Update top_k if necessary
            retriever.top_k = top_k * depth
            # Get relevant documents using the retriever
            documents = retriever(query)  # Use retriever as a callable
            if not documents:
                continue
            # Compile context
            context = compile_context(documents)
            print(f"Context at depth {depth}: {context}")
        # Generate answer
        answer = generate_answer(context, query)
        print(f"Answer at depth {depth}:")
        print(answer)
        # Implement a check to determine if the answer is acceptable
        # If acceptable, break the loop
        # if is_answer_acceptable(answer):
        #     break
    return answer

# Main execution
user_query = "Prove this statement: Suppose that a, b, and n are whole numbers. If n does not divide a times b, then n does not divide a and n does not divide b."

answer = answer_user_query(user_query, driver, max_depth=3, top_k=5)

print("Final Answer:")
print(answer)
driver.close()
