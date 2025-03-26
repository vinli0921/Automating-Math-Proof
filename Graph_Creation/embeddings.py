import openai
import csv
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

def generate_embeddings(input_csv, output_csv):
    with open(input_csv, mode='r') as infile, open(output_csv, mode='w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        writer.writerow(['id', 'embedding'])
        
        # skip the header row
        next(reader, None)
        
        count = 1
        for row in reader:
            node_id, type, title, name, content = row
            print(f"{count}. Generating embedding for Node {node_id} : {name}")  # debugging statement
            embedding = get_embedding(title + content)
            writer.writerow([node_id, embedding])
            count += 1

# 3072
def embedding_size():
    response = openai.embeddings.create(
        input="Hello, World!",
        model="text-embedding-3-large"
    )
    return len(response.data[0].embedding)

if __name__ == "__main__":
    input_csv = '../nodes.csv'  # path to input CSV file
    output_csv = 'embeddings.csv'  # path to output CSV file
    # print(f"Embedding size: {embedding_size()}")  # debugging statement
    generate_embeddings(input_csv, output_csv)