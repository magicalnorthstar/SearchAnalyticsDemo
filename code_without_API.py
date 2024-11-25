from pinecone import Pinecone, ServerlessSpec

import os
os.environ['PINECONE_API_KEY'] = "pcsk_Y7XUd_9yzWdv4YXjjAYsVUfL1ZPNQ1nGhGSUK2VGyQ3EhYJWR3D1iLHgiyhN8oAH6YdG5"

pc = Pinecone()

index_name = "my-index"

# Connect to the index
index = pc.Index(index_name)
print(f"Index '{index_name}:' index is ready!")


import psycopg2

def get_data_from_postgres():
    conn = psycopg2.connect(dbname='postgres', user='postgres', password='Silver123', host='database-1.c74q2qwyk2gh.us-east-1.rds.amazonaws.com', port='5432')
    cursor = conn.cursor()
    cursor.execute("SELECT product_id, product_name, description FROM products;")
    data = cursor.fetchall()
    conn.close()
    return data


data = get_data_from_postgres()

print(data)

product_data = [{"id": row[0], "name": row[1], "description": row[2]} for row in data]

from transformers import AlbertTokenizer, TFAlbertModel
import tensorflow as tf

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = TFAlbertModel.from_pretrained('albert-base-v2')

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    
    # Get the embeddings from the BERT model
    outputs = model(**inputs)
    
    # Extract the last hidden state (embedding) and average the token embeddings
    # The output is of shape (batch_size, sequence_length, hidden_size), we average along the sequence dimension
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)  # Mean across sequence length (axis=1)
    
    # Return the embeddings as a numpy array
    return embeddings.numpy()

product_descriptions = [product["description"] for product in product_data]
embeddings = get_embedding(product_descriptions)

# Prepare data for Pinecone ingestion
vectors = [(str(product["id"]), embedding.tolist()) for product, embedding in zip(product_data, embeddings)]

# Insert into Pinecone
#index.upsert(vectors)
#print("upsert done!")

#Query Embedding: Convert the user's query into an embedding and search for similar vectors in Pinecone
query = "Find me a great product for testing"
query_embedding = get_embedding([query])


# Search for similar vectors in Pinecone
results = index.query(vector=query_embedding[0].tolist(), top_k=3)

#print(results)

for match in results['matches']:
    doc_id = match['id']
    conn = psycopg2.connect(dbname='postgres', user='postgres', password='Silver123', host='database-1.c74q2qwyk2gh.us-east-1.rds.amazonaws.com', port='5432')
    cursor = conn.cursor()
    cursor.execute("SELECT description FROM products WHERE product_id = %s", (doc_id,))
    document = cursor.fetchone()
    conn.close()
    print(f"Match: {document[0]}")