from fastapi import FastAPI
from pydantic import BaseModel
import psycopg2
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

# Initialize Pinecone

from pinecone import Pinecone, ServerlessSpec

import os
os.environ['PINECONE_API_KEY'] = "pcsk_Y7XUd_9yzWdv4YXjjAYsVUfL1ZPNQ1nGhGSUK2VGyQ3EhYJWR3D1iLHgiyhN8oAH6YdG5"

pc = Pinecone()

index_name = "my-index"

# Connect to the index
index = pc.Index(index_name)
print(f"Index '{index_name}:' index is ready!")



# Connect to PostgreSQL
conn = psycopg2.connect(dbname='postgres', user='postgres', password='Silver123', host='database-1.c74q2qwyk2gh.us-east-1.rds.amazonaws.com', port='5432')
cursor = conn.cursor()

# FastAPI app setup
app = FastAPI()

# Hugging Face BERT model setup
from transformers import AlbertTokenizer, TFAlbertModel
import tensorflow as tf

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = TFAlbertModel.from_pretrained('albert-base-v2')

def get_embedding(text):
    """
    Generate an embedding for a given text using BERT.
    """
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    return embeddings.numpy().squeeze()

# Pydantic model to accept input for the query
class QueryRequest(BaseModel):
    query: str

@app.post("/search/")
def search(query_request: QueryRequest):
    """
    Handle search query, generate embedding, and find matching documents from Pinecone.
    """
    query = query_request.query
    
    # Step 1: Generate embedding for the query
    query_vector = get_embedding(query)
    
    # Step 2: Convert numpy array to list before passing to Pinecone
    query_vector = query_vector.tolist()

    # Step 3: Query Pinecone to find the most similar vectors
    results = index.query(vector=[query_vector], top_k=3)  # Get top 3 most similar vectors
    

    # Step 4: Fetch matching documents from PostgreSQL
    matched_docs = []
    for match in results['matches']:
        doc_id = match['id']
        cursor.execute("SELECT description FROM products WHERE product_id = %s", (doc_id,))
        document = cursor.fetchone()
        matched_docs.append({"id": doc_id, "text_data": document[0]})
    
    # Step 5: Return the final results as JSON
    return {"results": matched_docs}