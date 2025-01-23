import os
import argparse
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import logging

# Suppress logging for transformers and PyTorch
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Pre-download the summarization model
MODEL_NAME = "facebook/bart-large-cnn"
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
summarizer_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
summarizer = pipeline("summarization", model=summarizer_model, tokenizer=summarizer_tokenizer, device=-1)  # Set to CPU explicitly

# Function to load text chunks from a file
def load_text_chunks():
    current_path = os.getcwd()
    text_chunks_path = os.path.join(current_path, "text_chunks.txt")
    with open(text_chunks_path, "r") as f:
        text_chunks = f.readlines()
    return [chunk.strip() for chunk in text_chunks]

# Function to summarize text
def summarize_text(text, max_length=50):
    summary = summarizer(text, max_length=max_length, min_length=20, do_sample=False)
    return summary[0]['summary_text']

# Function to query the FAISS vector index
def query_vector_index(query, text_chunks, model_name="all-MiniLM-L6-v2"):
    current_path = os.getcwd()
    index_path = os.path.join(current_path, "vector_index.faiss")
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])

    # Load the FAISS index
    index = faiss.read_index(index_path)

    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), 1)  # Only get the top 1 result
    result_text = text_chunks[indices[0][0]]
    summarized_text = summarize_text(result_text)
    return summarized_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a vector index.")
    parser.add_argument("query", help="The query to search for in the vector index.")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name.")

    args = parser.parse_args()

    # Load the text chunks
    text_chunks = load_text_chunks()

    # Query the FAISS index
    result_text = query_vector_index(args.query, text_chunks, args.model_name)

    # Print only the result text
    print(result_text)
