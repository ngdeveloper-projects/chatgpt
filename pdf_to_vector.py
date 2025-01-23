import os
import argparse
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

# Function to create a FAISS vector index
def create_vector_index(text_chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    return index, embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feed a PDF document and create a vector index.")
    parser.add_argument("pdf_path", help="Path to the PDF document.")
    parser.add_argument("--chunk_size", type=int, default=500, help="Size of text chunks in words.")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name.")

    args = parser.parse_args()

    # Define output paths based on the current working directory
    current_path = os.getcwd()
    output_index_path = os.path.join(current_path, "vector_index.faiss")
    output_text_chunks_path = os.path.join(current_path, "text_chunks.txt")

    # Extract text from the PDF
    text = extract_text_from_pdf(args.pdf_path)

    # Split the text into chunks
    text_chunks = split_text_into_chunks(text, args.chunk_size)

    # Save the text chunks
    with open(output_text_chunks_path, "w") as f:
        for chunk in text_chunks:
            f.write(chunk + "\n")

    # Create the FAISS vector index
    index, embeddings = create_vector_index(text_chunks, args.model_name)

    # Save the FAISS index
    faiss.write_index(index, output_index_path)

    print(f"Vector index saved to: {output_index_path}")
    print(f"Text chunks saved to: {output_text_chunks_path}")
