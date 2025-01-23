import os
from sentence_transformers import SentenceTransformer
import numpy as np
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    """
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def split_text_into_chunks(text, chunk_size=100):
    """
    Split text into smaller chunks of approximately chunk_size words.
    """
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def save_embeddings(embeddings, chunks, file_path):
    """
    Save embeddings and corresponding chunks to a file.
    """
    np.savez(file_path, embeddings=embeddings, chunks=chunks)

def load_embeddings(file_path):
    """
    Load embeddings and corresponding chunks from a file.
    """
    data = np.load(file_path, allow_pickle=True)
    return data['embeddings'], data['chunks']

def main():
    # Step 1: Load the model
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Define file paths
    pdf_path = input("Enter the PDF file path: ").strip()  # Interactive input for PDF path
    embeddings_file = f"{os.path.splitext(pdf_path)[0]}_embeddings.npz"  # Unique file for each PDF

    # Check if embeddings are already stored
    if os.path.exists(embeddings_file):
        print("Loading stored embeddings...")
        corpus_embeddings, chunks = load_embeddings(embeddings_file)
    else:
        # Step 2: Extract text from the PDF
        print(f"Extracting text from {pdf_path}...")
        text = extract_text_from_pdf(pdf_path)

        # Step 3: Preprocess and split text into chunks
        print("Splitting text into chunks...")
        chunks = split_text_into_chunks(text)

        # Step 4: Create embeddings for the chunks
        print("Creating embeddings...")
        corpus_embeddings = model.encode(chunks)

        # Save embeddings for future use
        print(f"Saving embeddings to {embeddings_file}...")
        save_embeddings(corpus_embeddings, np.array(chunks, dtype=object), embeddings_file)

    # Interactive querying loop
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break

        print(f"Query: {query}")
        query_embedding = model.encode(query)

        # Compute cosine similarity
        cosine_similarities = np.dot(corpus_embeddings, query_embedding) / (
            np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Find the most relevant chunk
        most_similar_idx = np.argmax(cosine_similarities)
        print(f"Most relevant chunk: {chunks[most_similar_idx]}")
        print(f"Similarity Score: {cosine_similarities[most_similar_idx]:.4f}")

if __name__ == "__main__":
    main()