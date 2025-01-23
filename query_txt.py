import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline


def extract_text_from_file(file_path):
    """
    Extract text from a plain text file and return it as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def split_text_into_chunks(text, chunk_size=100, overlap=50):
    """
    Split text into overlapping chunks of approximately `chunk_size` words with `overlap`.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks


def save_embeddings(embeddings, chunks, file_path):
    """
    Save embeddings and corresponding chunks to a .npz file.
    """
    np.savez(file_path, embeddings=embeddings, chunks=chunks)


def load_embeddings(file_path):
    """
    Load embeddings and corresponding chunks from a .npz file.
    """
    data = np.load(file_path, allow_pickle=True)
    return data['embeddings'], data['chunks']


def compute_cosine_similarity(embeddings, query_embedding):
    """
    Compute cosine similarity between embeddings and the query embedding.
    """
    return np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )


def get_relevant_chunks(corpus_embeddings, query_embedding, chunks, top_k=5, relevance_threshold=0.5):
    """
    Retrieve the top-k most relevant chunks based on cosine similarity with a relevance threshold.
    """
    cosine_similarities = compute_cosine_similarity(corpus_embeddings, query_embedding)
    top_indices = np.argsort(cosine_similarities)[-top_k:][::-1]
    relevant_chunks = [(chunks[idx], cosine_similarities[idx]) for idx in top_indices]
    return [(chunk, score) for chunk, score in relevant_chunks if score > relevance_threshold]


def generate_response(llm, context, query, max_length=512):
    """
    Generate a detailed response using the LLM with improved prompt engineering.
    Truncate context to fit within the model's maximum input length.
    """
    truncated_context = context[:max_length]
    prompt = (
        f"Context: {truncated_context}\n\n"
        f"Question: {query}\n\n"
        f"Provide a clear, specific, and accurate answer based only on the context provided."
    )
    response = llm(prompt)
    return response[0]['generated_text']


def interactive_query_loop(embedding_model, llm, corpus_embeddings, chunks):
    """
    Run an interactive query loop for the user to ask questions.
    """
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break

        if not query:
            print("Please enter a valid query.")
            continue

        print(f"Query: {query}")
        query_embedding = embedding_model.encode(query)

        # Retrieve top-k relevant chunks
        top_k = 5
        relevant_chunks = get_relevant_chunks(corpus_embeddings, query_embedding, chunks, top_k)

        # Combine the top-k chunks for context
        combined_context = " ".join([chunk for chunk, _ in relevant_chunks])
        print("\nMost Relevant Chunks:")
        for chunk, score in relevant_chunks:
            print(f"Chunk: {chunk[:100]}... (Similarity: {score:.4f})")

        if not combined_context.strip():
            print("No relevant chunks found. Please try rephrasing your query.")
            continue

        # Generate response using the LLM
        print("\nGenerating response...")
        response = generate_response(llm, combined_context, query)
        print(f"Response: {response}")


def initialize_models():
    """
    Load the embedding model and the language model.
    """
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Loading language model...")
    llm = pipeline("text2text-generation", model="google/flan-t5-base")  # Hugging Face LLM
    return embedding_model, llm


def process_text_file(file_path, embedding_model):
    """
    Extract text from a text file, split it into chunks, and generate embeddings.
    """
    print(f"Reading text from {file_path}...")
    text = extract_text_from_file(file_path)

    print("Splitting text into chunks...")
    chunks = split_text_into_chunks(text)

    print("Creating embeddings...")
    corpus_embeddings = embedding_model.encode(chunks)

    return corpus_embeddings, chunks


def main():
    # Load models
    embedding_model, llm = initialize_models()

    # Get the text file path and check if embeddings already exist
    file_path = input("Enter the text file path: ").strip()
    if not os.path.exists(file_path):
        print("Error: File not found. Please provide a valid text file path.")
        return

    embeddings_file = f"{os.path.splitext(file_path)[0]}_embeddings.npz"

    if os.path.exists(embeddings_file):
        print("Loading stored embeddings...")
        corpus_embeddings, chunks = load_embeddings(embeddings_file)
    else:
        # Process the text file and generate embeddings
        corpus_embeddings, chunks = process_text_file(file_path, embedding_model)

        # Save embeddings for reuse
        print(f"Saving embeddings to {embeddings_file}...")
        save_embeddings(corpus_embeddings, np.array(chunks, dtype=object), embeddings_file)

    # Start interactive query loop
    interactive_query_loop(embedding_model, llm, corpus_embeddings, chunks)


if __name__ == "__main__":
    main()
