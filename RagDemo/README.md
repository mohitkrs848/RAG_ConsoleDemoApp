# RAG (Retrieval-Augmented Generation) — Project Guide

This README contains a detailed explanation of the simple RAG demo included in this repository. It is written for a developer who is new to RAG and to this codebase.

## Goal

Provide a minimal, self-contained example that demonstrates the RAG flow:
- Index documents (create embeddings and store them)
- Retrieve the most relevant documents for a query using vector similarity
- Use retrieved context to produce or compose an answer (simulated in this demo)

This project intentionally uses a local, deterministic embedding generator so you can focus on the RAG flow without needing external APIs or keys.

## Key files and responsibilities

- Program.cs
  - Entry point. Creates RagEngine, ensures the `Docs` folder exists, indexes documents, accepts queries, and prints answers.

- Services/EmbeddingService.cs
  - Produces deterministic float[] embeddings from text using SHA256. The same input always yields the same vector.
  - Purpose: stand-in for a real embeddings API so newcomers can learn retrieval and scoring.

- Services/VectorStoreService.cs
  - Stores tuples of (Id, Embedding, Text).
  - Implements a simple cosine similarity function and returns top-K matches with scores.

- Services/RagEngine.cs
  - Orchestrates indexing and querying.
  - IndexDocuments resolves the repository `Docs` folder reliably (works when running from bin/Debug or the project root), indexes all `*.txt` files, and logs what it finds.
  - AskAsync embeds the query, retrieves top results from the vector store, attempts a simple direct answer extraction for Q/A-formatted documents, and otherwise composes a context prompt and returns a simulated answer.

## High-level flow

1. Indexing
   - Read all .txt files inside the resolved `Docs` folder.
   - For each file, compute an embedding with EmbeddingService.GenerateAsync(text).
   - Store (filePath, embedding, text) in VectorStoreService.

2. Querying
   - User inputs a query.
   - Compute query embedding with EmbeddingService.GenerateAsync(query).
   - Call VectorStoreService.Search(queryEmbedding, topK) to get most similar documents (cosine similarity).
   - Attempt to extract a direct answer from top documents (matches Q:/A: style or nearby text).
   - If extraction fails, compose a prompt combining the retrieved context and the query, and return a simulated response.

## Technical notes

- Deterministic embeddings (SHA256->float[]):
  - Not semantically meaningful like a real model, but suitable for demonstrating vector similarity and retrieval code paths.

- Cosine similarity:
  - The vector store computes cosine similarity and returns scores. Higher is more similar.

- Folder resolution and running from bin/Debug:
  - When running the app inside Visual Studio or with `dotnet run`, the process working directory may be the bin folder. The indexer prefers the project root (looks for a .csproj) to locate the repository `Docs` folder so your root-level docs are found reliably.

## Running the demo

1. Open the solution in Visual Studio or use the command line from the project folder (where RagDemo.csproj is located).
2. Build and run:
   - dotnet run --project RagDemo.csproj
3. The app will index `Docs/*.txt` and print logs like:
   - "Indexing folder: <path> - found N .txt files"
   - "Indexed file: <path> (length: XYZ chars)"
4. When prompted, type a question, for example:
   - What are your store hours?
5. Expected behavior:
   - The app prints found results and either:
	 - a direct extracted answer (if the document contains a matching Q:/A: entry) or
	 - a simulated prompt/answer demonstrating how retrieved context is used.

## Troubleshooting

- If the app logs that it found 0 files:
  - Verify `Docs` directory exists in the project root (not only in bin).
  - Run from project root, or pass an explicit absolute path to IndexDocuments in Program.cs for testing.

- If query returns no answer:
  - Check the indexed files and scores printed during AskAsync. Low similarity scores mean the embedding method didn't match the query well (expected with SHA256 embeddings).
  - Try phrasing the query similarly to the text in documents (the simple extractor requires similar words to match).

## Next steps (when ready to use real models)

1. Replace EmbeddingService with a real embeddings API client (e.g., OpenAI or Azure OpenAI) and return the true embedding vector type.
2. Optionally add document chunking and batching to support long documents.
3. Use an actual model (chat/completion) to generate answers from the assembled prompt instead of the simulated responder.
4. Add caching, persistence (disk or DB) for the vector store, and better retrieval (FAISS, Annoy, or an external vector database) for large datasets.

## Security and configuration

- When you integrate real cloud models, do not hard-code API keys. Use environment variables or a secrets store and avoid checking keys into source control.

## Summary

This demo is intentionally simple and local. It shows the fundamental RAG loop: embed, index, retrieve, and generate. Replace the embedding and generation pieces with real services as you become comfortable with the flow.

If you want, I can add:
- a step-by-step tutorial (scripts) to switch to OpenAI/Azure embeddings,
- document chunking code, or
- persistent storage for the vector store.
