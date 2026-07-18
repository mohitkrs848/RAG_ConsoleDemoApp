using System;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// VectorStoreService
///
/// Simple in-memory vector store for demo purposes.
/// Stores tuples of (Id, Embedding, Text) and provides a naive search by cosine similarity.
///
/// Notes for newcomers:
/// - This is intentionally basic and not optimized for production. For large datasets use an
///   approximate nearest neighbor library (FAISS, Annoy) or a hosted vector DB.
/// - Embeddings are float[] in this demo. A production system may use different types or storage.
public class VectorStoreService
{
    // Internal storage for records. Id is typically a path or document identifier.
    private readonly List<(string Id, float[] Embedding, string Text)> _records = new();

    /// <summary>
    /// Add
    ///
    /// Adds a document to the in-memory store.
    /// Parameters:
    /// - id: unique identifier for the document (e.g., file path)
    /// - embedding: float[] produced by the embedding service
    /// - text: the original document text
    ///
    /// Rationale: In production you'd persist embeddings and metadata to disk or an external DB.
    /// This demo keeps everything in memory to simplify the flow.
    /// </summary>
    public void Add(string id, float[] embedding, string text)
    {
        _records.Add((id, embedding, text));
    }

    /// <summary>
    /// Search
    ///
    /// Parameters:
    /// - queryEmbedding: embedding vector for the user's query
    /// - topK: number of top results to return
    ///
    /// Returns: IEnumerable of (Id, Embedding, Text, Score) ordered by descending similarity.
    ///
    /// Implementation details:
    /// - Computes cosine similarity between the query and each stored vector.
    /// - Orders results by score and returns topK.
    /// - Scores are in range [-1, 1], higher is more similar.
    /// </summary>
    public IEnumerable<(string Id, float[] Embedding, string Text, double Score)> Search(float[] queryEmbedding, int topK = 3)
    {
        // Simple cosine similarity search
        return _records
            .Select(r => (r.Id, r.Embedding, r.Text, Score: CosineSimilarity(queryEmbedding, r.Embedding)))
            .OrderByDescending(x => x.Score)
            .Take(topK);
    }

    /// <summary>
    /// CosineSimilarity
    ///
    /// Computes cosine similarity between two vectors.
    /// - Uses the minimum length to avoid out-of-range access if vectors are different sizes.
    /// - Returns 0 when either norm is zero to avoid divide-by-zero.
    ///
    /// In production:
    /// - Ensure consistent embedding dimensions or pad/trim appropriately.
    /// - Use optimized math libraries for performance.
    /// </summary>
    private static double CosineSimilarity(float[] v1, float[] v2)
    {
        double dot = 0, norm1 = 0, norm2 = 0;
        int len = Math.Min(v1.Length, v2.Length);
        for (int i = 0; i < len; i++)
        {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }
        if (norm1 == 0 || norm2 == 0) return 0;
        return dot / (Math.Sqrt(norm1) * Math.Sqrt(norm2));
    }
}

