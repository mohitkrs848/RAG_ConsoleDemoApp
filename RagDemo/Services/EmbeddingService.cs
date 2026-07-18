using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

/// <summary>
/// EmbeddingService (TF-IDF based)
///
/// This class provides a simple, local TF-IDF vectorizer to produce embeddings that are
/// text-aware and allow semantic-like similarity for small demos. It builds a vocabulary
/// from a corpus (via Fit) and produces TF-IDF vectors for documents and queries.
///
/// Note: This is still a toy vectorizer and not a replacement for model-based embeddings,
/// but it generally performs much better than a hash-based mapping for matching similar text.
/// </summary>
public class EmbeddingService
{
    private readonly HashSet<string> _stopWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "the","is","in","at","which","on","a","an","and","or","of","to","for","with","by","that","this","it","as","are"
    };

    private Dictionary<string, int> _vocab = new();
    private double[] _idf = Array.Empty<double>();

    /// <summary>
    /// Fit the vectorizer on a corpus of documents. This builds the vocabulary and idf values.
    /// Call this before transforming documents/queries for meaningful vectors.
    /// </summary>
    public void Fit(IEnumerable<string> documents, int maxVocabSize = 2000)
    {
        var docs = documents?.ToList() ?? new List<string>();
        var df = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var doc in docs)
        {
            var terms = Tokenize(doc).Distinct(StringComparer.OrdinalIgnoreCase);
            foreach (var t in terms)
            {
                if (_stopWords.Contains(t)) continue;
                df[t] = df.TryGetValue(t, out var c) ? c + 1 : 1;
            }
        }

        // Select top terms by document frequency to build a stable vocabulary
        _vocab = df.OrderByDescending(kv => kv.Value)
                   .Take(maxVocabSize)
                   .Select((kv, idx) => (kv.Key, idx))
                   .ToDictionary(x => x.Key, x => x.idx, StringComparer.OrdinalIgnoreCase);

        int n = docs.Count;
        _idf = new double[_vocab.Count];
        for (int i = 0; i < _idf.Length; i++) _idf[i] = 1.0; // default

        foreach (var kv in _vocab)
        {
            var term = kv.Key;
            var idx = kv.Value;
            df.TryGetValue(term, out var docFreq);
            // idf smoothing
            _idf[idx] = Math.Log((n + 1.0) / (docFreq + 1.0)) + 1.0;
        }
    }

    /// <summary>
    /// GenerateAsync produces a TF-IDF embedding for the input text using the fitted vocabulary.
    /// If Fit was not called, it will build a tiny vocabulary from the text itself.
    /// </summary>
    public Task<float[]> GenerateAsync(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) text = string.Empty;

        if (_vocab.Count == 0)
        {
            // Fallback: build vocab from this single document so transform works
            Fit(new[] { text }, maxVocabSize: 1000);
        }

        var vec = new double[_vocab.Count];
        var terms = Tokenize(text);
        var tf = new Dictionary<int, double>();
        foreach (var t in terms)
        {
            if (!_vocab.TryGetValue(t, out var idx)) continue;
            tf[idx] = tf.TryGetValue(idx, out var c) ? c + 1 : 1;
        }

        // compute tf-idf
        foreach (var kv in tf)
        {
            var idx = kv.Key;
            var freq = kv.Value;
            vec[idx] = freq * _idf[idx];
        }

        // normalize to unit length
        var norm = Math.Sqrt(vec.Sum(x => x * x));
        var result = new float[vec.Length];
        if (norm > 0)
        {
            for (int i = 0; i < vec.Length; i++) result[i] = (float)(vec[i] / norm);
        }

        return Task.FromResult(result);
    }

    private IEnumerable<string> Tokenize(string text)
    {
        if (string.IsNullOrEmpty(text)) yield break;
        // simple tokenization: lowercase, remove punctuation, split on whitespace
        var cleaned = Regex.Replace(text.ToLowerInvariant(), "[^a-z0-9\\s]", " ");
        foreach (var tok in cleaned.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries))
        {
            if (tok.Length <= 1) continue;
            yield return tok;
        }
    }
}
