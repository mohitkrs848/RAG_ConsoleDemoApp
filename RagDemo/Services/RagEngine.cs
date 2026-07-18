using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public class RagEngine
{
    private readonly EmbeddingService _embedding;
    private readonly VectorStoreService _store;

    public RagEngine()
    {
        _embedding = new EmbeddingService();
        _store = new VectorStoreService();
    }

    // Create a document file (helper) and optionally place it in the specified folder.
    public void CreateDocument(string fileName, string content, string folderPath = "Docs")
    {
        if (string.IsNullOrWhiteSpace(fileName)) throw new ArgumentException("fileName required", nameof(fileName));

        Directory.CreateDirectory(folderPath);
        var path = Path.Combine(folderPath, fileName);
        File.WriteAllText(path, content ?? string.Empty);
    }

    // Index all .txt files from the provided folderPath. This method resolves relative paths
    // by searching the current working directory and its parent folders to find a matching
    // directory name (case-insensitive). If the directory is not found, it will be created
    // under the current working directory. The method searches for .txt files in the
    // target directory and its subdirectories.
    public async Task IndexDocuments(string folderPath)
    {
        var requested = string.IsNullOrWhiteSpace(folderPath) ? "Docs" : folderPath;

        // Determine a sensible start directory. Prefer the project root (where the .csproj lives)
        // to avoid working directory issues when running from bin/Debug/... folders.
        var cwd = Directory.GetCurrentDirectory();
        var baseDir = AppContext.BaseDirectory ?? cwd;
        var projectRoot = FindProjectRoot(baseDir) ?? FindProjectRoot(cwd) ?? cwd;

        string folder;
        if (Path.IsPathRooted(requested))
        {
            folder = requested;
        }
        else
        {
            // Prefer path relative to project root so Docs in the repository root is found even when running from bin.
            folder = Path.GetFullPath(Path.Combine(projectRoot, requested));
        }

        if (!Directory.Exists(folder))
        {
            // Try to find any directory with the requested name under the project root (case-insensitive)
            var deepMatch = Directory.GetDirectories(projectRoot, requested, SearchOption.AllDirectories)
                .FirstOrDefault(d => string.Equals(Path.GetFileName(d), requested, StringComparison.OrdinalIgnoreCase));
            if (deepMatch != null)
            {
                folder = deepMatch;
            }
            else
            {
                // Try parent search from current working directory as a last resort
                var parentMatch = FindDirectoryInParents(cwd, requested);
                if (parentMatch != null)
                {
                    folder = parentMatch;
                }
                else
                {
                    // Create the folder under project root so users can add docs
                    Directory.CreateDirectory(folder);
                }
            }
        }

        // First look for files directly under the folder, then include subdirectories if none found.
        var files = Directory.GetFiles(folder, "*.txt", SearchOption.TopDirectoryOnly);
        if (files.Length == 0)
        {
            files = Directory.GetFiles(folder, "*.txt", SearchOption.AllDirectories);
        }

        Console.WriteLine($"Indexing folder: {folder} - found {files.Length} .txt files");

        foreach (var file in files)
        {
            string text = File.ReadAllText(file);
            var embedding = await _embedding.GenerateAsync(text);
            _store.Add(file, embedding, text);
            Console.WriteLine($"Indexed file: {file} (length: {text.Length} chars)");
        }

        // After indexing all documents, fit the embedding vectorizer on the corpus so
        // subsequent query embeddings are comparable to document embeddings.
        // This step is required for the TF-IDF based EmbeddingService.
        try
        {
            var allTexts = files.Select(f => File.ReadAllText(f)).ToList();
            _embedding.Fit(allTexts);
            Console.WriteLine("EmbeddingService: fitted vocabulary on indexed documents.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: failed to fit embedding service: {ex.Message}");
        }
    }

    private static string? FindProjectRoot(string start)
    {
        try
        {
            var dir = new DirectoryInfo(start);
            while (dir != null)
            {
                // Look for a .csproj file in this directory
                var csproj = dir.GetFiles("*.csproj", SearchOption.TopDirectoryOnly).FirstOrDefault();
                if (csproj != null) return dir.FullName;
                dir = dir.Parent;
            }
        }
        catch
        {
            // ignore
        }
        return null;
    }

    private static string? FindDirectoryInParents(string startDirectory, string folderName)
    {
        var dir = new DirectoryInfo(startDirectory);
        while (dir != null)
        {
            var match = dir.GetDirectories().FirstOrDefault(d => string.Equals(d.Name, folderName, StringComparison.OrdinalIgnoreCase));
            if (match != null) return match.FullName;
            dir = dir.Parent;
        }
        return null;
    }

    public async Task<string> AskAsync(string query)
    {
        var queryEmbedding = await _embedding.GenerateAsync(query);
        var results = _store.Search(queryEmbedding, 3).ToList();

        Console.WriteLine($"Found {results.Count} matching documents.");
        for (int i = 0; i < results.Count; i++)
        {
            var r = results[i];
            Console.WriteLine($"Result {i + 1}: {Path.GetFileName(r.Id)} (score: {r.Score:F4})");
        }

        string context = string.Join("\n", results.Select(r => r.Text));

        // Try to extract a direct answer from the most relevant documents using simple Q/A pattern matching.
        foreach (var r in results)
        {
            var extracted = ExtractAnswerFromDocument(r.Text, query);
            if (!string.IsNullOrWhiteSpace(extracted))
            {
                return await Task.FromResult(extracted);
            }
        }

        // Fallback: return the composed prompt as a simulated model input/answer.
        string prompt = $"Use this context:\n{context}\n\nQuestion: {query}";
        var sb = new StringBuilder();
        sb.AppendLine("[LocalResponder]");
        sb.AppendLine(prompt);
        return await Task.FromResult(sb.ToString());
    }

    // Simple extractor: looks for Q: / A: style pairs or question/answer lines in the document.
    private static string? ExtractAnswerFromDocument(string docText, string query)
    {
        if (string.IsNullOrWhiteSpace(docText) || string.IsNullOrWhiteSpace(query)) return null;

        var lines = docText.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(l => l.Trim()).ToArray();

        // Normalize query for simple substring matching
        var qnorm = NormalizeText(query);

        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            // Match lines that start with Q: or contain the question text
            if (line.StartsWith("Q:", StringComparison.OrdinalIgnoreCase) || line.EndsWith("?"))
            {
                var content = line.StartsWith("Q:", StringComparison.OrdinalIgnoreCase) ? line.Substring(2).Trim() : line;
                if (NormalizeText(content).Contains(qnorm))
                {
                    // Look for an A: line after this
                    for (int j = i + 1; j < Math.Min(lines.Length, i + 6); j++)
                    {
                        var next = lines[j];
                        if (next.StartsWith("A:", StringComparison.OrdinalIgnoreCase))
                        {
                            return next.Substring(2).Trim();
                        }
                        // also accept plain sentence as answer
                        if (!next.StartsWith("Q:", StringComparison.OrdinalIgnoreCase))
                        {
                            return next;
                        }
                    }
                }
            }
            else
            {
                // If the line contains the question text directly
                if (NormalizeText(line).Contains(qnorm))
                {
                    // try next lines for answer
                    if (i + 1 < lines.Length) return lines[i + 1];
                }
            }
        }

        return null;
    }

    private static string NormalizeText(string s)
    {
        return new string(s?.ToLowerInvariant().Where(c => !char.IsPunctuation(c)).ToArray() ?? Array.Empty<char>()).Trim();
    }
}
