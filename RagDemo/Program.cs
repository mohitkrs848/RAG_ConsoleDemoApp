class Program
{
    static async Task Main()
    {
        var rag = new RagEngine();

        // ensure docs folder exists
        Directory.CreateDirectory("Docs");
        await rag.IndexDocuments("Docs");

        Console.WriteLine("Ask a question:");
        string query = Console.ReadLine();

        string answer = await rag.AskAsync(query);
        Console.WriteLine($"Answer: {answer}");
    }
}
