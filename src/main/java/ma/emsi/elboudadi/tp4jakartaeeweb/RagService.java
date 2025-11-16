package ma.emsi.elboudadi.tp4jakartaeeweb;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
// Import for the corrected class name (EmbeddingStoreContentRetriever)
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import ma.emsi.elboudadi.tp4jakartaeeweb.llm.Assistant;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Named;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

@Named
@ApplicationScoped
public class RagService {

    private Assistant assistant;
    private final String LLM_KEY = System.getenv("GEMINI-API-KEY");

    @PostConstruct
    public void init() {
        try {
            if (LLM_KEY == null || LLM_KEY.isEmpty()) {
                System.err.println("GEMINI_API_KEY not set.");
                return;
            }

            // 1️⃣ Chat model (Gemini)
            ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                    .apiKey(LLM_KEY)
                    .modelName("gemini-pro")
                    .logRequestsAndResponses(true)
                    .build();

            // 2️⃣ Embedding model
            EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

            // 3️⃣ In-memory store for your PDFs
            EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();

            // 4️⃣ Load and index PDFs
            ApacheTikaDocumentParser parser = new ApacheTikaDocumentParser();
            DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);

            Path iaPath = Paths.get(getClass().getResource("/docs/machinelearning.pdf").toURI());
            Path nonIaPath = Paths.get(getClass().getResource("/docs/ERP.pdf").toURI());

            // Process IA document
            Document iaDoc = FileSystemDocumentLoader.loadDocument(iaPath, parser);
            List<TextSegment> iaSegments = splitter.split(iaDoc);
            List<Embedding> iaEmbeddings = embeddingModel.embedAll(iaSegments).content();
            store.addAll(iaEmbeddings, iaSegments);

            // Process non-IA document
            Document nonIaDoc = FileSystemDocumentLoader.loadDocument(nonIaPath, parser);
            List<TextSegment> nonIaSegments = splitter.split(nonIaDoc);
            List<Embedding> nonIaEmbeddings = embeddingModel.embedAll(nonIaSegments).content();
            store.addAll(nonIaEmbeddings, nonIaSegments);

            // 5️⃣ Retriever from store
            // 💡 FIX: Use the builder pattern to pass the EmbeddingStore and EmbeddingModel
            EmbeddingStoreContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                    .embeddingStore(store)
                    .embeddingModel(embeddingModel)
                    .maxResults(3) // Retrieve the top 3 most relevant segments
                    .build();

            // 6️⃣ Build assistant with RAG
            // The contentRetriever method is correct from the previous step's fix.
            this.assistant = AiServices.builder(Assistant.class)
                    .chatModel(chatModel)
                    .contentRetriever(retriever)
                    .build();

            System.out.println("RAG service initialized successfully.");

        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Error initializing RAG: " + e.getMessage());
        }
    }

    public String chat(String userMessage) {
        if (assistant == null) {
            return "RAG service not initialized (missing API key or error at startup).";
        }
        return assistant.chat(userMessage);
    }
}