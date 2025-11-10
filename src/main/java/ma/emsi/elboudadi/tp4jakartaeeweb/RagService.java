package ma.emsi.elboudadi.tp4jakartaeeweb;


import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.bge.small.en.v15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.WebSearchContentRetriever;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import dev.langchain4j.retriever.ContentRetriever;
import dev.langchain4j.retriever.impl.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.RetrievalAugmentor;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.service.AiServices;

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
    private final String LLM_KEY = System.getenv("GEMINI_API_KEY");
    private final String TAVILY_KEY = System.getenv("TAVILY_API_KEY");

    @PostConstruct
    public void init() {
        if (LLM_KEY == null || LLM_KEY.isEmpty()) {
            System.err.println("❌ GEMINI_API_KEY not set. RAG service will not function.");
            return;
        }

        // 1. Models
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(LLM_KEY)
                .modelName("gemini-2.0-flash") // or "gemini-2.5-flash" if available
                .logRequestsAndResponses(true)
                .build();

        EmbeddingModel embeddingModel = new BgeSmallEnV15EmbeddingModel();

        // 2. Document ingestion
        EmbeddingStore<TextSegment> iaEmbeddingStore = new InMemoryEmbeddingStore<>();
        EmbeddingStore<TextSegment> nonIaEmbeddingStore = new InMemoryEmbeddingStore<>();

        try {
            Path iaDocPath = Paths.get(getClass().getResource("/docs/docia.pdf").toURI());
            Path nonIaDocPath = Paths.get(getClass().getResource("/docs/doc-nonia.pdf").toURI());

            DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
            ApacheTikaDocumentParser parser = new ApacheTikaDocumentParser();

            // IA document
            Document iaDocument = FileSystemDocumentLoader.loadDocument(iaDocPath, parser);
            List<TextSegment> iaSegments = splitter.split(iaDocument);
            List<Embedding> iaEmbeddings = embeddingModel.embedAll(iaSegments).content();
            iaEmbeddingStore.addAll(iaEmbeddings, iaSegments);

            // Non-IA document
            Document nonIaDocument = FileSystemDocumentLoader.loadDocument(nonIaDocPath, parser);
            List<TextSegment> nonIaSegments = splitter.split(nonIaDocument);
            List<Embedding> nonIaEmbeddings = embeddingModel.embedAll(nonIaSegments).content();
            nonIaEmbeddingStore.addAll(nonIaEmbeddings, nonIaSegments);

            // 3. Content retrievers
            ContentRetriever iaDocRetriever = EmbeddingStoreContentRetriever.builder()
                    .embeddingStore(iaEmbeddingStore)
                    .embeddingModel(embeddingModel)
                    .maxResults(2)
                    .minScore(0.5)
                    .build();

            WebSearchEngine tavilyEngine = TavilyWebSearchEngine.builder()
                    .apiKey(TAVILY_KEY)
                    .build();

            ContentRetriever webRetriever = WebSearchContentRetriever.builder()
                    .webSearchEngine(tavilyEngine)
                    .maxResults(3)
                    .build();

            // 4. Query Router (combines document and web retrievers)
            ContentRetriever combinedRetriever = DefaultQueryRouter.builder()
                    .contentRetriever(iaDocRetriever)
                    .contentRetriever(webRetriever)
                    .build();

            // 5. Retrieval Augmentor
            RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                    .contentRetriever(combinedRetriever)
                    .build();

            // 6. Assistant setup
            this.assistant = AiServices.builder(Assistant.class)
                    .chatModel(chatModel)
                    .retrievalAugmentor(retrievalAugmentor)
                    .build();

        } catch (Exception e) {
            System.err.println("⚠️ Error initializing RagService: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public String chat(String userMessage) {
        if (assistant == null) {
            return "❌ Erreur : Le service RAG n'a pas pu être initialisé (vérifiez les clés API).";
        }
        return assistant.chat(userMessage);
    }
}
