package ma.emsi.elboudadi.tp4jakartaeeweb.llm;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.rag.RetrievalAugmentor;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

@ApplicationScoped
public class LlmClient {

    private String systemRole;
    private ChatMemory chatMemory;

    @Inject
    private RagTavily RagTavily;

    private ChatModel model;

    /** Constructeur : initialise le modèle Gemini */
    public LlmClient() {
        String apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("Clé API GEMINI_KEY manquante !");
        }

        this.model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.7)
                .build();

        this.chatMemory = MessageWindowChatMemory.withMaxMessages(10);
    }

    /**
     * Définit le rôle système et réinitialise la mémoire.
     */
    public void setSystemRole(String role) {
        this.systemRole = role;
        chatMemory.clear();
        if (role != null && !role.isBlank()) {
            chatMemory.add(SystemMessage.from(role));
        }
    }

    /**
     * Envoie un prompt et renvoie une réponse enrichie par le RAG + Tavily.
     */
    public String ask(String prompt) {
        RetrievalAugmentor augmentor = RagTavily.getAugmentor(); // 🔹 Récupération du RAG

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(augmentor) // 🔹 Activation du RAG
                .build();

        return assistant.chat(prompt);
    }

    public String getSystemRole() {
        return systemRole;
    }
}