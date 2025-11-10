package ma.emsi.elboudadi.tp4jakartaeeweb.llm;

/**
 * Interface utilisée par LangChain4j pour définir l'interaction de base avec le LLM.
 * LangChain4j fournit automatiquement l'implémentation (proxy).
 */
public interface Assistant {
    /**
     * Envoie un prompt à l'assistant et reçoit la réponse.
     * LangChain4j gère l'historique (ChatMemory) et les SystemMessages.
     * @param prompt La question de l'utilisateur.
     * @return La réponse du LLM.
     */
    String chat(String prompt);
}