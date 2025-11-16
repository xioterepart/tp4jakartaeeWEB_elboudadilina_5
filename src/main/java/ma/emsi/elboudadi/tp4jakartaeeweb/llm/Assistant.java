package ma.emsi.elboudadi.tp4jakartaeeweb.llm;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;

public interface Assistant {
    @SystemMessage("You are a helpful assistant. Your knowledge base includes documents on AI and you can also search the web for up-to-date information.")
    @UserMessage("{userMessage}")
    String chat(String userMessage);
}