# ********** IMPORT FRAMEWORKS **************
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_community.callbacks import get_openai_callback
from langchain.chains.conversation.memory import ConversationSummaryMemory

# ********** IMPORT LIBRARY **************
import os
from dotenv import load_dotenv
import tiktoken
from typing import Dict, List

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

model_llm = "gpt-4o-mini"
model = ChatOpenAI(model=model_llm, temperature=0.9, max_tokens=512)


summary_memory = ConversationSummaryMemory(llm=model) 


def create_prompt_template() -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate with a system message and a MessagesPlaceholder for the previous messages.
    """
    system_message = """
    Please reason step by step to solve the problem and put your answer in //boxed. 
    """
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")  
    ])

def invoke_chain(prompt: ChatPromptTemplate, model: ChatOpenAI, question: str, chat_history: List[Dict[str, str]]) -> str:
    """
    Invoke the chain with a given question and return the clean response content.
    Args:
        prompt (ChatPromptTemplate): The chat prompt template for the assistant.
        model (ChatOpenAI): The language model to process the input.
        question (str): The user's input question.
        chat_history (List[Dict[str, str]]): Existing chat history.

    Returns:
        str: The LLM's response to the user's question.
    """
    if not isinstance(prompt, ChatPromptTemplate):
        raise TypeError("The 'prompt' argument must be of type ChatPromptTemplate.")
    if not hasattr(model, 'invoke'): 
        raise AttributeError("The 'model' argument must have an 'invoke' method.")
    if not isinstance(question, str) or not question.strip(): 
        raise ValueError("The 'question' argument must be a non-empty string.")
    
    try:
        past_messages = inverse_construct_history(chat_history)  
        response = prompt | model
        result = response.invoke({"history": past_messages, "question": question})
        return result.content if hasattr(result, 'content') else ""
    except Exception as e:
        raise RuntimeError(f"An error occurred while invoking the chain: {e}")


def construct_history(memory_messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """
    Construct message history into list of dictionaries (for serialization).
    Args:
        memory_messages (List[BaseMessage]): Memory messages

    Returns:
        List[Dict[str, str]]: A list of messages as dictionaries with 'human' and 'AI' keys.
    """
    parsed_messages = []
    for message in memory_messages:
        if isinstance(message, HumanMessage):
            parsed_messages.append({"human": message.content})
        elif isinstance(message, AIMessage):
            parsed_messages.append({"AI": message.content})
    return parsed_messages


def inverse_construct_history(parsed_messages: List[Dict[str, str]]) -> List[BaseMessage]:
    """
    Convert chat history back into HumanMessage and AIMessage.
    Args:
        parsed_messages (List[Dict[str, str]]): A list of messages in dict format.

    Returns:
        List[BaseMessage]: List of HumanMessage and AIMessage objects.
    """
    restored_messages: List[BaseMessage] = []
    for message in parsed_messages:
        if "human" in message:
            restored_messages.append(HumanMessage(content=message["human"]))
        elif "AI" in message:
            restored_messages.append(AIMessage(content=message["AI"]))
    return restored_messages


def qa_system(question: str, ch: List[Dict[str, str]]) -> Dict[str, any]:
    """
    Runs the QA system for the provided question, handles memory, and returns the response.
    Args:
        question (str): The user's question.
        ch (list[dict]): The chat history in list[dict] format.

    Returns:
        dict: The final response, tokens, and memory details.
    """
    prompt = create_prompt_template()
    with get_openai_callback() as cb:
        answer = invoke_chain(prompt, model, question, ch)
        tokens_in = cb.prompt_tokens
        tokens_out = cb.completion_tokens
    
    ch.append({"human": question, "AI": answer})
    
    summary_memory.save_context(inputs={"input": question}, outputs={"output": answer})
    
    return {
        "question": question,
        "answer": answer,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "chat_history": ch,
        "summary_history": summary_memory.load_memory_variables(inputs={})['history'],
    }


if __name__ == "__main__":
    ch = []  
    while True:
        input_message = str(input("\n(type 'exit' to end): ")).lower()
        if input_message == "exit":
            break 
        output = qa_system(input_message, ch)
        print("\nResponse:", output)
