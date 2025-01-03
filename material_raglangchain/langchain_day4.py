# ********** IMPORT FRAMEWORKS **************
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory

# ********** IMPORT LIBRARY **************
import os
from dotenv import load_dotenv
import tiktoken

def get_model(model_name: str) -> ChatOpenAI:
        key = os.getenv("OPENAI_API_KEY")
        model = ChatOpenAI(
            api_key=key,
            model_name=model_name,
            temperature=0.9,
            max_tokens=256
        )
        return model
    
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
model_llm = "gpt-4o-mini"
model = ChatOpenAI(model=model_llm, temperature=0.9, max_tokens=512)

buffer_memory: ConversationBufferMemory = ConversationBufferMemory(return_messages=True)  # Full conversation history
summary_memory: ConversationSummaryMemory = ConversationSummaryMemory(llm=model)  # Running summary 

# ****** Funtion to count tokens ******
# def count_tokens(text: str, model_name: str) -> int:
#     """_summary_
#     Count the number of tokens in a given text using the tiktoken library
    
#     Arguments:
#         text (str): prompt format
#         model_name (str): llm model name

#     Returns:
#         int: count tokens
#     """
#     enc = tiktoken.encoding_for_model(model_name)
#     tokens = enc.encode(text)
#     return len(tokens)



# ****** Funtion to count tokens ******
def create_prompt_template() -> ChatPromptTemplate:
    """_summary_
    Create a ChatPromptTemplate with a system message and a placeholder for the problem.
    
    Arguments:
        template(str): Template prompt
    
    Returns:
        _type_: chat prompt template
    """
    template = """
    Please reason step by step to solve the problem and put your answer in //boxed. 
    """
    return ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="problem")
    ])

def invoke_chain(prompt: ChatPromptTemplate, model, question: str) -> str:
    """_summary_
    Invoke the chain with a given question and return the clean response content.
    Arguments:
        prompt: template prompt
        model: llm model
        question: problem user input
        
    Returns:
        _type_: llm response content
    """
    
    # Input validation
    if not isinstance(prompt, ChatPromptTemplate):
        raise TypeError("The 'prompt' argument must be of type ChatPromptTemplate.")
    if not hasattr(model, 'invoke'): 
        raise AttributeError("The 'model' argument must have an 'invoke' method.")
    if not isinstance(question, str) or not question.strip(): 
        raise ValueError("The 'question' argument must be a non-empty string.")
    
    try:
        response = prompt | model
        result = response.invoke({"problem": [HumanMessage(content=question)]})
        return result.content if hasattr(result, 'content') else ""
    except Exception as e:
        raise RuntimeError(f"An error occurred while invoking the chain: {e}")

    
def qa_system(question):
    """_summary_
    Main execution function for running the chain and token count.
    Returns:
        Arguments: 
            question (str): prompt question 
        Returns:
            _type_: _dictionary_ question, answer, tokens_in, tokens_out, cost
    """
    prompt = create_prompt_template()
    clean_answer = ""
    tokens_in, tokens_out = 0, 0
    
    with get_openai_callback() as cb:
        clean_answer = invoke_chain(prompt, model, question)
        tokens_in = cb.prompt_tokens
        tokens_out = cb.completion_tokens
    
    # buffer_memory.save_context(inputs={"input": question}, outputs={"output": clean_answer})
    # summary_memory.save_context(inputs={"input": question}, outputs={"output": clean_answer})
        
    return {
        "question": question,
        "answer": clean_answer,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }

if __name__ == "__main__":
    question = "Simplify the expression 4m+5+2m-1"
    output = qa_system(question)
    print(output)