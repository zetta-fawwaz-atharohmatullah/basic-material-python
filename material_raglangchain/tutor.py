# ********** IMPORT FRAMEWORKS **************
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.callbacks import get_openai_callback
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel,  RunnableBranch
from langchain_core.output_parsers import StrOutputParser

# ********** IMPORT LIBRARY **************
import os
from dotenv import load_dotenv
import tiktoken
import re
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL


load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
model_llm = "gpt-4o-mini"
model = ChatOpenAI(model=model_llm, temperature=0.9)



# CHAIN BRANCHING
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
         ("system", "You are a helpful customer service."),
        ("human",
          "Generate a thank you note for this positive feedback: {feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful customer service."),
        ("human",
         "Generate a response addressing this negative feedback: {feedback}."),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful customer service."),
        (
            "human",
            "Generate a request for more details for this neutral feedback: {feedback}.",
        ),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful customer service."),
        (
            "human",
            "Generate a message to escalate this feedback to a human agent: {feedback}.",
        ),
    ]
)

# Define the feedback classification template
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
         "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."),
    ]
)


branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()
            
    ),
     (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()  # Negative feedback chain
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()  # Neutral feedback chain
    ),
    escalate_feedback_template | model | StrOutputParser()
)

classification_chain = classification_template | model | StrOutputParser()
chain = classification_chain | branches

review =  str(input())
result = chain.invoke({"feedback": review})

# Output the result
print(result)






















# vagueness_detector = ChatPromptTemplate.from_messages(
#     [
#       ("system", "You are an expert in identifying if questions are too vague."),
#       ("human", "Is this question too vague: '{question}'? Respond with 'Yes' or 'No'.")
#     ]
# )

# clarification_generator = ChatPromptTemplate.from_messages(
#     [
#       ("system", "You are an expert at asking clarification questions."),
#       ("human", "If the user's question is too vague, ask for clarification: '{question}'")
#     ]
# )

# def update_question(original_question, clarification):
#     """Combine the original question with the user's clarification."""
#     return f"{original_question} ({clarification})"


# final_answer_generator = ChatPromptTemplate.from_messages(
#     [
#       ("system", "You are an expert AI assistant."),
#       ("human", "Provide a detailed, logical answer to this question: '{question}'")
#     ]
# )


# def iterative_question(question):
#     is_vague = (
#         RunnableLambda(lambda x: {'question': x['question']}) 
#         | vagueness_detector 
#         | model 
#         | StrOutputParser()
#     ).invoke({'question': question})
    
#     while True:
#         print(f"🤖 Current Question: {question}")
#         # Ask Clarification
#         clarification = (
#             RunnableLambda(lambda x: {'question': x['question']}) 
#             | clarification_generator 
#             | model 
#             | StrOutputParser()
#         ).invoke({'question': question})
#         print(f"🤔 Is this question vague? {is_vague}")
        
#         if "No" in is_vague:
#             break
#         # Simulate user input for now (in real usage, you'd capture the user response)
       
#         user_clarification = input("👤 User Response: ")
        
#         # Update Question
#         question = update_question(question, user_clarification)
    
#     final_answer = (
#         RunnableLambda(lambda x: {'question': x['question']}) 
#         | final_answer_generator 
#         | model 
#         | StrOutputParser()
#     ).invoke({'question': question})
    
#     return final_answer
        
# result = iterative_question("can you help me")
# print(result)

# CHAIN PARALLEL
# messages = ChatPromptTemplate.from_messages(
#     [
#       ("system", "You are an expert tech product reviewer."),
#       ("human", "List the main features of the product {product}"),
#     ]
# )

# def analyze_pros(features):
#     pros_template = ChatPromptTemplate. from_messages(
#         [
#         ("system", "You are an expert product reviewer."),
#         ("human",
#         "Given these features: {features}, list the pros of these features.",)
#         ]
#     )

#     return pros_template.format_prompt (features=features)

# def analyze_cons(features):
#     pros_template = ChatPromptTemplate. from_messages(
#         [
#         ("system", "You are an expert product reviewer."),
#         ("human",
#         "Given these features: {features}, list the cons of these features.",)
#         ]
#     )

#     return pros_template.format_prompt(features=features)

# def combine_pros_cons(pros, cons):
#     return f"Pros:\n{pros}\n\nCons:\n{cons}"

# pros_branch = (
#     RunnableLambda(lambda x: analyze_pros(x)) 
#     | model
#     | StrOutputParser()
# )

# cons_branch = (
#     RunnableLambda(lambda x: analyze_cons(x)) 
#     | model
#     | StrOutputParser()
# )


# chain = (
#     messages 
#     | model 
#     | StrOutputParser()
#     | RunnableParallel(branches = {"pros":pros_branch, "cons":cons_branch})
#     | RunnableLambda(lambda x: combine_pros_cons(x['branches']['pros'], x['branches']['cons']))
# )

# result = chain.invoke({'product': 'macbook air m3 16gb ram' })

# print(result)


















#CHAIN EXTENDER
# messages = ChatPromptTemplate.from_messages(
#     [
#       ("system", "you are a comedian tell joke abot {topic}"),
#       ("human", "tell me {joke_count} jokes"),
#     ]
# )


# uppercaseoutput = RunnableLambda(lambda x: x.upper())
# count_words = RunnableLambda(lambda x: f"word counts: {len(x.split())}\n{x}")

# #create combine chain
# chain = messages | model | StrOutputParser() | uppercaseoutput | count_words
# result = chain.invoke({"topic":"AI", "joke_count":3})
# print(result)


# format_prompt = RunnableLambda(lambda x: messages.format_prompt(**x))
# invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
# parse_output = RunnableLambda(lambda x: x.content)

# chain = RunnableSequence(first = format_prompt, middle=[invoke_model], last=parse_output)
# response = chain.invoke({"topic":"AI", "joke_count":3})
# print(response)

# chat_history = []
# system_messages = SystemMessage(content="you are a helpful AI assitant")
# chat_history.append(system_messages)

# while True:
#     query = input("You: ")
#     if query.lower() == "exit":
#         break
#     chat_history.append(HumanMessage(content=query))
    
#     result = model.invoke(chat_history)
#     response = result.content
#     chat_history.append(AIMessage(content=response))
    
#     print(f"AI: {response}")

# print("message history")
# print(chat_history)










