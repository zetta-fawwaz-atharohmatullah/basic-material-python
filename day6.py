# ********** IMPORT FRAMEWORKS **************
from langchain_astradb import AstraDBVectorStore
from astrapy.info import CollectionVectorServiceOptions
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Union

# ********** IMPORT LIBRARY **************
from day6_load_doc import load_document_update  
from dotenv import load_dotenv
from astrapy import DataAPIClient
import tiktoken
import os 
import logging
from material_raglangchain.langchain_day5 import inverse_construct_history
from day6_format_pdf import process_document

load_dotenv(override=True)

ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE") 
ASTRA_DB_KEYCOLLECTION = os.getenv("ASTRA_DB_KEYCOLLECTION")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
model_llm = "gpt-4o-mini"
model = ChatOpenAI(model=model_llm, temperature=0.9, max_tokens=356)
    
vector_store = AstraDBVectorStore(
    collection_name=ASTRA_DB_KEYCOLLECTION,
    embedding=embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_KEYSPACE,
)

# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("process.log"),
#         logging.StreamHandler()
#     ]
# )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

#client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
# db = client.get_database_by_api_endpoint(ASTRA_DB_API_ENDPOINT)
# collection_name = "test_push_one"
# collection = db.get_collection(name= collection_name, namespace="default_keyspace") 
#collection.delete_many(filter={"metadata.document_id":"1234-5678-uuid"})

def count_tokens(text: str, model_name: str = "text-embedding-ada-002") -> int:
    """_summary_
    Count the number of tokens in a given text using tiktoken.
    Returns:
        _type_: _description_
    """
    try:
        enc = tiktoken.encoding_for_model(model_name)
        tokens = enc.encode(text)
        logging.debug(f"Token count for text: {len(tokens)} tokens")
        return len(tokens)
    except Exception as e:
        logging.error(f"Error while counting tokens: {e}")
        return 0

def split_text_into_chunks(page_content: str, metadata: dict) -> list[Document]:
    """_summary_
    Split text into smaller chunks using RecursiveCharacterTextSplitter.
    Returns:
        _type_: _description_
    """
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            is_separator_regex=False
        )
        #chunks = splitter.create_documents([page_content])
        #chunks = splitter.split_documents([Document(page_content=page_content)])
        chunks = splitter.split_documents([Document(page_content=page_content, metadata=metadata)])
        logging.info(f"Split text into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logging.error(f"Error while splitting text into chunks: {e}")
        return []
        
def process_and_push(url: str, document_name: str, document_id: str):
    """_summary_
    Process and push chunks to AstraDB."
    """
    try:
        #extracted_data = load_document_update(url, document_name, document_id)
        document_metadata = {
            "document_name": document_name,
            "document_id": document_id
        }
        extracted_data = process_document(url, document_metadata)
        logging.info(f"Total Sections Extracted: {len(extracted_data)}")

        documents = []
        for chunk in extracted_data:
            page_content = chunk.get("page_content", "")

            if not page_content.strip():
                logging.warning(f"Skipping empty page content for chunk with metadata: {chunk['metadata']}")
                continue

            metadata = {
                "header1": chunk['metadata'].get('header1'),
                "header2": chunk['metadata'].get('header2'),
                "header3": chunk['metadata'].get('header3'),
                "header4": chunk['metadata'].get('header4'),
                "document_name": document_name,
                "document_id": document_id
            }

            chunks = split_text_into_chunks(page_content, metadata)
            
            for sub_chunk in chunks:
                token_count = count_tokens(sub_chunk.page_content)
                sub_chunk.metadata["token_embed"] = token_count
                documents.append(sub_chunk)

        logging.info(f"Total Chunks After Splitting: {len(documents)}")
        push_to_astra(documents)
    except Exception as e:
        logging.error(f"Error in process_and_push: {e}")


def push_to_astra(documents: list[Document]):
    """_summary_
    Push the chunks to AstraDB using the Vector Store.
    Args: document (list[Document])
    """
    try:
        logging.info("Pushing chunks to AstraDB...")
        _ = vector_store.add_documents(documents)
        logging.info(f"Successfully inserted {len(documents)} chunks into AstraDB with auto-embedding.")
    except Exception as e:
        logging.error(f"Error while pushing data to AstraDB: {e}")


def get_context(query: str) -> str:
    """_summary_
    Retrieve the most relevant context from AstraDB using vector similarity search.
    Returns:
        _type_: _description_
    """
    try:
        logging.info(f"Starting vector similarity search for query: '{query}'")
        retrieved_docs = vector_store.similarity_search(query, k=3, filter={"document_id": "1234-5678-uuid"})
        docs_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        if not docs_context:
            logging.warning("No context retrieved from the vector store.")
        
        return docs_context
    except Exception as e:
        logging.error(f"Error while retrieving context from vector store: {e}")
        return ""

def create_template()  -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template("""
        You are a machine learning engineer expert. Please integrate natural language reasoning to assist. 
        Use the following context to answer the question.

        Context:
        {docs_context}

        Chat History:
        {history}
        
        Question:
        {query}

        Answer:
    """)

def generate(query: str) -> str:
    """Generate an answer to the given query using the RAG (Retrieval-Augmented Generation) chain."""
    try:
        docs_context = get_context(query)
        logging.info(f"Context retrieved: \n{docs_context}")

        if not docs_context:
            logging.warning("No relevant context found to answer the question.")
            return "No relevant context found to answer the question."

        prompt_template = create_template()
        chain = prompt_template | model
        response = chain.invoke({"query": query, "docs_context": docs_context})
        response_clean = response.content
        return response_clean
    except Exception as e:
        logging.error(f"Error in generate: {e}")
        return "An error occurred while generating the response."


class RAGChatbot:
    def __init__(self):
        """Initialize with an empty chat history."""
        self.chat_history: List[Dict[str, str]] = []

    def get_context(self, query: str, document_id: str = '1234-5678-uuid') -> str:
        """
        Retrieve the most relevant context from AstraDB using vector similarity search.
        """
        try:
            retrieved_docs = vector_store.similarity_search(query=query, k=3, filter={"document_id": document_id})
            if not retrieved_docs:
                logging.warning("No relevant context found.")
                return ""
            
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            logging.info(f"Context successfully retrieved with {len(retrieved_docs)} documents.")
            return context
        except Exception as e:
            logging.error(f"Error during context retrieval: {e}")
            return ""

    def create_prompt_template(self) -> ChatPromptTemplate:
        """
        Create the prompt template for the LLM.
        """
        return ChatPromptTemplate.from_template("""
            You are a machine learning engineer expert. 
            Please integrate natural language reasoning to assist. 
            Use the following context and chat history to answer the question.

            Context:
            {docs_context}

            Chat History:
            {history}

            Question:
            {query}

            Answer:
        """)

    def generate(self, query: str) -> str:
        """
        Generate an answer to the given query using RAG (Retrieval-Augmented Generation) with chat history.
        """
        try:
            context = self.get_context(query)
            if not context:
                logging.warning("No relevant context found for query.")
                return "No relevant context found to answer the question."

            restored_history = inverse_construct_history(self.chat_history)
            
            prompt_template = self.create_prompt_template()
            
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, max_tokens=512)
            chain = prompt_template | llm
            
            response = chain.invoke({
                "query": query, 
                "docs_context": context, 
                "history": "\n".join([f"{type(msg).__name__}: {msg.content}" for msg in restored_history])
            })
            
            if response:
                logging.info(f"Generated response: {response.content}")
                self.chat_history.append({"human": query, "AI": response.content}) 
                return response.content
            else:
                logging.warning("No response from the LLM.")
                return "The model could not generate a response."
        except Exception as e:
            logging.error(f"Error in RAG generation: {e}")
            return "An error occurred while generating the response."

if __name__ == "__main__":
    pdf_url = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
    document_name = 'attention_is_all_you_need.pdf'
    document_id = '1234-5678-uuid'

    # push to astradb
    # process_and_push(pdf_url, document_name, document_id)
    
    # search similarity
    query = str(input("\n query: ").lower())
    results = vector_store.similarity_search_with_score(
    query, k=3, filter={'document_id': document_id}
    )
    for res, score in results:
        print(f"* \n\n[SIM={score:3f}] {res.page_content} [{res.metadata}]")
    
    # rag
    # query = str(input("\n query: ").lower())
    # result = generate(query)
    # print(f"llm generate: \n {result}")
    # chatbot  = RAGChatbot()
    # while True:
    #     try:
    #         user_input = input("\nHuman: ")
    #         if user_input.lower() in ['exit', 'quit']:
    #             logging.info("Exiting the chat session.")
    #             break

    #         answer = chatbot.generate(user_input)
    #         print(f"\nAI: {answer}")
        
    #     except KeyboardInterrupt:
    #         logging.info("Session interrupted by user. Exiting.")
    #         break
   


