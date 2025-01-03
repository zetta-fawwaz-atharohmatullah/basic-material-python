# ********** IMPORT FRAMEWORKS **************
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
# ********** IMPORT LIBRARY **************
from dotenv import load_dotenv
from astrapy import DataAPIClient
import tiktoken
import os 
import logging
from day6_format_pdf_new import DocumentPreprocess
from astrapy import DataAPIClient
from typing import Optional
from material_raglangchain.langchain_day4 import get_model
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import LLMListwiseRerank

load_dotenv(override=True)

class APISetup:
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE") 
    ASTRA_DB_KEYCOLLECTION = os.getenv("ASTRA_DB_KEYCOLLECTION")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def vectorstore():
    vector_store = AstraDBVectorStore(
        collection_name=APISetup.ASTRA_DB_KEYCOLLECTION,
        embedding=APISetup.embeddings,
        api_endpoint=APISetup.ASTRA_DB_API_ENDPOINT,
        token=APISetup.ASTRA_DB_APPLICATION_TOKEN,
        namespace=APISetup.ASTRA_DB_KEYSPACE,
    )
    return vector_store

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
        chunks = splitter.split_documents([Document(page_content=page_content, metadata=metadata)])
        logging.info(f"Split text into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logging.error(f"Error while splitting text into chunks: {e}")
        return []
        
def process_and_push(url: str, document_name: str, document_id: str):
    """_summary_
    Process and push chunks to AstraDB.
    Args: doc url, document name, document id
    """
    try:
        processor = DocumentPreprocess(url, document_name, document_id)
        processor.extract_html_from_pdf()
        extracted_data = processor.process_document()
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
        vector_store = vectorstore()
        _ = vector_store.add_documents(documents)
        logging.info(f"Successfully inserted {len(documents)} chunks into AstraDB with auto-embedding.")
    except Exception as e:
        logging.error(f"Error while pushing data to AstraDB: {e}")

def type_retrieval(key:str, query: str, k: Optional[int]=3):
    llm = get_model("gpt-4o-mini")
    if key == "astra":
        try:
            logging.info("Start retrieval with astra client...")
            client = DataAPIClient(APISetup.ASTRA_DB_APPLICATION_TOKEN)
            db = client.get_database_by_api_endpoint(
            APISetup.ASTRA_DB_API_ENDPOINT
            )
            collection = db.get_collection(APISetup.ASTRA_DB_KEYCOLLECTION)
            results = collection.find(
                sort={"$vectorize": query},
                limit=k,
                projection={"$vectorize": True},
                include_similarity=True,
            )
            print(results)
            
            formatted_results = []
            for doc in results:
                similarity = doc.get("$similarity", 0)
                text = doc.get("content", "")  
                formatted_results.append(f"Similarity: {similarity:.3f} - {text}")
            logging.info("succesfully retrieve with astra client...")
            return "\n".join(formatted_results)
            
        except Exception as e:
            logging.error(f"Error while retrieve data with astra client: {e}")
        
    elif key == "langchain1":
        try:
            logging.info("Start retrieval with langchain as retriever...")
            vector_store = vectorstore()
            retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k, 
                "score_threshold": 0.7,
                "filter": {"document_id": "1234-5678-uuid"}
            }
        )
            docs = retriever.invoke(query)
            logging.info("succesfully retrieve with langchain as retriever...")
            return "\n\n".join(doc.page_content for doc in docs)
        except Exception as e:
            logging.error(f"Error while retrieve data with langchain as retriever: {e}")
    
    elif key == "langchain2":
        try:
            logging.info("Start retrieval with langchain multiquery retriever...")
            vector_store = vectorstore()
            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=vector_store.as_retriever(), llm=llm
            )
            result = retriever_from_llm.invoke(query)
            logging.info("succesfully retrieve with langchain multiquery retriever...")
            return "\n\n".join(doc.page_content for doc in result)
        except:
            logging.error(f"Error while retrieve data with multiquery: {e}")
    
    elif key == "langchain3":
        try:
            logging.info("Start retrieval with langchain ContextualCompressionRetriever...")
            vector_store = vectorstore()
            compressor = LLMChainExtractor.from_llm(llm)
            compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                                base_retriever=vector_store.as_retriever())
            compressed_docs = compression_retriever.invoke(
                query
            )
            logging.info("succesfully retrieve with langchain ContextualCompressionRetriever...")
            return "\n\n".join(doc.page_content for doc in compressed_docs)
        except:
            logging.error(f"Error while retrieve data with ContextualCompressionRetriever: {e}")
    
    elif key == "langchain4":
        try:
            logging.info("Start retrieval with langchain LLMListwiseRerank...")
            vector_store = vectorstore()
            compressor = LLMChainExtractor.from_llm(llm)
            compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                                base_retriever=vector_store.as_retriever())
            _filter = LLMListwiseRerank.from_llm(llm, top_n=2)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=_filter, base_retriever=compression_retriever
            )

            compressed_docs = compression_retriever.invoke(
                query
            )
            logging.info("succesfully retrieve with langchain LLMListwiseRerank...")
            return "\n\n".join(doc.page_content for doc in compressed_docs)
        except:
            logging.error(f"Error while retrieve data with LLMListwiseRerank: {e}")
    

if __name__ == "__main__":
    pdf_url = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
    document_name = 'attention_is_all_you_need.pdf'
    document_id = '1234-5678-uuid'

    # push to astradb
    #process_and_push(pdf_url, document_name, document_id)
    
    # search similarity
    # astra_re = type_retrieval("astra", "how attention works?")
    # print(f"\n{astra_re}")
    
    # lang_as = type_retrieval("langchain1", "how attention works?", 3)
    # print(f"\n{lang_as}")
    
    lang_as = type_retrieval("langchain2", "how attention works in transformer?")
    print(f"\n{lang_as}")
    
    
    
    
   


