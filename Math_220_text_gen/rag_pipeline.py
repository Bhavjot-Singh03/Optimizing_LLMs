import logging
import pprint
import os
import langchain_community
import chromadb

from workflows.custom_model_logic import ChatParrotLink
from typing_extensions import List
from transformers import AutoModelForCausalLM

from prefect import flow, task
from prefect.cache_policies import TASK_SOURCE, INPUTS
from prefect.input.run_input import receive_input
logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("httpx").setLevel(logging.WARNING)

cache_policy = (TASK_SOURCE + INPUTS).configure(key_storage=r"W:\ML-DL-GENAI\MLOps\Projects\Optimizing LLMs\Math_220\cache")

from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import JSONLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv('PythonProject/Optimizing_LLMs/keys.txt', override = True)

# ------------Initializing the adapter------------
adapter_path = r'C:\Users\singh\PycharmProjects\PythonProject\GPT2-SFT\GPT-Math-2.3M'
model_name = 'openai-community/gpt2'

model = AutoModelForCausalLM.from_pretrained(model_name)
model.load_adapter(adapter_path)

model = ChatParrotLink(parrot_buffer_length=1000, model=model)
# ---------------RAG Pipeline----------------

@task(cache_policy=TASK_SOURCE + INPUTS)
def load_jsonl(path:str) -> List:
    """
    Loads the JSONl file and converts it into langchain documents.

    Args:
        path (str) : The path to the JSONl file.

    Returns:
        (List) : List of langchain documents.
    """
    file_name = path.split('/')[-1]
    logging.info("Loading file : %s", file_name)
    loader = JSONLoader(path, jq_schema = "{problem: .problem, solution: .solution}", json_lines=True, text_content=False)
    documents = loader.load()

    return documents

@task(cache_policy=TASK_SOURCE + INPUTS)
def semantic_chunker(data:List, breakpoint_threshold_type:str) -> List:
    """
    Performs semantic chunking by splitting the text based on the breakpoint_threshold_type.

    Args:
        data (List): List of documents.
        breakpoint_threshold_type (str): The type of breakpoint threshold to use.

    Returns:
        (List) : List of chunks
    """
    text_splitter = SemanticChunker(
        OpenAIEmbeddings(),
        breakpoint_threshold_type=breakpoint_threshold_type
    )
    chunks = text_splitter.split_documents(data)
    logging.info("Total number of chunks : %d", len(chunks))
    return chunks

@task(cache_policy=TASK_SOURCE + INPUTS)
def create_embeddings(chunks:List, persist_directory:str) -> Chroma:
    """
    Returns a chroma vector store object containing the embeddings.

    Args:
        chunks (List): A list of chunks.
        persist_directory (str): The path where the embeddings are stored.

    Returns:
        (Chroma): A chroma vector store object containing the embeddings.
    """
    embedding_model = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    vector_store = langchain_community.vectorstores.chroma.Chroma.from_documents(
      documents=chunks,
      embedding=embedding_model,
      persist_directory=persist_directory
    )
    logging.info("Constructed vector store of type : %s", type(vector_store))
    return vector_store

@task(cache_policy=TASK_SOURCE + INPUTS)
def load_embeddings(persist_directory:str) -> Chroma:
    """
    Returns a chroma vector store object containing the embeddings.

    Args:
        persist_directory (str): The path where the embeddings are stored.

    Returns:
        (Chroma) : A chroma vector store object containing the embeddings.
    """
    embedding_model = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    vector_store = langchain_community.vectorstores.chroma.Chroma(
      persist_directory=persist_directory,
      embedding_function=embedding_model
    )
    logging.info("Embeddings already exist. Loading..")
    return vector_store

@task(cache_policy=TASK_SOURCE + INPUTS)
def generate_results(vector_store:Chroma, query:str) -> str:
    """
    Generates the response by passing in the query to the vector store.

    Args:
        vector_store (Chroma): A chroma vector store object containing the embeddings.
        query (str): The query to be passed to the vector store.

    Returns:
        (str): The answer to the query.
    """
    llm = ChatOpenAI()
    logging.info("Generating results...")
    retriever = MultiQueryRetriever.from_llm(retriever=vector_store.as_retriever(search_type='mmr'), llm=llm)
    docs = retriever.get_relevant_documents(query=query)
    chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever)
    answer = chain.invoke(docs)
    return answer

@flow(name="rag-pipeline-math", log_prints=True)
def rag_pipeline_math(path:str, persist_directory:str) -> None:
    """
    Workflow to implement RAG.

    Args:
        path (str) : The path to the data file.
        persist_directory (str) : The path to the vectordb.
    Returns:
        None:
    """

    if os.path.exists(persist_directory):
        # Check if the directory is empty
        if not os.listdir(persist_directory):
            print(f"Directory '{persist_directory}' exists but is empty. Proceeding with vector store creation.")
            documents = load_jsonl(path=path)
            chunks = semantic_chunker(data=documents, breakpoint_threshold_type='percentile')
            vector_store = create_embeddings(chunks=chunks, persist_directory=persist_directory)
            que = input("Provide the query: ")
            answer = generate_results(vector_store=vector_store, query=que)
            doc = answer['query'][0]
            content = doc.page_content
            pprint.pprint(content)

        else:
            print(f"Directory '{persist_directory}' exists and is not empty. Skipping vector store creation.")
            vector_store = load_embeddings(persist_directory=persist_directory)
            que = input("Provide the query: ")
            answer = generate_results(vector_store=vector_store, query=que)
            doc = answer['query'][0]
            content = doc.page_content
            pprint.pprint(content)

    else:
        print(f"Directory '{persist_directory}' does not exist. Creating it and proceeding with vector store creation.")
        os.makedirs(persist_directory, exist_ok=True)
        documents = load_jsonl(path=path)
        chunks = semantic_chunker(data=documents, breakpoint_threshold_type='percentile')
        vector_store = create_embeddings(chunks=chunks, persist_directory=persist_directory)
        que = input("Provide the query: ")
        answer = generate_results(vector_store=vector_store, query=que)
        doc = answer['query'][0]
        content = doc.page_content
        pprint.pprint(content)

if __name__ == "__main__":
    client = chromadb.PersistentClient(path=r'C:\Users\singh\PycharmProjects\PythonProject\Optimizing_LLMs\Math_220_text_gen\persist')
    rag_pipeline_math(path=r'C:\Users\singh\PycharmProjects\PythonProject\Optimizing_LLMs\Math_220_text_gen\Dataset\test_data_2025-03-18-09-08-21.jsonl',
                      persist_directory=r'C:\Users\singh\PycharmProjects\PythonProject\Optimizing_LLMs\Math_220_text_gen\persist')





