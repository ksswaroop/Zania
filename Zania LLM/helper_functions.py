# Import necessary libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from operator import itemgetter
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K = 3 

# Load environment variables from a .env file
load_dotenv()

# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

# Function to generate multiple perspectives on a user question
def prompt_perspective(template):
    prompt_perspectives = ChatPromptTemplate.from_template(template)
    return prompt_perspectives

# Function to create a vector database from PDF documents
def create_vector_database(pdf_parser):
    loader = PyPDFLoader(pdf_parser)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_documents(pages)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    return retriever

# Function to get unique union of retrieved documents
def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

# Generate multiple perspectives on the user question
prompt_perspectives = prompt_perspective(template)
generate_queries = (prompt_perspectives | ChatOpenAI(temperature=0) | StrOutputParser() | (lambda x: x.split("\n")))

# Define retrieval chain
def retrieval_chain(retriever):
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    return retrieval_chain

# Function to answer questions based on retrieved context
def question_answer(retrieval_chain, question):
    # RAG
    template = """Answer the following question based on this context.If you feel like you don't have enough information to answer the question, say "Data Not Available":
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0)

    final_rag_chain = (
        {"context": retrieval_chain, 
         "question": itemgetter("question")} | prompt | llm | StrOutputParser())
    return final_rag_chain.invoke({"question": question})


