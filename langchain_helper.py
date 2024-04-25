"""
The code file is a wrapper for Langchain helper functions
Input: PDF file, Questions
Output: Question, Response
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader

load_dotenv()

"""Embeddings to convert strings to numeric representation"""
embeddings= OpenAIEmbeddings()

"""Function to create a database of vectors from PDF that taken as input"""
def create_vector_db_from_pdf(pdf_reader) -> FAISS:
    # PDF Reader to read padf
    pdf_reader = PdfReader(pdf_reader)
    text= "" 
    for page in pdf_reader.pages:
            text += page.extract_text() # Extract text from each page and concatenate 
    """
    Split the text into chunks and overlap 
    For Example, A sentence with 1000 words will be divided into multiple sentences of size 100 in which second sentence start 
    from 80th word rather than 101 so on.
    """
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size =1000,
            chunk_overlap =200,
            length_function = len
            )
    chunks = text_splitter.split_text(text=text)
    """
    Using FAISS to create vectorstores
    """
    db = FAISS.from_texts(chunks,embedding=embeddings)
    return db

"""
Function takes query as input and do similarity search in vector store return the docs.
Docs get concatenated and return as one paragraph
"""
def get_response_from_query(db,query,k=3):
    # gpt-3.5-turbo-0125 can handle 16385 tokens and returns 4096
    """Similarity search with Database vectorstores"""
    docs = db.similarity_search(query,k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    """Object for OpenAI llm"""
    llm=OpenAI(name="gpt-3.5-turbo-0125",verbose=True,temperature=0.5) 
    """Promt to OpenAI gpt (Format of the input and template that explain llm how to behave)"""
    prompt = PromptTemplate(
        input_variables=["question","docs"],
        template="""You are a helpful pdf assistant that that can answer questions about
        document based on the text in pdf.
        Answer the following question:{question}
        By searching the attached pdf: {docs}
        Only use the factual information from the pdf to answer the question.
        If you feel like you don't have enough information to answer the question,say "Data Not Available"."""
    )
    """LLMChain to integrate the functionalities of llm and prompt"""
    chain=LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query,docs=docs_page_content)
    response = response.replace("\n","") # Remove newline characters
    return response
