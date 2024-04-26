from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
from dotenv import load_dotenv

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K = 3  


"""Loads environment variables from a .env file."""
load_dotenv()

def create_embedding_model():
    """Creates an instance of the OpenAI embedding model."""
    return OpenAIEmbeddings()

def create_vector_database(pdf_reader):
    """
    Creates a FAISS vector database from the provided PDF reader object.

    Args:
        pdf_reader (PyPDF2.PdfReader): The PDF reader object containing the document content.

    Returns:
        FAISS: The created FAISS vector database.
    """
    pdf_reader=PdfReader(pdf_reader)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()  # Extract text from each page and concatenate

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    db = FAISS.from_texts(chunks, embedding=create_embedding_model())
    return db

def get_response_to_query(db, query):
    """
    Retrieves a response to a given query using the vector database.

    Args:
        db (FAISS): The FAISS vector database containing document representations.
        query (str): The user's query to be answered.

    Returns:
        str: The generated response to the query, or "Data Not Available" if insufficient information is found.
    """

    similar_docs = db.similarity_search(query, k=K)
    concatenated_content = " ".join([doc.page_content for doc in similar_docs])

    llm = OpenAI(name="gpt-3.5-turbo-0125", verbose=True, temperature=0.5)

    prompt_template = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful PDF assistant that can answer questions about a document based on its text.
        Answer the following question: {question}
        By searching the attached PDF: {docs}
        Only use the factual information from the PDF to answer the question.
        If you feel like you don't have enough information to answer the question, say "Data Not Available".
        """
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    response = llm_chain.run(question=query, docs=concatenated_content)
    return response.replace("\n", "")  # Remove newline characters