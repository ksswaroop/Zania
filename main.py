import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import langchain_helper as lch
import textwrap

# Sidebar Content
with st.sidebar:
    st.title("Zania Document Assistant")
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
                
    ''')
    add_vertical_space(5)
    #st.write('Made by Saiswaroop using streamlit')

# Upload a pdf file
pdf = st.file_uploader("Upload your pdf that you want to get information from",type='pdf')
    # Question 
query = st.text_input("Ask questions about your PDF file")

if query and pdf:
    db = lch.create_vector_db_from_pdf(pdf)
    response= lch.get_response_from_query(db,query)
    #response1=textwrap.wrap(response,width=100)
    st.header("Answer")
    st.write({query:response})