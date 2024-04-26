import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import langchain_helper as llm_helper 

def main():
  """
  The main function for the Zania Document Assistant Streamlit app.
  """

  # Sidebar Content
  with st.sidebar:
    st.title("Zania Document Assistant")
    st.markdown(
        """
        ## About

        This app is an LLM-powered document assistant built using:

        - Streamlit 
        - LangChain 
        - OpenAI LLM models

        """
    )
    add_vertical_space(5)

  """Upload a PDF File"""
  uploaded_pdf = st.file_uploader("Upload a PDF file for information extraction", type="pdf")

  """Question about the PDF"""
  user_query = st.text_input("Ask a question about your uploaded PDF")

  if uploaded_pdf and user_query:  # Validate both file and query presence
    # Process the uploaded PDF
    document_database = llm_helper.create_vector_database(uploaded_pdf)

    # Generate a response to the user's query
    response = llm_helper.get_response_to_query(document_database, user_query)

    # Display the response in a clear and user-friendly format
    st.write(response)

if __name__ == "__main__":
  main()
