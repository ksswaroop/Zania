import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import helper_functions as llm_helper
import os 
import tempfile
import shutil

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

    # Upload a PDF File
    uploaded_pdf = st.file_uploader("Upload a PDF file for information extraction", type="pdf")

    # Question about the PDF
    user_query = st.text_input("Ask a question about your uploaded PDF")

    if uploaded_pdf and user_query:  # Validate both file and query presence
        # Process the uploaded PDF
        # Create a temporary directory using the tempfile module
        temp_dir = tempfile.mkdtemp()
        tmp_location = os.path.join(temp_dir, uploaded_pdf.name)
        with open(tmp_location, 'wb') as f:
            f.write(uploaded_pdf.getvalue())
        retriever = llm_helper.create_vector_database(tmp_location)
        retrieval_chain = llm_helper.retrieval_chain(retriever)

        # Generate a response to the user's query
        response = llm_helper.question_answer(retrieval_chain, user_query)

        # Display the response in a clear and user-friendly format
        st.write({user_query: response})

if __name__ == "__main__":
    main()


# import streamlit as st
# from streamlit_extras.add_vertical_space import add_vertical_space
# import helper_functions as llm_helper
# import os 
# import tempfile
# import shutil

# def main():
#   """
#   The main function for the Zania Document Assistant Streamlit app.
#   """

#   # Sidebar Content
#   with st.sidebar:
#     st.title("Zania Document Assistant")
#     st.markdown(
#         """
#         ## About

#         This app is an LLM-powered document assistant built using:

#         - Streamlit 
#         - LangChain 
#         - OpenAI LLM models

#         """
#     )
#     add_vertical_space(5)

#   """Upload a PDF File"""
#   uploaded_pdf = st.file_uploader("Upload a PDF file for information extraction", type="pdf")

#   """Question about the PDF"""
#   user_query = st.text_input("Ask a question about your uploaded PDF")

#   if uploaded_pdf and user_query:  # Validate both file and query presence
#     # Process the uploaded PDF
#     # Create a temporary directory using the tempfile module
#     temp_dir = tempfile.mkdtemp()
#     tmp_location = os.path.join(temp_dir, uploaded_pdf.name)
#     with open(tmp_location, 'wb') as f:
#         f.write(uploaded_pdf.getvalue())
#     retriever = llm_helper.create_vector_database(tmp_location)
#     retrieval_chain= llm_helper.retrieval_chain(retriever)

#     # Generate a response to the user's query
#     response = llm_helper.question_answer(retrieval_chain, user_query)

#     # Display the response in a clear and user-friendly format
#     st.write({user_query:response})

# if __name__ == "__main__":
#   main()
