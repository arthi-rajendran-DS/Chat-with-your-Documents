import streamlit as st
import os
import io
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import UnstructuredURLLoader

def main():
    st.title("Your PDF Gini")
    st.subheader("Upload a PDF file and enter your Hugging Face API key")

    # Add file upload functionality
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    st.write("How to get your API? https://youtu.be/jo_fTD2H4xA")

    # Add input for Hugging Face API key
    api_key = st.text_input("Enter your Hugging Face API key")

    

    if uploaded_file is not None and api_key:
        # Set the Hugging Face API key
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key

        # Load the PDF document
        pdf_loader = pdf_loader = UnstructuredPDFLoader(io.BytesIO(uploaded_file.read()))
        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(),
            text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        ).from_loaders([pdf_loader])

        # Load the retrieval QA chain
        llm2 = HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", model_kwargs={"temperature": 0, "max_length": 1512})
        chain = RetrievalQA.from_chain_type(
            llm=llm2,
            chain_type="stuff",
            retriever=index.vectorstore.as_retriever(),
            input_key="question"
        )

        # Run the question answering chain
        chain.run()

        # Display the results
        st.subheader("PDF says:")
        st.write(chain.results)

if __name__ == "__main__":
    main()
