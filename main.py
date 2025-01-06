import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  # Take environment variables from .env (especially OpenAI API key)

# Streamlit UI Setup
st.title("Scheme Research Tool 📚🔍")
st.sidebar.title("Input Options")

# URL Input and File Upload
url_input_type = st.sidebar.radio("Choose Input Method", ("Enter URLs", "Upload Text File with URLs"))

urls = []

# Handle URL Input
if url_input_type == "Enter URLs":
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i+1}")
        if url:
            urls.append(url)

# Handle File Upload
elif url_input_type == "Upload Text File with URLs":
    uploaded_file = st.sidebar.file_uploader("Upload a text file containing URLs", type=["txt"])
    if uploaded_file:
        file_contents = uploaded_file.read().decode("utf-8")
        urls = file_contents.splitlines()

# Process URL Button
process_url_clicked = st.sidebar.button("Process URLs and Build Knowledge Base")
file_path = "faiss_store_scheme_research.pkl"
main_placeholder = st.empty()

# Define OpenAI Model
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked and urls:
    main_placeholder.text("Data Loading...Started...✅✅✅")
    # Load documents from URLs
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    main_placeholder.text("Text Splitter...Started...✅✅✅")
    # Split the loaded documents into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)

    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    # Create embeddings for the document chunks and store them in FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_scheme = FAISS.from_documents(docs, embeddings)
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_scheme, f)

    main_placeholder.text("Data Processed and Knowledge Base Created! Ready for queries.")

# User Query Section
query = main_placeholder.text_input("Ask a Question About the Schemes: ")

if query:
    if os.path.exists(file_path):
        # Load the FAISS index for querying
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        # Set up retrieval chain using the FAISS index
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        # Display the answer and sources
        st.header("Answer")
        st.write(result["answer"])

        # Display sources if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
