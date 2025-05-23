
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
st.set_page_config(page_title="GenAI Q&A Bot", layout="centered")
st.title("📚 GenAI Q&A Bot for Company Documents")
st.markdown("This tool answers questions based on uploaded or predefined documents. Optionally, you can restrict your question to a specific page.")

uploaded_file = st.file_uploader("Upload your .txt document:", type="txt")
default_file = st.selectbox("Or choose from sample documents:", [
    "company_document.txt", "dataset1.txt", "dataset2.txt", "dataset3.txt"
])

if uploaded_file:
    with open("uploaded.txt", "wb") as f:
        f.write(uploaded_file.read())
    filename = "uploaded.txt"
else:
    filename = f"data/{default_file}"

loader = TextLoader(filename)
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

for i, chunk in enumerate(chunks):
    chunk.metadata["page"] = i + 1

question = st.text_input("🔎 Enter your question:")
page_number = st.number_input("📄 Restrict to page (optional)", min_value=0, max_value=len(chunks), value=0)

embeddings = HuggingFaceEmbeddings()

if page_number > 0:
    page_chunks = [chunk for chunk in chunks if chunk.metadata["page"] == page_number]
    if not page_chunks:
        st.warning("⚠️ Invalid page number.")
        st.stop()
    vectorstore = FAISS.from_documents(page_chunks, embeddings)
else:
    vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever()

llm = ChatOpenAI(
    model_name="llama3-8b-8192",
    temperature=0,
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ["GROQ_API_KEY"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

if question.strip():
    with st.spinner("Thinking..."):
        answer = qa_chain.run(question)
        st.success(answer)
