GenAI Q&A Bot (Streamlit + Groq)

This is a simple project built as part of a mock placement test for Akaike Technologies.

It allows you to upload a text document and ask questions based on its content. You can also focus on specific pages of the document.

 Technologies Used
- Python 3
- LangChain
- FAISS
- Groq LLaMA 3
- HuggingFace Embeddings
- Streamlit

 Features
- Upload or select from multiple documents
- Ask natural language questions
- Optionally restrict to a page number
- Web-based Streamlit UI (no coding required to use)

 Folder Structure
genai-qa-bot/
├── app/
│ └── streamlit_app.py
├── data/
│ ├── company_document.txt
│ ├── dataset1.txt
│ ├── dataset2.txt
│ └── dataset3.txt
├── requirements.txt
└── README.md

To Run:
pip install -r requirements.txt
streamlit run app/streamlit_app.py
