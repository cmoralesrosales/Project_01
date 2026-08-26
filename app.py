
import streamlit as st
import os
import json
import requests
from langchain_community.document_loaders import PyMuPDFLoader
from openai import OpenAI
import tiktoken
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import tempfile

file_name = 'config.json'
with open(file_name, 'r') as file:
    config = json.load(file)
    OPENAI_API_KEY = config.get("OPENAI_API_KEY")
    OPENAI_API_BASE = config.get("OPENAI_API_BASE")

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE


## Initialize OpenAI client
client = OpenAI(
     api_key=OPENAI_API_KEY,
     base_url=OPENAI_API_BASE
 )

## Define the system prompt for the model
qna_system_message = """
You are an AI assistant designed to help research teams in efficiently reviewing and extract key insights from extensive reports.
Your task is to provide evidence-based, concise, and relevant answers to researchers' questions based on the context provided.

User input will include the necessary context for you to answer their questions. This context will begin with the token:

###Context
The context contains excerpts from one or more research papers, along with associated metadata such as titles,
authors,and publication details. Each excerpt will be clearly marked with its original page number, e.g., 'Context from Page X: [content]'.

When crafting your response:
- Use ONLY the provided context to answer the question.
- If the answer is found in the context, respond with concise and insight-focused summaries.
- For sources, always include the paper title, all identified authors, the publishing organization, and the publication date.
- If information for a single answer comes from multiple pages, cite all relevant page numbers in the source.
- If the question is unrelated to the context or the context is empty, clearly respond with: "Sorry, this is out of my knowledge base."

Please adhere to the following response guidelines:
- Provide clear, direct answers using only the given context.
- Do not include any additional information outside of the context.
- Avoid rephrasing or generalizing unless explicitly relevant to the question.
- If no relevant answer exists in the context, respond with: "Sorry, this is out of my knowledge base."
- If the context is not provided, your response should also be: "Sorry, this is out of my knowledge base."

Here is an example of how to structure your response:

Answer:
[Answer based on context]

Source:
[Source details with page or section]
"""

## Define the user message template
qna_user_message_template = """

"""

@st.cache_resource
def load_and_process_pdfs(uploaded_files):
     all_documents = []
     for uploaded_file in uploaded_files:
         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
             tmp_file.write(uploaded_file.getvalue())
             tmp_file_path = tmp_file.name
         loader = PyMuPDFLoader(tmp_file_path)
         documents = loader.load()
         all_documents.extend(documents)
         os.remove(tmp_file_path) # Clean up the temporary file
     text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
         encoding_name='cl100k_base',
         chunk_size=1000,
     )
     document_chunks = text_splitter.split_documents(all_documents)

     embedding_model = OpenAIEmbeddings(
         openai_api_key=OPENAI_API_KEY,
         openai_api_base=OPENAI_API_BASE
     )

     # Create an in-memory vector store (or use a persistent one if needed)
     vectorstore = Chroma.from_documents(
         document_chunks,
         embedding_model
     )
     return vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 5})

def generate_rag_response(user_input, retriever, max_tokens=500, temperature=0, top_p=0.95):
     # Retrieve relevant document chunks
     relevant_document_chunks = retriever.get_relevant_documents(query=user_input)
     context_list = [d.page_content for d in relevant_document_chunks]

     # Combine document chunks into a single context
     context_for_query = ". ".join(context_list)

     user_message = qna_user_message_template.replace('{context}', context_for_query)
     user_message = user_message.replace('{question}', user_input)

     # Generate the response
     try:
         response = client.chat.completions.create(
             model="gpt-4o-mini",
             messages=[
                 {"role": "system", "content": qna_system_message},
                 {"role": "user", "content": user_message}
             ],
             max_tokens=max_tokens,
             temperature=temperature,
             top_p=top_p
         )
         response = response.choices[0].message.content.strip()
     except Exception as e:
         response = f'Sorry, I encountered the following error: \n {e}'

     return response

# # Streamlit App
st.title("LLM-Powered Research Assistant")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

retriever = None
if uploaded_files:
     st.info("Processing uploaded PDFs...")
     retriever = load_and_process_pdfs(uploaded_files)
     st.success("PDFs processed and ready for questioning!")


if retriever:
     user_question = st.text_input("Ask a question about the uploaded documents:")
     if user_question:
         with st.spinner("Generating response..."):
             rag_response = generate_rag_response(user_question, retriever)
             st.write(rag_response)
