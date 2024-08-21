from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.embeddings import GooglePalmEmbeddings
import os
import streamlit as st
import os.path

# Set the Google API key
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAv5jmX3j3utT5ewZXAmJEf-ShG3eurDuU'

# Create Google Palm LLM model
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.environ['GOOGLE_API_KEY'], temperature=0.1)

# Initialize instructor embeddings using GooglePalmEmbeddings
instructor_embeddings = GooglePalmEmbeddings()

# Define the vector database file path
vectordb_file_path = "faiss_index"

# Function to create the vector database if it doesn't exist
def create_vector_db():
    if not os.path.exists(vectordb_file_path):
        # Load data from FAQ sheet
        loader = CSVLoader(file_path='./all_questions_answers.csv', source_column="prompt")
        data = loader.load()

        # Create a FAISS instance for vector database from 'data'
        vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

        # Save vector database locally
        vectordb.save_local(vectordb_file_path)
    else:
        print("Vector database already exists. Skipping creation.")

# Function to get the QA chain
def get_qa_chain():
    # Load the vector database from the local folder with dangerous deserialization allowed
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

# Streamlit interface
st.title("ARK-X Q&A 🌱")

# Automatically create the knowledgebase if it doesn't exist
create_vector_db()

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])
