import streamlit as st
import pandas as pd
import pytesseract
from PyPDF2 import PdfReader
from langchain.agents import create_csv_agent
import os
from apikey import apikey
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
import time
from genai.extensions.langchain import LangChainInterface
from genai.schemas import ModelType, GenerateParams
from genai.model import Credentials



genai_api_key = st.sidebar.text_input("GenAI API Key", type="password")
genai_api_url = st.sidebar.text_input("GenAI API URL", type="default")
max_tokens = st.sidebar.number_input("Max new tokens")
min_tokens = st.sidebar.number_input("Min new tokens")
decoding_method = st.sidebar.text_input("Decoding method (Choose either greedy or sample) ", type="default")
repetition_penalty = st.sidebar.number_input("Repetition penalty (Choose either 1 or 2)")
temperature = st.sidebar.number_input("Temperature (Choose a decimal number between 0 & 2)")

creds = Credentials(api_key=genai_api_key, api_endpoint=genai_api_url)
params = GenerateParams(decoding_method=decoding_method, temperature=temperature, max_new_tokens=max_tokens, min_new_tokens=min_tokens, repetition_penalty=repetition_penalty)
llm=LangChainInterface(model=ModelType.FLAN_UL2, params=params, credentials=creds)

def pdf_to_text(pdf_path):
    # Step 1: Convert PDF to images
    images = convert_from_path(pdf_path)

    with open('output.txt', 'w') as f:  # Open the text file in write mode
        for i, image in enumerate(images):
            # Save pages as images in the pdf
            image_file = f'page{i}.jpg'
            image.save(image_file, 'JPEG')

            # Step 2: Use OCR to extract text from images
            text = pytesseract.image_to_string(image_file)

            f.write(text + '\n')  # Write the text to the file and add a newline for each page

def load_csv_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.to_csv("uploaded_file.csv")
    return df

def load_txt_data(uploaded_file):
    with open('uploaded_file.txt', 'w') as f:
        f.write(uploaded_file.getvalue().decode())
    return uploaded_file.getvalue().decode()

def load_pdf_data(uploaded_file):
    with open('uploaded_file.pdf', 'wb') as f:
        f.write(uploaded_file.getbuffer())
    pdf = PdfReader('uploaded_file.pdf')
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    pdf_to_text('uploaded_file.pdf')
    return text

def main():
    st.title("Chat With Your Documents (csv, txt and pdf)")

    file = st.file_uploader("Upload a file", type=["csv", "txt", "pdf"])


    if file is not None:
        if file.type == "text/csv":
            doc = "csv"
            data = load_csv_data(file)
            agent = create_csv_agent(OpenAI(temperature=0), 'uploaded_file.csv', verbose=True)
            st.dataframe(data)

        elif file.type == "text/plain":
            doc = "text"
            data = load_txt_data(file)
            loader = TextLoader('uploaded_file.txt')
            index = VectorstoreIndexCreator().from_loaders([loader])

        elif file.type == "application/pdf":
            doc = "text"
            data = load_pdf_data(file)
            loader = TextLoader('output.txt')
            index = VectorstoreIndexCreator().from_loaders([loader])

        # do something with the data


        question = st.text_input("Once uploaded, you can chat with your document. Enter your question here:")
        submit_button = st.button('Submit')

        if submit_button:
            if doc == "text":
                response = index.query(question)
            else:
                response = agent.run(question)

            if response:
                st.write(response)


if __name__ == "__main__":
    main()
