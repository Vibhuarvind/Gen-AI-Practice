import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()

# Load the environment variables from the .env file
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

#prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you're a helpful assistant. Kindly reposnd to the questions asked." ),
        ("user","Question:{question}")
    ]
)

#streamlit framework
st.title("Gemma-2B with Lanchain Framework")
input_text = st.text_input("Hola! what's on ur mind ?")

#ollama gemma2b model
llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))

    
