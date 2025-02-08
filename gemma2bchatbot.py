"""
Gemma-2B Chatbot using LangChain, Ollama, and Streamlit.

This script sets up a chatbot that interacts with users via a simple
Streamlit UI and uses the Gemma-2B model for responses.
"""

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import Ollama  # pylint: disable=import-error
from langchain_core.prompts import ChatPromptTemplate  # pylint: disable=import-error
from langchain_core.output_parsers import StrOutputParser  # pylint: disable=import-error

def load_env_variables() -> None:
    """Loads environment variables from the .env file."""
    load_dotenv()
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "")

def create_prompt() -> ChatPromptTemplate:
    """Creates and returns a prompt template for the chatbot."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You're a helpful assistant. Kindly respond to the questions asked."),
            ("user", "Question: {question}"),
        ]
    )

def main() -> None:
    """Main function to run the Streamlit chatbot."""
    st.title("Gemma-2B with LangChain Framework")
    input_text = st.text_input("Hola! What's on your mind?")

    # Load prompt and model
    prompt = create_prompt()
    llm = Ollama(model="gemma:2b")  
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    if input_text:
        response = chain.invoke({"question": input_text})
        st.write(response)

if __name__ == "__main__":
    load_env_variables()
    main()
