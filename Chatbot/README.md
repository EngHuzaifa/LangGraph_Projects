# ChatGPT-like Chatbot using LangGraph and Streamlit

## Overview
This project integrates LangGraph, Streamlit, and Tavily to build a ChatGPT-like chatbot UI. It utilizes Google's Gemini LLM for responses and allows interaction with external search results using Tavily.

## Features
- **Interactive Chat UI**: Built using Streamlit, enabling seamless conversations.
- **Advanced LLM**: Powered by Google's Gemini model for intelligent responses.
- **Tool Integration**: Includes Tavily search for fetching relevant results.

## Requirements
- Python 3.8 or higher
- Internet access for API calls

## Installation
    

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Set up environment variables:
    ```bash
    export LANGCHAIN_API_KEY=<your-langchain-api-key>
    export GEMINI_API_KEY=<your-gemini-api-key>
    export TAVILY_API_KEY=<your-tavily-api-key>
    ```

## Usage
1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Interact with the chatbot via the web UI.


