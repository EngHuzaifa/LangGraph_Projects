# Gardening Agent Streamlit Application

This project is a **Gardening Assistant** built using Streamlit and various AI technologies, including Google Generative AI (Gemini), Tavily, and LangChain. It allows users to input gardening-related queries and provides personalized gardening advice, plant recommendations, and gardening tips based on the user's input.

## Features

- **Understand Garden Query**: The agent processes user input to extract key information about the user's gardening goals.
- **Web Search Integration**: Performs web searches to gather additional information and gardening tips.
- **Garden Analysis**: Analyzes the user's garden type, climate zone, space, and other factors to provide personalized insights.
- **Plant Recommendations**: Based on the analysis, the agent recommends plants suited for the user's garden conditions.
- **Final Advice**: Provides a detailed gardening guide, including planting instructions, care guidelines, and seasonal tips.

## Requirements

### Python Libraries

Ensure you have the following libraries installed:
- `streamlit`
- `langchain`
- `langchain-google-genai`
- `langgraph`
- `pydantic`
- `tavily`

You can install them via pip:

```bash
pip install streamlit langchain langchain-google-genai langgraph pydantic tavily
