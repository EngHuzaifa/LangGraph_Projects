# 🌱 Gardener Assistant

The **Gardener Assistant** is an intelligent gardening tool designed to assist users in planning and maintaining their gardens. Powered by advanced AI technologies like LangGraph, LangChain, and Google Generative AI, this interactive Streamlit application provides tailored gardening insights, web search results, plant recommendations, and comprehensive advice.


## Features

- **Garden Analysis**: Understand the user's gardening needs and goals.
- **Web Search Results**: Retrieve relevant gardening tips and resources.
- **Plant Recommendations**: Suggest plants based on climate, soil, and user preferences.
- **Final Gardening Advice**: Provide actionable advice to ensure a thriving garden.

### Key Highlights
- Easy-to-use **Streamlit** interface.
- Advanced natural language processing using **LangChain**.
- Real-time web search with **Tavily** integration.
- Modular state management with **LangGraph**.

---

## How It Works

The Gardener Assistant uses a state-driven flow to provide personalized results:

1. **Input**: The user enters a gardening-related question or query.
2. **Process**:
   - Garden analysis interprets the query.
   - Relevant web search results are fetched.
   - Recommendations are generated based on the query and results.
   - Final comprehensive advice is prepared.
3. **Output**: The results are displayed in an interactive web interface.

---

## Installation

### Prerequisites
- **Python 3.8 or higher**.
- **API keys** for:
  - Tavily (`TAVILY_API_KEY`)
  - LangSmith (`LANGSMITH_API_KEY`)
  - Google Generative AI (`GEMINI_API_KEY`)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gardener-assistant.git
   cd gardener-assistant
