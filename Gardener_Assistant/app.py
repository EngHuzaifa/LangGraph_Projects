import os
from pydantic import BaseModel
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient
import streamlit as st
from dotenv import load_dotenv

# Set environment variables
load_dotenv()
api_key = os.getenv("TAVILY_API_KEY")
gemini_api_key=os.getenv('GEMINI_API_KEY')


# Initialize ChatGoogleGenerativeAI and Tavily Search
llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=gemini_api_key,
)
tavily_client = TavilyClient()
tavily_search = TavilySearchResults(max_results=1, client=tavily_client)

# Define the GardenerState model
class GardenerState(BaseModel):
    user_query: str
    garden_analys: str
    plant_recomend: str
    final: str
    search_results: str

# Define analysis and recommendation functions
def garden_analysis(state: GardenerState) -> Dict[str, Any]:
    prompt = f"""
    You are given a user query about gardening.
    Analyze the query and determine the user's gardening needs or problems.
    Provide a concise summary of the garden analysis based on the query.

    User Query: {user_query}
    """
    output = llm.invoke(prompt)
    return {"garden_analys": output.content}

def web_search(state: GardenerState) -> Dict[str, Any]:
    search_results_raw = tavily_search.invoke(state.user_query)
    search_results = "\n".join(
        [f" URL: {item.get('url', 'N/A')}" for item in search_results_raw]
    )
    return {"search_results": search_results}

def plant_recommendations(state: GardenerState) -> Dict[str, Any]:
    prompt = f"""
    You are a gardening expert.
    You are given a garden analysis and search results.
    Recommend plants based on the analysis and results.
    Provide a concise summary of the plant recommendations.

    Garden Analysis: {state.garden_analys}
    Search Results: {state.search_results}
    """
    output = llm.invoke(prompt)
    return {"plant_recomend": output.content}

def final_advise(state: GardenerState) -> Dict[str, Any]:
    prompt = f"""
    You are a gardening expert.
    You are given a garden analysis, plant recommendations, and search results.
    Provide comprehensive gardening advice based on the analysis, recommendations, and results.
    Consider factors like climate, soil type, and potential challenges.
    Provide a concise summary of the final advice.

    Garden Analysis: {state.garden_analys}
    Plant Recommendations: {state.plant_recomend}
    Search Results: {state.search_results}
    """
    output = llm.invoke(prompt)
    return {"final": output.content}

# Create the state graph
gardener_state_graph = StateGraph(GardenerState)
gardener_state_graph.add_node(garden_analysis)
gardener_state_graph.add_node(web_search)
gardener_state_graph.add_node(plant_recommendations)
gardener_state_graph.add_node(final_advise)

gardener_state_graph.add_edge(START, "garden_analysis")
gardener_state_graph.add_edge("garden_analysis", "web_search")
gardener_state_graph.add_edge("web_search", "plant_recommendations")
gardener_state_graph.add_edge("plant_recommendations", "final_advise")
gardener_state_graph.add_edge("final_advise", END)

compiled_state_graph:CompiledStateGraph = gardener_state_graph.compile()

# Streamlit UI
st.title("🌱 Gardener Assistant")

user_query = st.text_input("Enter your gardening query:")
if st.button("Submit"):
    if user_query.strip():
        # Execute state graph
        input_data = GardenerState(
            user_query=user_query,
            garden_analys="",
            plant_recomend="",
            final="",
            search_results="",
        )
        result = compiled_state_graph.invoke(input=input_data)

        # Display results
        st.subheader("Garden Analysis:")
        st.write(result.get("garden_analys", "No analysis available."))

        st.write(result.get("search_results", "No search results available."))

        st.subheader("Plant Recommendations:")
        st.write(result.get("plant_recomend", "No recommendations available."))

        st.subheader("Final Advice:")
        st.write(result.get("final", "No advice available."))
    else:
        st.warning("Please enter a query to proceed.")
