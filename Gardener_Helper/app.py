import os
from typing_extensions import TypedDict
import streamlit as st
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI

# Streamlit Title
st.title("Gardening Assistant")

# Load environment variables
load_dotenv()
groq_api_key: str = os.getenv("GEMINI_API_KEY")
tavily_api_key: str = os.getenv("TAVILY_API_KEY")

# Initialize LLM and Tavily Client
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    api_key=groq_api_key,
)

tavily_client = TavilyClient(api_key=tavily_api_key)
search_tool = TavilySearchResults(max_results=3, client=tavily_client)

# Define the GardenerState type
class GardenerState(TypedDict):
    user_query: str
    garden_analysis: str
    plant_recommendations: str
    final_advice: str
    search_results: str

# Node 1: Understand the query
def understand_query(state: GardenerState) -> GardenerState:
    user_query: str = state['user_query']
    prompt: str = f"""
    Based on the user's gardening query, extract details in JSON format:
    
    User Query: {user_query}

    {{
        "garden_type": "<type of garden (vegetable/ornamental/mixed)>",
        "climate_zone": "<location or climate zone>",
        "available_space": "<size/space description>",
        "primary_goals": ["<goal 1>", "<goal 2>", ...],
        "specific_challenges": ["<challenge 1>", "<challenge 2>", ...]
    }}
    """
    output = llm.invoke(prompt)
    state["garden_analysis"] = output.content.strip()
    return state

# Node 2: Perform Web Search using Tavily
def perform_web_search(state: GardenerState) -> GardenerState:
    user_query: str = state['user_query']
    try:
        search_results = search_tool.invoke(user_query)
        # Format the search results into a string for display
        formatted_results = "\n".join([f"**{result['title']}**: {result['url']}\n{result['content']}" for result in search_results])
        state["search_results"] = formatted_results
    except Exception as e:
        state["search_results"] = f"Search error: {e}"
    return state

# Node 3: Analyze Garden
def analyze_garden(state: GardenerState) -> GardenerState:
    garden_analysis: str = state['garden_analysis']
    prompt: str = f"""
    Analyze this garden situation:
    
    {garden_analysis}
    
    Provide a detailed analysis covering suitability, challenges, and recommendations.
    """
    output = llm.invoke(prompt)
    state["garden_analysis"] = output.content.strip()  # Updating the state with analysis
    return state

# Node 4: Recommend Plants
def recommend_plants(state: GardenerState) -> GardenerState:
    garden_analysis: str = state['garden_analysis']
    prompt: str = f"""
    Based on this analysis, recommend 3 plants:
    
    {garden_analysis}
    
    Provide recommendations in this format:
    recommendations: [
    {{
        name: "Plant Name",
        sunlight_requirements: "Sunlight needs",
        soil_type: "Soil requirements",
        care_difficulty: "Difficulty level",
        reasons: ["Reason 1", "Reason 2"]
    }}
]

    """
    output = llm.invoke(prompt)
    state["plant_recommendations"] = output.content.strip()
    return state

# Node 5: Generate Final Advice
def generate_final_advice(state: GardenerState) -> GardenerState:
    garden_analysis: str = state['garden_analysis']
    plant_recommendations: str = state['plant_recommendations']
    prompt: str = f"""
    Based on the following:
    
    Garden Analysis: {garden_analysis}
    Recommended Plants: {plant_recommendations}
    
    Provide final gardening advice including:
    - Planting instructions
    - Care guidelines
    - Seasonal tips
    - Challenge solutions
    """
    output = llm.invoke(prompt)
    state["final_advice"] = output.content.strip()
    return state

# Build the StateGraph
gardening_pipeline_builder: StateGraph = StateGraph(GardenerState)

# Add nodes to the pipeline
gardening_pipeline_builder.add_node("understand_query", understand_query)
gardening_pipeline_builder.add_node("perform_web_search", perform_web_search)
gardening_pipeline_builder.add_node("analyze_garden", analyze_garden)
gardening_pipeline_builder.add_node("recommend_plants", recommend_plants)
gardening_pipeline_builder.add_node("generate_final_advice", generate_final_advice)

# Add edges between nodes
gardening_pipeline_builder.add_edge(START, "understand_query")
gardening_pipeline_builder.add_edge("understand_query", "perform_web_search")
gardening_pipeline_builder.add_edge("perform_web_search", "analyze_garden")
gardening_pipeline_builder.add_edge("analyze_garden", "recommend_plants")
gardening_pipeline_builder.add_edge("recommend_plants", "generate_final_advice")
gardening_pipeline_builder.add_edge("generate_final_advice", END)

# Compile the StateGraph
gardening_pipeline_compiled_graph: CompiledStateGraph = gardening_pipeline_builder.compile()

# Streamlit Input for Gardening Query
user_query = st.text_area("Describe your gardening scenario:", height=200)

# Analyze Button
if st.button("Get Gardening Advice"):
    if user_query:
        # Run the gardening pipeline
        graph_output = gardening_pipeline_compiled_graph.invoke({"user_query": user_query})
        
        # Display the Results
        st.subheader("Gardening Advice")
        st.write("**Garden Analysis**:", graph_output["garden_analysis"])
        st.write("**Recommended Plants**:", graph_output["plant_recommendations"])
        st.write("**Final Advice**:", graph_output["final_advice"])
        
        
    else:
        st.warning("Please enter a gardening query.")
