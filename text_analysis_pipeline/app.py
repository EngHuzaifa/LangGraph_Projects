import os
from typing_extensions import TypedDict
import streamlit as st
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from dotenv import load_dotenv



# Streamlit Title
st.title("Text Analysis Pipeline with LangGraph and Streamlit")

load_dotenv()
groq_api_key: str = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.7,
    max_tokens=500,
    api_key=groq_api_key,
)



# Define the TextAnalysisState type
class TextAnalysisState(TypedDict):
    text: str
    classification: str
    entities: str
    summary: str

# Node 1: Classification
def classification_node(state: TextAnalysisState) -> TextAnalysisState:
    text: str = state['text']
    prompt: str = """
    You are given a Text as input.
    You need to categorize the text into one domain: NEWS, ARTICLE, BLOG.
    Just return the Category. For example: for news classification, return NEWS.
    Text is: {text}
    Note: please  make sure that you do not return  code just category name
    """
    output = llm.invoke(prompt.format(text=text))
    state["classification"] = output.content.strip()
    return state

# Node 2: Entity Extraction
def entity_extraction(state: TextAnalysisState) -> TextAnalysisState:
    text: str = state['text']
    classification: str = state['classification']
    prompt: str = """
    You are given a Text as input.
    You have to extract the entities present in the text.
    Return the names of the entities present in JSON format i.e: {{"entities": ["entity1", "entity2", ...]}}
    Text is: {text}
    Classification is: {classification}
    Note: please  make sure that you do not return  code just entities names
    """
    output = llm.invoke(prompt.format(text=text, classification=classification))
    print(output)
    state["entities"] = output.content.strip()
    return state


# Node 3: Text Summarization
def text_summarization(state: TextAnalysisState) -> TextAnalysisState:
    text: str = state['text']
    classification: str = state['classification']
    entities: str = state['entities']
    prompt: str = """
    You are given a Text as input.
    You have to generate a summary of the text under 150 words.
    Text is: {text}
    Classification is: {classification}
    Entities are: {entities}
    Note: please  make sure that you do not return  code just summary
    """
    output = llm.invoke(prompt.format(text=text, classification=classification, entities=entities))
    state["summary"] = output.content.strip()
    return state

# Build the StateGraph
text_pipeline_builder: StateGraph = StateGraph(TextAnalysisState)

# Add nodes
text_pipeline_builder.add_node("classification_node", classification_node)
text_pipeline_builder.add_node("entity_extraction", entity_extraction)
text_pipeline_builder.add_node("text_summarization", text_summarization)

# Add edges
text_pipeline_builder.add_edge(START, "classification_node")
text_pipeline_builder.add_edge("classification_node", "entity_extraction")
text_pipeline_builder.add_edge("entity_extraction", "text_summarization")
text_pipeline_builder.add_edge("text_summarization", END)

# Compile the StateGraph
text_pipeline_compiled_graph: CompiledStateGraph = text_pipeline_builder.compile()

# Streamlit Input for Text
input_text = st.text_area("Enter the text you want to analyze:", height=200)

# Analyze Button
if st.button("Analyze Text"):
    # if not GEMINI_API_KEY:
    #     st.error("Please provide a GEMINI API key to proceed.")
    # elif input_text:
        # Run the pipeline with the input text
        graph_output = text_pipeline_compiled_graph.invoke({"text": input_text})
        
        # Display Outputs
        st.subheader("Analysis Results")
        st.write("**Classification**:", graph_output["classification"])
        st.write("**Entities**:", graph_output["entities"])
        st.write("**Summary**:", graph_output["summary"])
    # else:
    #     st.warning("Please enter some text to analyze.")

# To run Streamlit application, use the command:
# streamlit run main.py
