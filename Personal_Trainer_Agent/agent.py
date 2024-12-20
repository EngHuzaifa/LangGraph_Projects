import os
#from google import genai
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langgraph.graph.state import CompiledStateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq






groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize ChatGroq model
llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=groq_api_key)


class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str

    
def categorize(state: State) -> State:
    """
    Categorize the user's query into one of the predefined categories:
    Fitness Plan, Nutrition, or General.
    """
    prompt = (
        "Please categorize the following query into one of these categories: "
        "Fitness Plan, Nutrition, or General. "
    )+ state["query"]

    category = llm.invoke(prompt).content.strip()

    return {"category": category}

def analyze_sentiment(state: State) -> State:
    """
    Analyze the sentiment of a user query and classify it as Positive, Neutral, or Negative.

    Args:
        state (State): A dictionary containing the user query under the key "query".

    Returns:
        State: A dictionary containing the sentiment classification under the key "sentiment".
    """
    prompt = (
        "Analyze the sentiment of the following user query. "
        "Classify it as 'Positive', 'Neutral', or 'Negative'. Query: "
    ) + state["query"]

    sentiment = llm.invoke(prompt).content.strip()

    return {"sentiment": sentiment}

def handle_fitness_plan(state: State) -> State:
    """
    Provide a fitness plan recommendation based on the user's query.
    """
    prompt = (
        "Based on the user's query, recommend a suitable fitness plan or workout routine. "
        "Ensure the response is personalized and practical. "
    ) + state["query"]

    response = llm.invoke(prompt).content.strip()

    return {"response": response}

def handle_nutrition(state: State) -> State:
    """
    Provide a nutrition plan or dietary advice based on the user's query.
    """
    prompt = (
        "Based on the user's query, recommend a suitable nutrition plan or dietary advice. "
        "Ensure the response is evidence-based and practical. "
    )  + state["query"]

    response = llm.invoke(prompt).content.strip()

    return {"response": response}

def handle_general(state: State) -> State:
    """
    Provide a response to a general query about fitness, health, or training.
    """
    prompt = (
        "Provide a response to the following general query about fitness, health, or training. "
    
    ) + state["query"]

    response = llm.invoke(prompt).content.strip()

    return {"response": response}

def escalate(state: State) -> State:
    """
    Escalate the query to a human fitness expert for detailed guidance.
    """
    return {
        "response": "This query has been escalated to a human fitness expert for detailed guidance."
    }

# Routing Logic
def route_query(state: State) -> str:
    if state.get("sentiment") == "Negative":
        return "escalate"
    elif state.get("category") == "Fitness Plan":
        return "handle_fitness_plan"
    elif state.get("category") == "Nutrition":
        return "handle_nutrition"
    else:
        return "handle_general"

# Workflow Setup
workflow = StateGraph(State)

# Add nodes
workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("handle_fitness_plan", handle_fitness_plan)
workflow.add_node("handle_nutrition", handle_nutrition)
workflow.add_node("handle_general", handle_general)
workflow.add_node("escalate", escalate)

# Set entry point
workflow.set_entry_point("categorize")
workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_conditional_edges(
    "analyze_sentiment",
    route_query,
    {
        "handle_fitness_plan": "handle_fitness_plan",
        "handle_nutrition": "handle_nutrition",
        "handle_general": "handle_general",
        "escalate": "escalate"
    }
)
workflow.add_edge("handle_fitness_plan", END)
workflow.add_edge("handle_nutrition", END)
workflow.add_edge("handle_general", END)
workflow.add_edge("escalate", END)

# Compile Workflow
memory = MemorySaver()
graph: CompiledStateGraph = workflow.compile(checkpointer=memory)






