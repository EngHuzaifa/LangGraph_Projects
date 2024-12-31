import os
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

# Retrieve API key from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize ChatGroq model
llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=groq_api_key)

# Define a system message for the customer support assistant
system_message = (
    "You are a customer support assistant specializing in technical, billing, and general inquiries. "
    "Your goal is to categorize customer queries and provide helpful, accurate responses based on their "
    "category. If the query's sentiment is negative, escalate it to a human agent. Always ensure your responses are user-friendly."
)

class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str

def categorize(state: State) -> State:
    prompt = (
        system_message + "\n"
        "Please categorize the following customer query into one of these categories: "
        "Technical, Billing, or General. Query: "
    ) + state["query"]

    category = llm.invoke(prompt).content.strip()

    return {"category": category}

def analyze_sentiment(state: State) -> State:
    prompt = (
        system_message + "\n"
        "Analyze the sentiment of the following customer query. "
        "Classify it as 'Positive', 'Neutral', or 'Negative'. Query: "
    ) + state["query"]

    sentiment = llm.invoke(prompt).content.strip()

    return {"sentiment": sentiment}

def handle_technical(state: State) -> State:
    prompt = (
        system_message + "\n"
        "Provide a technical support response to the following customer query: "
    ) + state["query"]

    response = llm.invoke(prompt).content.strip()

    return {"response": response}

def handle_billing(state: State) -> State:
    prompt = (
        system_message + "\n"
        "Provide a billing support response to the following customer query: "
    ) + state["query"]

    response = llm.invoke(prompt).content.strip()

    return {"response": response}

def handle_general(state: State) -> State:
    prompt = (
        system_message + "\n"
        "Provide a general support response to the following customer query: "
    ) + state["query"]

    response = llm.invoke(prompt).content.strip()

    return {"response": response}

def escalate(state: State) -> State:
    return {
        "response": "This query has been escalated to a human agent due to its negative sentiment."
    }

def route_query(state: State) -> str:
    if state.get("sentiment") == "Negative":
        return "escalate"
    elif "FAQ" in state["query"]:
        return "perform_web_scraping"
    elif state.get("category") == "Technical":
        return "handle_technical"
    elif state.get("category") == "Billing":
        return "handle_billing"
    else:
        return "handle_general"

# Create and configure the workflow
workflow = StateGraph(State)
workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("handle_billing", handle_billing)
workflow.add_node("handle_general", handle_general)
workflow.add_node("escalate", escalate)

workflow.set_entry_point("categorize")
workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_conditional_edges(
    "analyze_sentiment",
    route_query,
    {
        "handle_technical": "handle_technical",
        "handle_billing": "handle_billing",
        "handle_general": "handle_general",
        "escalate": "escalate"
    }
)
workflow.add_edge("handle_technical", END)
workflow.add_edge("handle_billing", END)
workflow.add_edge("handle_general", END)
workflow.add_edge("escalate", END)

# Compile the workflow with memory checkpointing
memory = MemorySaver()
graph: CompiledStateGraph = workflow.compile(checkpointer=memory)