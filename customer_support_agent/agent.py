import os
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END


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
    Categorize the customer query into one of the predefined categories:
    Technical, Billing, or General.

    Args:
        state (State): A dictionary containing the customer query under the key "query".

    Returns:
        State: A dictionary containing the categorized result under the key "category".
    """
    prompt = (
        "Please categorize the following customer query into one of these categories: "
        "Technical, Billing, or General. Query: "
    ) + state["query"]

    # Invoke the chain with a string input
    category = llm.invoke(prompt).content.strip()

    return {"category": category}

def analyze_sentiment(state: State) -> State:
    """
    Analyze the sentiment of a customer query and classify it as Positive, Neutral, or Negative.

    Args:
        state (State): A dictionary containing the customer query under the key "query".

    Returns:
        State: A dictionary containing the sentiment classification under the key "sentiment".
    """
    prompt = (
        "Analyze the sentiment of the following customer query. "
        "Classify it as 'Positive', 'Neutral', or 'Negative'. Query: "
    ) + state["query"]

    sentiment = llm.invoke(prompt).content.strip()

    return {"sentiment": sentiment}

def handle_technical(state: State) -> State:
    """
    Generate a technical support response to a customer query.
    Ensure the Tavily search tool is invoked if the LLM response is insufficient.
    """
    prompt = (
        "Provide a technical support if customer ask latest updated  you call the tool and provide latest news about technical response to the following customer query: "
    ) + state["query"]

    response = llm.invoke(prompt).content.strip()



    return {"response": response}

def handle_billing(state: State) -> State:
    """
    Generate a billing support response to a customer query.

    Args:
        state (State): A dictionary containing the customer query under the key "query".

    Returns:
        State: A dictionary containing the generated billing support response under the key "response".
    """
    prompt = (
        "Provide a billing support if customer ask latest updated  you call the tool and provide latest news about billing response to the following customer query: "
    ) + state["query"]

    response = llm.invoke(prompt).content.strip()

    return {"response": response}

def handle_general(state: State) -> State:
    """
    Generate a general support response to a customer query.
    Ensure the Tavily search tool.
    """
    prompt = (
        "Provide a general support if customer ask latest updated  you call the tool and provide latest news about general query response to the following customer query: "
    ) + state["query"]

    response = llm.invoke(prompt).content.strip()


    return {"response": response}

def escalate(state: State) -> State:
    """
    Escalate the query to a human agent.
    Always invoke the Tavily search tool to provide additional context.
    """

    return {
        "response": "This query has been escalated to a human agent due to its negative sentiment. "

    }

# Update Routing Logic
def route_query(state: State) -> str:
    if state.get("sentiment") == "Negative":
        return "escalate"
    elif "FAQ" in state["query"]:  # Example: Trigger web scraping for FAQ-related queries
        return "perform_web_scraping"
    elif state.get("category") == "Technical":
        return "handle_technical"
    elif state.get("category") == "Billing":
        return "handle_billing"
    else:
        return "handle_general"

# Create Workflow
workflow = StateGraph(State)
# Add nodes
workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("handle_billing", handle_billing)
workflow.add_node("handle_general", handle_general)
workflow.add_node("escalate", escalate)

# Set entry point
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

# Compile Workflow
memory = MemorySaver()
graph: CompiledStateGraph = workflow.compile(checkpointer=memory)
