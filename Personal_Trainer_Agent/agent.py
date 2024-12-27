import os
from typing import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from exa_py import Exa
from langchain_core.messages import SystemMessage

# Global system message
SYSTEM_MESSAGE = SystemMessage(content=(
    "You are a fitness and nutrition assistant. Your goal is to provide helpful "
    "recommendations and responses related to fitness plans, nutrition advice, "
    "exercise recommendations, training techniques, and personalized routines. "
    "If a user's query requires information from external sources, perform a web search "
    "and summarize the findings clearly. Always ensure your responses are user-friendly and informative."
))

exa_api_key = os.getenv('EXA_API_KEY')
exa = Exa(api_key=exa_api_key)

gemini_api_key= os.getenv('GOOGLE_API_KEY')
# Initialize ChatGroq model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=gemini_api_key,temperature=0.1)


class State(TypedDict): 
    query: str
    category: str
    response: str

def categorize(state: State) -> State:
    """
    Categorize the user's query into one of the predefined categories:
    Fitness Plan, Nutrition, recommend_exercises, training_techniques, personalize_routine.
    """
    prompt = (
        f"{SYSTEM_MESSAGE.content}\n"
        "Please categorize the following query into one of these categories: "
        "Fitness Plan, Nutrition, recommend_exercises, training_techniques, personalize_routine. "
        "If the query cannot be categorized, you may choose 'perform_web_search'. Query: "
    ) + state["query"]

    category = llm.invoke(prompt).content.strip()
    return {"category": category}

def recommend_exercises(state: State) -> State:
    """
    Recommend exercises based on the user's fitness goals or preferences.
    """
    prompt = (
        f"{SYSTEM_MESSAGE.content}\n"
        "Recommend a set of exercises based on the following user query. "
        "Ensure the exercises align with the user's goals or preferences: "
    ) + state["query"]

    response = llm.invoke(prompt).content.strip()
    return {"response": response}

def fitness_plan(state: State) -> State:
    """
    Provide a fitness plan recommendation based on the user's query.
    """
    prompt = (
        f"{SYSTEM_MESSAGE.content}\n"
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
        f"{SYSTEM_MESSAGE.content}\n"
        "Based on the user's query, recommend a suitable nutrition plan or dietary advice. "
        "Ensure the response is evidence-based and practical. "
    ) + state["query"]

    response = llm.invoke(prompt).content.strip()
    return {"response": response}

def training_techniques(state: State) -> State:
    """
    Explain how to perform a specific training exercise or technique.
    """
    prompt = (
        f"{SYSTEM_MESSAGE.content}\n"
        "Provide a clear and detailed explanation of how to perform the following exercise or "
        "training technique safely and effectively: "
    ) + state["query"]

    response = llm.invoke(prompt).content.strip()
    return {"response": response}

def personalize_routine(state: State) -> State:
    """
    Create a personalized training routine based on the user's preferences or fitness level.
    """
    prompt = (
        f"{SYSTEM_MESSAGE.content}\n"
        "Design a personalized training routine for the user based on the following preferences or "
        "fitness level: "
    ) + state["query"]

    response = llm.invoke(prompt).content.strip()
    return {"response": response}

def perform_web_search(state: State) -> State:
    """
    Perform a web search based on the user's query using the Exa API.
    """
    try:
        query = state["query"]
        result = exa.search_and_contents(
            query,
            num_results=1,
            text=True,
            highlights=True,
            #summary=True,
            category="news",  # Can be adjusted based on requirements
            start_crawl_date="2024-11-20T09:39:37.885Z",
            end_crawl_date="2024-12-20T09:39:37.885Z",
            subpages=1,
            extras={
                "links": 1,
                "image_links": 1
            }
        )
        return {"response": result}
    except Exception as e:
        return {"response": f"An error occurred during the web search: {str(e)}"}

def process_web_search_results(state: State) -> State:
    """
    Use the LLM to analyze and respond to the web search results.
    """
    web_search_results = state.get("response", "No results to process.")
    query = state["query"]

    if web_search_results == "No results to process.":
        prompt = (
            f"{SYSTEM_MESSAGE.content}\n"
            f"The user asked: '{query}'.\n"
            "Unfortunately, no relevant results were found from the web search. "
            "Please provide a helpful response based on your knowledge."
        )
    else:
        prompt = (
            f"{SYSTEM_MESSAGE.content}\n"
            f"The user asked: '{query}'.\n"
            f"Here are the web search results:\n{web_search_results}\n"
            "Based on these results, provide a detailed and helpful response:"
        )

    response = llm.invoke(prompt).content.strip()
    return {"response": response}

def route_query(state: State) -> str:
    category = state.get("category")
    if category == "personalize_routine":
        return "personalize_routine"
    elif category == "Fitness Plan":
        return "fitness_plan"
    elif category == "Nutrition":
        return "handle_nutrition"
    elif category == "recommend_exercises":
        return "recommend_exercises"
    elif category == "training_techniques":
        return "training_techniques"
    else:
        return "perform_web_search"

# Workflow Setup
workflow = StateGraph(State)

# Add nodes
workflow.add_node("categorize", categorize)
workflow.add_node("fitness_plan", fitness_plan)
workflow.add_node("handle_nutrition", handle_nutrition)
workflow.add_node("training_techniques", training_techniques)
workflow.add_node("personalize_routine", personalize_routine)
workflow.add_node("perform_web_search", perform_web_search)
workflow.add_node("process_web_search_results", process_web_search_results)
workflow.add_node("recommend_exercises", recommend_exercises)

# Set entry point
workflow.set_entry_point("categorize")
workflow.add_conditional_edges(
    "categorize",
    route_query,
    {
        "fitness_plan": "fitness_plan",
        "handle_nutrition": "handle_nutrition",
        "recommend_exercises": "recommend_exercises",
        "training_techniques": "training_techniques",
        "personalize_routine": "personalize_routine",
        "perform_web_search": "perform_web_search"
    }
)
workflow.add_edge("fitness_plan", END)
workflow.add_edge("handle_nutrition", END)
workflow.add_edge("recommend_exercises", END)
workflow.add_edge("training_techniques", END)
workflow.add_edge("personalize_routine", END)
workflow.add_edge("perform_web_search", "process_web_search_results")
workflow.add_edge("process_web_search_results", END)

# Compile Workflow
memory = MemorySaver()
graph: CompiledStateGraph = workflow.compile(checkpointer=memory)
