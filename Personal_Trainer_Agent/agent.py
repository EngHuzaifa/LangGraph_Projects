import os
from typing import  TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from exa_py import Exa


exa_api_key = os.getenv('EXA_API_KEY')
exa = Exa(api_key=exa_api_key)


groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize ChatGroq model
llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=groq_api_key)


class State(TypedDict):
    query: str
    category: str
    response: str

    
def categorize(state: State) -> State:
    """
    Categorize the user's query into one of the predefined categories:
    Fitness Plan, Nutrition, recommend_exercises,training_techniques,personalize_routine.
    """
    prompt = (
        "Based on the following query, categorize it into one of these categories: "
    "1) Fitness Plan, 2) Nutrition, 3) Recommend Exercises, "
    "4) Training Techniques, 5) Personalize Routine or perform web search "
    )+ state["query"]

    category = llm.invoke(prompt).content.strip()

    return {"category": category}



def recommend_exercises(state: State) -> State:
    """
    Recommend exercises based on the user's fitness goals or preferences.
    """
    prompt = (
        "Recommend a set of exercises based on the following user query. "
        "Ensure the exercises align with the user's goals or preferences: "
    ) + state["query"]

    response = llm.invoke(prompt).content.strip()
    return {"response": response}

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

def explain_training_techniques(state: State) -> State:
    """
    Explain how to perform a specific training exercise or technique.
    """
    prompt = (
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
            num_results=2,
            text=True,
            highlights=True,
            summary=True,
            category="pdf",  # Can be adjusted based on requirements
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
        # Handle errors gracefully
        return {"response": f"An error occurred during the web search: {str(e)}"}

def process_web_search_results(state: State) -> State:
    """
    Use the LLM to analyze and respond to the web search results.
    """
    perform_web_search= state.get("perform_web_search", "No results to process.")
    query = state["query"]

    # Combine user query and web search results for LLM processing
    prompt = (
        f"The user asked: '{query}'.\n"
        f"Here are the web search results:\n{result}\n"
        "Based on these results, provide a detailed and helpful response:"
    )

    response = llm.invoke(prompt).content.strip()
    return {"response": response}

    

# Routing Logic# Routing Logic
def route_query(state: State) -> str:
    if state.get("category") == "personalize_routine":
        return "personalize_routine"
    elif state.get("category") == "Fitness Plan":
        return "handle_fitness_plan"
    elif state.get("category") == "Nutrition":
        return "handle_nutrition"
    elif state.get("category") == "recommend_exercises":
        return "recommend_exercises"
    elif state.get("category") == "training_techniques":
        return "explain_training_techniques"
    else:
      return "perform_web_search"

    


# Workflow Setup
workflow = StateGraph(State)

# Add nodes
workflow.add_node("categorize", categorize)
workflow.add_node("handle_fitness_plan", handle_fitness_plan)
workflow.add_node("handle_nutrition", handle_nutrition)
workflow.add_node("explain_training_techniques", explain_training_techniques)
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
        "handle_fitness_plan": "handle_fitness_plan",
        "handle_nutrition": "handle_nutrition",
        "recommend_exercises": "recommend_exercises",
        "explain_training_techniques": "explain_training_techniques",
        "personalize_routine": "personalize_routine",
        "perform_web_search": "perform_web_search"
    }

    )
workflow.add_edge("handle_fitness_plan",END)
workflow.add_edge("handle_nutrition",END)
workflow.add_edge("recommend_exercises",END)
workflow.add_edge("explain_training_techniques",END)
workflow.add_edge("personalize_routine",END)
workflow.add_edge("perform_web_search", "process_web_search_results")
workflow.add_edge("process_web_search_results",END)






# Compile Workflow
memory = MemorySaver()
graph: CompiledStateGraph = workflow.compile(interrupt_before=["perform_web_search"],checkpointer=memory)





