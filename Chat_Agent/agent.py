import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain_core.messages import trim_messages
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
api_key = os.getenv("TAVILY_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize the ChatGroq LLM
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    api_key=groq_api_key
)

# Test the LLM connection (optional)
response = llm.invoke("Hi, I am Huzaifa")
print("LLM Test Response:", response)

# Initialize TavilySearchResults tool
tool = TavilySearchResults(max_results=2)
tools = [tool]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Define State using TypedDict
class State(TypedDict):
    messages: Annotated[list, add_messages]
    


# Initialize StateGraph
workflow = StateGraph(State)

# Define assistant function
def assistant(state: State):
    messages = trim_messages(
        state["messages"],
        max_tokens=100,
        strategy="last",
        token_counter=ChatGroq(model="llama-3.1-70b-versatile"),
        allow_partial=False,
    )
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Add nodes to the workflow
workflow.add_node("assistant", assistant)
tool_node = ToolNode(tools=[tool])
workflow.add_node("tools", tool_node)

# Define edges in the workflow
workflow.add_edge(START, "assistant")
workflow.add_conditional_edges("assistant", tools_condition)
workflow.add_edge("tools", "assistant")

# Compile the graph with memory checkpointing
memory = MemorySaver()
graph:CompiledStateGraph = workflow.compile(checkpointer=memory)

# Display the graph visualization (ensure this method is supported)
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print("Graph visualization error:", e)

# Define configuration and user input
config = {"configurable": {"thread_id": "1"}}
user_input = "What is my name? Do you remember?"

# Stream user input through the graph
try:
    events = graph.stream(
        {"messages": [HumanMessage(content=user_input)]}, config, stream_mode="values"
    )
    for event in events:
        print("Response:", event["messages"][-1].content)
except Exception as e:
    print("Error during execution:", e)