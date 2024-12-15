import os
from langchain_groq import ChatGroq
from typing import Annotated
from langgraph.graph import StateGraph, START, MessagesState
from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages
#from typing_extensions import TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode



tavily_api_key = os.getenv('TAVILY_API_KEY')

groq_api_key = os.getenv('GROQ_API_KEY')


llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    api_key=groq_api_key
)
llm.invoke("hi I am Huzaifa")

class State(MessagesState):
    messages: Annotated[list, add_messages]


builder = StateGraph(State) # Provide state_schema

tool = TavilySearchResults(
    max_results=2,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True
    )

tools = [tool]
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="you are a helpfull assistant")
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}




builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")




memory = MemorySaver()
graph: CompiledStateGraph = builder.compile(checkpointer=memory)
