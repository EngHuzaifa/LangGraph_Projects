import os
import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage, AIMessage
from pydantic import BaseModel


# Set environment variables

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Build_Chatbot"
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "your-tavily-api-key")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.getenv("GEMINI_API_KEY", "AIzaSyAtO6pecQOisVOd8JRmpcE6Bmde9sApPKs"))

# Define the state class
class State(TypedDict):
    messages: Annotated[list, add_messages]
    ask_human: bool

# Define a RequestAssistance tool for escalation
class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert."""
    request: str

# Initialize LangChain tools
tool = TavilySearchResults(max_results=3)
tools = [tool]
llm_with_tools = llm.bind_tools(tools + [RequestAssistance])

# Define chatbot function
def chatbot(state: State):
    # Ensure state["messages"] is valid and non-empty
    if not state.get("messages"):
        raise ValueError("Messages cannot be empty when invoking the LLM.")
    
    # Ensure all messages have content
    for msg in state["messages"]:
        if isinstance(msg, tuple):
            role, content = msg
            if not content.strip():
                raise ValueError("Message content cannot be empty.")
        elif hasattr(msg, "content") and not msg.content.strip():
            raise ValueError("Message content cannot be empty.")
    
    # Invoke the LLM
    response = llm_with_tools.invoke(state["messages"])
    ask_human = False

    # Check if the response includes a tool call for human assistance
    if (
        response.tool_calls
        and response.tool_calls[0]["name"] == RequestAssistance.__name__
    ):
        ask_human = True

    return {"messages": [response], "ask_human": ask_human}


# Graph definition and compilation
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=[tool]))

def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(content=response, tool_call_id=ai_message.tool_calls[0]["id"])

def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        new_messages.append(create_response("No response from human.", state["messages"][-1]))
    return {"messages": new_messages, "ask_human": False}

def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    return tools_condition(state)

graph_builder.add_node("human", human_node)
graph_builder.add_conditional_edges("chatbot", select_next_node, {"human": "human", "tools": "tools", END: END})
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["human"])

# Streamlit UI
st.title(" Chatbot with LangGraph")

# Sidebar for instructions
st.sidebar.title("Instructions")
st.sidebar.write("This chatbot uses LangGraph and integrates with LangChain tools to provide intelligent responses.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input
user_input = st.text_input("Type your message:", key="input")

# Handle chatbot interaction
if st.button("Send") and user_input:
    st.session_state.messages.append(("user", user_input))
    config = {"configurable": {"thread_id": "1"}}
    events = graph.stream({"messages": st.session_state.messages}, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            st.session_state.messages.append(event["messages"][-1])

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, tuple):  # For user input messages
        role, content = message
        if role == "user":
            st.write(f"**You:** {content}")
    elif isinstance(message, AIMessage):  # For AI-generated messages
        st.write(f"**Chatbot:** {message.content}")
    elif isinstance(message, ToolMessage):  # For tool responses
        st.write(f"**Tool ({message.name}):** {message.content}")

