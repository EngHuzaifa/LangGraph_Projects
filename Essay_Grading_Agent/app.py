import os
import matplotlib.pyplot as plt
from langchain_groq import ChatGroq
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
import streamlit as st

# Install necessary libraries
#pip install -q -U langgraph langsmith langchain streamlit langchain_groq

# Set environment variables

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Build Essay Grading agent"

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.7,
    max_tokens=500,
    api_key=GROQ_API_KEY
)

# Define state
class EssayGradingState(TypedDict):
    text: str
    grade: str
    feedback: str

# Define grading, feedback, and graph functions
def grade_essay(state: EssayGradingState) -> EssayGradingState:
    text = state['text']
    prompt = f"""
      You are an essay grader.
      Grade the following essay:

      {text}

      Provide a total grade out of 100.
    """
    output = llm.invoke(prompt)
    return {"grade": output.content.strip()}

def feedback_essay(state: EssayGradingState) -> EssayGradingState:
    text = state['text']
    prompt = f"""
    You are an essay grader. Provide feedback on the following essay:

    {text}

    Provide feedback with these sections:
    Strengths: Highlight the essay's strengths (e.g., clear argument, strong evidence).
    Areas for Improvement: Point out areas needing improvement (e.g., structure, clarity, grammar).
    Suggestions: Offer actionable advice for how to strengthen the essay.
    """
    output = llm.invoke(prompt)
    return {"feedback": output.content.strip()}

def show_grade_graph(state: EssayGradingState) -> EssayGradingState:
    text = state['text']
    prompt = """
    Based on the essay content, simulate a distribution of grades for similar essays.
    Provide the grades as a comma-separated string (e.g., 85,90,78,92,88).
    """
    output = llm.invoke(prompt)
    grades_str = output.content.strip()
    try:
        grades = [int(grade.strip()) for grade in grades_str.split(',') if grade.strip().isdigit()]
        plt.figure(figsize=(10, 6))
        plt.hist(grades, bins=10, edgecolor='black', alpha=0.8, color='skyblue')
        plt.xlabel('Grades (out of 100)')
        plt.ylabel('Number of Essays')
        plt.title('Essay Grading Distribution')
        st.pyplot(plt)
        return {"grades": grades, "grade": str(sum(grades) // len(grades))}
    except ValueError as e:
        return {"grades": None, "grade": f"Error: {e}"}

# Streamlit UI
st.title("Essay Grading App")
essay_text = st.text_area("Enter your essay below:", height=200)

if st.button("Grade Essay"):
    # Build the state graph
    essay_grading_builder = StateGraph(EssayGradingState)
    essay_grading_builder.add_node("grade_essay", grade_essay)
    essay_grading_builder.add_node("feedback_essay", feedback_essay)
    essay_grading_builder.add_node("show_grade_graph", show_grade_graph)
    essay_grading_builder.add_edge(START, "grade_essay")
    essay_grading_builder.add_edge("grade_essay", "feedback_essay")
    essay_grading_builder.add_edge("feedback_essay", "show_grade_graph")
    essay_grading_builder.add_edge("show_grade_graph", END)
    essay_grading_graph = essay_grading_builder.compile()

    # Invoke the graph
    graph_output = essay_grading_graph.invoke({"text": essay_text})

    # Display results
    st.subheader("Grade")
    st.write(graph_output.get("grade", "No grade available."))

    st.subheader("Feedback")
    st.write(graph_output.get("feedback", "No feedback available."))
