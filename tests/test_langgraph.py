from typing import TypedDict
from langgraph.graph import StateGraph

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="deepseek-r1:8b",
    temperature=0.2
)

class State(TypedDict):
    input: str
    step1: str
    output: str

def agent_a(state: State):
    return {
        "step1": "A processed: " + state["input"]
    }

def agent_b(state: State):
    return {
        "output": "B got: " + state["step1"]
    }

builder = StateGraph(State)

builder.add_node("A", agent_a)
builder.add_node("B", agent_b)

builder.set_entry_point("A")

builder.add_edge("A", "B")

builder.set_finish_point("B")

graph = builder.compile()

response = llm.invoke("Say hello")

print(response.content)