from langgraph.graph import StateGraph, END
from state import MentalState
from agents import perception_node, knowledge_node, reasoning_node, audit_node

def should_continue(state):
    if "PASS" not in state["audit_comment"].upper() and state.get("retry_count", 0) < 2:
        return "retry"
    return "end"

workflow = StateGraph(MentalState)
workflow.add_node("perception", perception_node)
workflow.add_node("knowledge", knowledge_node)
workflow.add_node("reasoning", reasoning_node)
workflow.add_node("audit", audit_node)

workflow.set_entry_point("perception")
workflow.add_edge("perception", "knowledge")
workflow.add_edge("knowledge", "reasoning")
workflow.add_edge("reasoning", "audit")

workflow.add_conditional_edges("audit", should_continue, {"retry": "reasoning", "end": END})

app = workflow.compile()