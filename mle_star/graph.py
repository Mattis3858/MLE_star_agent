"""LangGraph wiring: research -> foundation -> refine loop -> finalize -> report."""

from langgraph.graph import END, StateGraph

from . import agents


def build():
    g = StateGraph(agents.AgentState)
    g.add_node("research", agents.research_node)
    g.add_node("foundation", agents.foundation_node)
    g.add_node("refine", agents.refine_node)
    g.add_node("finalize", agents.finalize_node)
    g.add_node("report", agents.report_node)

    g.set_entry_point("research")
    g.add_edge("research", "foundation")
    g.add_edge("foundation", "refine")
    g.add_conditional_edges(
        "refine", agents.should_continue, {"refine": "refine", "finalize": "finalize"}
    )
    g.add_edge("finalize", "report")
    g.add_edge("report", END)
    return g.compile()
