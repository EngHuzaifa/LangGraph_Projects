import pytest
from langsmith import unit

from react_agent import graph


@pytest.mark.asyncio
@unit
async def test_react_agent_simple_passthrough() -> None:
    res = await graph.ainvoke(
        {"messages": [("user", "hi?")]},
        {
            "configurable": {
                "system_prompt": "You are a helpful AI assistant.",
                "model": "gemini/gemini-2.0-flash",
            }
        },
    )

    assert len(str(res["messages"][-1].content).lower()) > 0
