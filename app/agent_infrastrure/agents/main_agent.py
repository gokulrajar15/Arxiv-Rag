import asyncio
from datetime import datetime
from typing import Dict, Any
from langchain_core.messages import AnyMessage

from langgraph.prebuilt import create_react_agent

from app.agent_infrastrure.prompt_templates import agent_prompt_template
from app.agent_infrastrure.infrastructure.llm_clients import gpt_41
from app.agent_infrastrure.agents.document_retriver import document_retriever
from app.schema.langgraph_agent_states import MAINAGENTSTATE
from app.db.client import Database as pg_client

def prompt(state: MAINAGENTSTATE) -> list[AnyMessage]:
    """
    Generate a prompt for the agent based on the current state.
    
    Args:
        state (MAINAGENTSTATE): The current state of the agent
        
    Returns:
        list[AnyMessage]: The system message and user messages
    """
    system_msg = agent_prompt_template(state["date"])
    return system_msg + state["messages"]

main_agent = create_react_agent(
    model=gpt_41,
    tools=[document_retriever],
    prompt=prompt,
    state_schema=MAINAGENTSTATE
)


async def main():
    await pg_client.init()
    current_date = datetime.now().strftime("%Y-%m-%d")
    response = await main_agent.ainvoke(    
        {
            "messages": [
                {
                    "role": "user",
                    "content": "brownian motion"
                }
            ],
            "date": current_date
        }
    )
    
    print(response["messages"][-1].content)
    await pg_client.close()


# asyncio.run(main())
