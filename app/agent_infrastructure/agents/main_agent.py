import asyncio
from datetime import datetime
import json
from typing import Dict, Any
from langchain_core.messages import AnyMessage

from langgraph.prebuilt import create_react_agent

from app.agent_infrastructure.prompt_templates import agent_prompt_template
from app.agent_infrastructure.infrastructure.llm_clients import gpt_41
from app.agent_infrastructure.agents.document_retriver import document_retriever
from app.schema.langgraph_agent_states import MAINAGENTSTATE
from app.db.client import Database
from app.agent_infrastructure.evaluation.deepeval import  run_deep_eval

def prompt(state: MAINAGENTSTATE) -> list[AnyMessage]:
    """
    Generate a prompt for the agent based on the current state.
    
    Args:
        state (MAINAGENTSTATE): The current state of the agent
        
    Returns:
        list[AnyMessage]: The system message and user messages
    """
    system_msg = agent_prompt_template()
    return system_msg + state["messages"]

rag_agent = create_react_agent(
    model=gpt_41,
    tools=[document_retriever],
    prompt=prompt,
    state_schema=MAINAGENTSTATE
)


async def main(state):
    response = await rag_agent.ainvoke(state)
    return response

if __name__ == "__main__":
    async def main():
        await Database.init()
        current_date = datetime.now().strftime("%Y-%m-%d")
        state = {
            "messages": [
                {
                    "role": "user",
                    "content": "explain brownian motion"
                }
            ],
            "retrieved_docs": [],
            "date": current_date
        }
        response = await rag_agent.ainvoke(state)
        await Database.close()
        return response
    print(asyncio.run(main()))