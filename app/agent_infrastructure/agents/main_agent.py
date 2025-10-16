from langchain_core.messages import AnyMessage

from langgraph.prebuilt import create_react_agent

from app.agent_infrastructure.prompt_templates import agent_prompt_template
from app.agent_infrastructure.infrastructure.llm_clients import gpt_41
from app.agent_infrastructure.tools.document_retriver import document_retriever
from app.schema.langgraph_agent_states import MAINAGENTSTATE


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
