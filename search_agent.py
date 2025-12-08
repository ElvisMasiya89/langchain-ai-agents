from dotenv import load_dotenv
from langchain.agents import create_agent

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

load_dotenv()


class Source(BaseModel):
    """Scource object for search results."""

    url: str = Field(description="The URL of the source.")


class AgentResponse(BaseModel):
    """Schema for the agent response,with answer and sources."""

    answer: str = Field(description="The final answer from the agent.")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer."
    )


llm = ChatOllama(model="gpt-oss:20b", temperature=0.7)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def start():
    print("Hello from LangChain Ollama Search Agent!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="Search LinkedIn for Python Developer jobs in Cape Town. Show me recent postings, full-time roles, and remote/hybrid options available in Cape Town."
            )
        }
    )
    print(result.answer)


start()
