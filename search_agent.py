from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()


@tool
def search(query: str) -> dict[str, Any]:
    """Tool that searches the web for a given query.
    Args:
        query (str): The search query.
    Returns:
        dict[str, Any]: The search results.
    """
    return tavily.search(query=query, include_answer=True)


llm = ChatOllama(model="gpt-oss:20b", temperature=0.7)
tools = [search]
agent = create_agent(model=llm, tools=tools)


def start():
    print("Hello from LangChain Ollama Search Agent!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="Search LinkedIn for Python Developer jobs in Cape Town. Show me recent postings, full-time roles, and remote/hybrid options available in Cape Town."
            )
        }
    )
    print(result)


start()
