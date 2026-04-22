import os

from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
# from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI

from langchain_tavily import TavilySearch

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    )
tools = [TavilySearch]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-agentic-ai!")
    print(os.getenv("TAVILY_API_KEY"))
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Tokyo?")]}
    )
    print(result)


if __name__ == "__main__":
    main()
