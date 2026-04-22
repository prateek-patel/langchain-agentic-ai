import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent, AgentExecutor
from langchain.tools import tool

load_dotenv()

# 1. Define the tool with a clear description
@tool
def get_weather(location: str):
    """Returns the current weather for a given location."""
    return f"The weather in {location} is 25°C and clear."

# 2. Initialize ChatGroq (This is NOT deprecated)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# 3. Create the agent using the 'suggested' import
tools = [get_weather]
agent = create_agent(llm, tools)

# 4. Create the Executor to handle the loop
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def main():
    # Note: create_agent usually expects a 'messages' key
    user_input = {"messages": [("user", "What is the weather in Paris?")]}
    
    response = agent_executor.invoke(user_input)
    
    # Print the last message from the assistant
    print("\nFINAL ANSWER:", response["messages"][-1].content)

if __name__ == "__main__":
    main()