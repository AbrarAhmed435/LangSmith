
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.messages import HumanMessage, ToolMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
import os

load_dotenv()

os.environ['LANGCHAIN_PROJECT']='Weather'

API_KEY = os.getenv("WEATHER_API_KEY")  # keep key in .env
BASE_URL = "http://api.weatherapi.com/v1"


@tool("search_on_web",description="Search you query on web")
def search_on_web(query:str):
    search_tool=DuckDuckGoSearchRun()
    result=search_tool.invoke(query)
    return result


@tool("get_current_weather",description="Know current weather of give location")
def get_current_weather(city):
    url = f"{BASE_URL}/current.json"
    params = {
        "key": API_KEY,
        "q": city
    }
    response = requests.get(url, params=params)
    return response.json()





from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
llm=HuggingFaceEndpoint(
    # repo_id="HuggingFaceH4/zephyr-7b-beta",
    # repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="HuggingFaceH4/zephyr-7b-gemma-v0.1",
    # repo_id="openai/gpt-oss-20b",
    repo_id="openai/gpt-oss-120b",
    # repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)




agent=create_agent(
    model=model,
    tools=[get_current_weather,search_on_web],
    system_prompt="You are helpfull Assistant"
)

user_query=input("Ask Anything! ")


result=agent.invoke(
    {
        "messages":[{"role":"user","content":user_query}]
    }
)

from pprint import pprint
# print(result['messages'])
print()
print(result['messages'][-1].content)


# llm_with_tools=model.bind_tools([get_current_weather])
# user_input=input("enter locaton")
# user_input=f"What is the current weather at {user_input} Give response in paragraph"
# messages=[HumanMessage(content=user_input)]
# result1=llm_with_tools.invoke(messages)
# print(result1.tool_calls)
# messages.append(result1)
# for tool_call in result1.tool_calls:
#     if tool_call['name']=='get_current_weather':
#         response=get_current_weather.invoke(tool_call['args'])
#         messages.append(
#             ToolMessage(
#                 content=str(response),
#                 tool_call_id=tool_call['id']
#             )
#         )
# result2=llm_with_tools.invoke(messages)
# messages.append(result2)
# print(result2.content)