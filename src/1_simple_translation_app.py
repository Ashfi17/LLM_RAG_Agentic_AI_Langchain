from pathlib import Path
from dotenv import load_dotenv
import os

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


env_path = Path('.').parent / '.env'
print(f"Loading environment variables from {env_path}")
if not env_path.exists():
    raise FileNotFoundError(f"Environment file {env_path} not found. Please create a .env file with the required variables.")
load_dotenv(dotenv_path=env_path)

AZURE_OPEN_API_KEY = os.getenv("AZURE_OPEN_API_KEY")
AZURE_OPEN_API_ENDPOINT = os.getenv("AZURE_OPEN_API_ENDPOINT")
AZURE_OPEN_API_VERSION = os.getenv("AZURE_OPEN_API_VERSION")

if not AZURE_OPEN_API_KEY or not AZURE_OPEN_API_ENDPOINT or not AZURE_OPEN_API_VERSION:
    raise ValueError("Please set the AZURE_OPEN_API_KEY, AZURE_OPEN_API_ENDPOINT, and AZURE_OPEN_API_VERSION environment variables.")
else:
    print("Environment variables loaded successfully.")

# Initialize the AzureChatOpenAI model
model = AzureChatOpenAI(
    api_key=AZURE_OPEN_API_KEY,  
    api_version=AZURE_OPEN_API_VERSION,
    azure_endpoint=AZURE_OPEN_API_ENDPOINT
)

# messages = [
#     SystemMessage("Translate the following from English into Italian"),
#     HumanMessage("hi!"),
# ]


system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language":"Italian", "text":"hi!"})
print(prompt.messages[0].content)

# Create a new instance of the model with the prompt
response = model.invoke(prompt)
print(response.content)