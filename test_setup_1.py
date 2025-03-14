import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Load environment variables
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")  # Get the API key

# Instantiate ChatNVIDIA
client = ChatNVIDIA(
  model="deepseek-ai/deepseek-r1",
  api_key=api_key,
  temperature=0.6,
  top_p=0.7,
  max_tokens=4096,
)

for chunk in client.stream([{"role":"user","content":""}]): 
  print(chunk.content, end="")
