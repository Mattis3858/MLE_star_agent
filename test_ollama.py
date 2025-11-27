from langchain_ollama import ChatOllama

from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

import os


load_dotenv()

# 填入您的 Key

OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")


try:

    print("Connecting to Ollama Cloud...")

    llm = ChatOllama(
        model="deepseek-v3.1:671b-cloud",  # 或 gpt-oss:120b
        base_url="https://ollama.com",
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
        temperature=0.1,
    )
    response = llm.invoke(
        [HumanMessage(content="Hello, explain what is MLE-STAR in one sentence.")]
    )
    print(f"\nSuccess! Response:\n{response.content}")


except Exception as e:

    print(f"\nConnection Failed. Error:\n{e}")
    print("\nTroubleshooting:")
    print("1. Check if 'langchain-ollama' is installed/updated.")
    print("2. Check if the model name requires specific access.")
    print("3. Verify the API Key.")
