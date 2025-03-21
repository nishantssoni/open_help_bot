import requests
import json

class LocalOllamaClient:
    """Simulate OpenAI's API while using a local Ollama instance."""
    
    def __init__(self, base_url="http://localhost:11434/api/generate"):
        self.base_url = base_url

    def chat_completions_create(self, model, messages, stream=False):
        """Simulate OpenAI's ChatCompletion API with Ollama."""
        prompt = messages[-1]["content"]  # Extract the last user message
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        if stream:
            return self._stream_response(payload)
        else:
            return self._sync_response(payload)

    def _sync_response(self, payload):
        """Handle non-streaming responses."""
        response = requests.post(self.base_url, json=payload)

        if response.status_code == 200:
            data = response.json()
            return {"choices": [{"message": {"role": "assistant", "content": data["response"]}}]}
        else:
            return {"error": f"Error: {response.status_code}, {response.text}"}

    def _stream_response(self, payload):
        """Handle streaming responses like OpenAI's API."""
        with requests.post(self.base_url, json=payload, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield {"choices": [{"delta": {"content": data["response"]}}]}
                    if data.get("done", False):
                        break

# Example usage
if __name__ == "__main__":
    ollama_client = LocalOllamaClient()
    prompt = "Explain quantum computing in one line"

    # Non-streaming response
    response = ollama_client.chat_completions_create(
        model="llama3.1:latest",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    print("Non-Streaming Response:", response["choices"][0]["message"]["content"])

    # Streaming response
    print("\nStreaming Response:")
    for chunk in ollama_client.chat_completions_create(
        model="llama3.1:latest",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    ):
        print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
