import os
import time
import requests
import traceback

def query_llm(messages, model, temperature=1.0, max_new_tokens=1024, stop=None, return_usage=False):
    """
    A generic function to query LLMs using standard OpenAI-like HTTP endpoints.
    Reads configuration from standard environment variables:
    - OPENAI_API_KEY: Default to "token-abc123" to match the default local vLLM server.
    - OPENAI_BASE_URL: e.g., "http://localhost:8000/v1" or "https://api.openai.com/v1"
    """
    api_key = os.environ.get("OPENAI_API_KEY", "token-abc123")
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    
    # Ensure base_url does not end with /chat/completions; we append it explicitly.
    if base_url.endswith("/chat/completions"):
        api_url = base_url
    else:
        api_url = f"{base_url.rstrip('/')}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }

    if stop is not None:
        payload["stop"] = stop

    tries = 0
    while tries < 5:
        tries += 1
        try:
            resp = requests.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=600
            )

            if resp.status_code != 200:
                raise Exception(resp.text)
            
            resp_json = resp.json()
            break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if "maximum context length" in str(e).lower():
                raise e
            elif "triggering" in str(e).lower() or "content management policy" in str(e).lower():
                return 'Trigger OpenAI\'s content management policy.'
            
            print(f"Error Occurs: \"{str(e)}\"        Retry ({tries}/5) ...")
            time.sleep(1)
    else:
        print("Max tries reached. Failed to get response.")
        return None

    try:
        message_data = resp_json["choices"][0]["message"]
        if "content" not in message_data and "content_filter_results" in resp_json["choices"][0]:
            content = "Trigger OpenAI's content management policy."
        else:
            content = message_data.get("content") or message_data.get("reasoning_content", "")

        if return_usage:
            return content, resp_json.get("usage", {})
        else:
            return content
    except Exception as e:
        print(f"Unexpected response format: {e}")
        traceback.print_exc()
        return None

if __name__ == '__main__':
    # Local quick test
    model = 'vllm-model' # Change based on your running vLLM model
    prompt = 'Who are you?'
    msg = [{'role': 'user', 'content': prompt}]
    res = query_llm(msg, model=model, temperature=0, max_new_tokens=10, stop=None, return_usage=True)
    print(res)
