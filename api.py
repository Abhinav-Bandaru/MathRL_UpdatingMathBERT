import time
import re
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    
def query_llm(prompt, max_tokens=1000, temperature=0, top_p=0, max_try_num=10, model="gpt-4o-mini", debug=False, return_json=False, json_schema=None, logprobs=False, system_prompt_included=True):
    if debug:
        if system_prompt_included:
            print(f"System prompt: {prompt['system']}")
            print(f"User prompt: {prompt['user']}")
        else:
            print(prompt)
        print(f"Model: {model}")

    curr_try_num = 0
    while curr_try_num < max_try_num:
        try:
            if 'gpt' in model or 'o3' in model:
                response = query_gpt(prompt, model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, return_json=return_json, json_schema=json_schema, logprobs=logprobs, system_prompt_included=system_prompt_included, debug=debug)
                if logprobs:
                    return response.choices[0].message.content.strip(), response.choices[0].logprobs
                return response, None
        except Exception as e:
            if 'gpt' in model:
                print(f"Error making OpenAI API call: {e}")
            else: 
                print(f"Error making API call: {e}")
            curr_try_num += 1
            print("HELLO")
            print(curr_try_num)
            print(curr_try_num >= 3 and (return_json or json_schema))
            if curr_try_num >= 3 and return_json:
                response = query_llm(prompt, model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, return_json=False, json_schema=json_schema, logprobs=logprobs, system_prompt_included=system_prompt_included, debug=debug)
                prompt=f"""Turn the following text into a JSON object: {response}"""
                json_response = query_llm(prompt, model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, return_json=True, json_schema=json_schema, logprobs=logprobs, system_prompt_included=False, debug=debug)
                print("Turning text into JSON by brute force...")
                return json_response, None
    return None, None

def query_gpt(prompt: str | dict, model: str = 'gpt-4o-mini', max_tokens: int = 4000, temperature: float = 0, top_p: float = 0, logprobs: bool = False, return_json: bool = False, json_schema = None, system_prompt_included: bool = False, is_hippa: bool = False, debug: bool = False):
    """OpenAI API wrapper; For HIPPA compliance, use client_safe e.g. model='openai-gpt-4o-high-quota-chat'"""
    if system_prompt_included:
        # Format chat prompt with system and user messages
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    if 'o3' in model:
        api_params = {
            "model": model,
            "reasoning_effort": "high",
            "messages": messages,
        }
    else:
        api_params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": 0
        }
    if logprobs:
        api_params["logprobs"] = logprobs
        api_params["top_logprobs"] = 3

    if return_json or json_schema:
        if json_schema is None:
            api_params["response_format"] = {"type": "json_object"}
            completion = client.chat.completions.create(**api_params)
            response = completion.choices[0].message.content.strip()
        else:
            api_params["response_format"] = json_schema
            completion = client.beta.chat.completions.parse(**api_params)
            response = completion.choices[0].message.parsed
    else: 
        completion = client.chat.completions.create(**api_params)
        response = completion.choices[0].message.content.strip()
    if debug:
        print(f"Response: {response}")
    if logprobs:
        return response, completion.choices[0].logprobs
    else:
        return response