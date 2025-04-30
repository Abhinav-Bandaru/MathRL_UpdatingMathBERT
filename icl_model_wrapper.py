# icl_model_wrapper.py

from openai import OpenAI
import numpy as np
from pydantic import BaseModel, Field
import time 

class LatexSolution(BaseModel):
    reasoning: str = Field(..., description="Step-by-step reasoning process to solve the math problem.")
    answer: str = Field(..., description="The final answer in LaTeX format.")

class MathSolution(BaseModel):
    reasoning: str = Field(..., description="Step-by-step reasoning process to solve the math problem.")
    answer: float = Field(..., description="The final numerical answer.")

class MathSolutionNumberOnly(BaseModel):
    answer: float = Field(..., description="The final numerical answer.")

class OpenAIICLModel:
    def __init__(self, api_key, model_name="gpt-4.1-nano", temperature=0.0, max_tokens=1024):
        """
        Args:
            api_key (str): Your OpenAI API key
            model_name (str): Model to use (e.g., 'gpt-4o', 'gpt-4-turbo', etc.)
            temperature (float): Sampling temperature
            max_tokens (int): Max tokens to generate
        """
        # openai.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key)

    def generate(self, inference_question, example_questions, logprobs=False):
        """
        Args:
            prompt (str): The full ICL prompt.

        Returns:
            predicted_answer (str): Model's generated output.
            mean_logprob (float): Mean log probability of generated tokens (confidence proxy).
        """
        role = "You are an expert mathematician capable of solving diverse, challenging math problems"
        detailed_instructions = "Solve the following math problem, thinking step-by-step. Please make sure to format your reasoning and the final answer in JSON. Keep your reasoning concise and to the point"
        prompt = {
            "system": f"{role}.\n# Task Description\n{detailed_instructions}\n\nHere are some examples to help you understand the task.\n# Example Solutions\n{example_questions}",
            "user": f"# Solve this problem:\n\"{inference_question}\""
        }
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt['system']},
                    {"role": "user", "content": prompt['user']}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                logprobs=logprobs
            )

            output_text = response.choices[0].message.content.strip()

            # Extract logprobs correctly (new SDK style)
            if response.choices[0].logprobs is not None:
                token_logprobs = response.choices[0].logprobs.token_logprobs  # list of floats
                if token_logprobs is not None:
                    mean_logprob = float(np.mean(token_logprobs))
                else:
                    mean_logprob = None
            else:
                mean_logprob = None  # fallback if no logprobs

            return output_text, mean_logprob

        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            return "Error", None

class OpenAIAdvanced:
    """
    A wrapper that mimics OpenAIICLModel’s public API but relies on the user-supplied
    `query_llm` helper (which in turn delegates to `query_gpt`).

    ───────────────────────────────────────────────────────────────────────────────
    Parameters
    ----------
    api_key : str
        Ignored by default because `query_llm`/`query_gpt` already knows how to
        find credentials (or uses a global OpenAI client). It is accepted here so
        that the signature matches OpenAIICLModel.
    model_name : str, default "gpt-4o-mini"
        Passed straight through to `query_llm(..., model=model_name, …)`.
    temperature : float, default 0.0
        Sampling temperature.
    max_tokens : int, default 1024
        Maximum tokens to generate.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 3600,
    ):
        # keeping the same public attributes as OpenAIICLModel for compatibility
        self.model_name   = model_name
        self.temperature  = temperature
        self.max_tokens   = max_tokens
        # api_key accepted only to mirror OpenAIICLModel; ignore / log if needed
        self.api_key      = api_key
        self.client = OpenAI(api_key=api_key)

    # ────────────────────────────────────────────────────────────────────────────
    def generate(self, inference_question, example_questions, logprobs: bool = False, debug=False):
        """
        Run the ICL prompt through query_llm / query_gpt.

        Returns
        -------
        predicted_answer : str
            The model’s textual output (stripped).
        mean_logprob : float | None
            Mean log-probability over generated tokens if logprobs=True,
            otherwise None.
        """
        # We want the same “math-only” answer style as the first wrapper.
        role = "You are an expert mathematician capable of solving diverse, challenging math problems"
        detailed_instructions = "Solve the following math problem, thinking step-by-step. Please make sure to format your reasoning and the final answer in JSON. Keep your reasoning concise and to the point"
        prompt = {
            "system": f"{role}.\n# Task Description\n{detailed_instructions}\n\nHere are some examples to help you understand the task.\n# Example Solutions\n{example_questions}",
            "user": f"# Solve this problem:\n\"{inference_question}\""
        }

        response, logs = self.query_llm(
            prompt,
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            logprobs=logprobs,
            system_prompt_included=True,
            return_json=True,
            debug=debug,            # set True to see prompt / response
            json_schema=LatexSolution
        )

        # `query_llm` returns (text, logprobs) when logprobs=True;
        # otherwise it returns just text.
        if logprobs:
            token_logprobs = logs.token_logprobs if hasattr(logs, "token_logprobs") else logs
            mean_lp = float(np.mean(token_logprobs)) if token_logprobs else None
        else:
            mean_lp = None

        return response, mean_lp
    
    def query_llm(self, prompt, max_tokens=1000, temperature=0, top_p=0, max_try_num=10, model="gpt-4.1-nano", debug=False, return_json=False, json_schema=None, logprobs=False, system_prompt_included=True):
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
                    response = self.query_gpt(prompt, model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, return_json=return_json, json_schema=json_schema, logprobs=logprobs, system_prompt_included=system_prompt_included, debug=debug)
                    if logprobs:
                        return response.choices[0].message.content.strip(), response.choices[0].logprobs
                    return response, None
            except Exception as e:
                if 'gpt' in model:
                    print(f"Error making OpenAI API call: {e}")
                else: 
                    print(f"Error making API call: {e}")
                curr_try_num += 1
                print(curr_try_num >= 3 and (return_json or json_schema))
                if curr_try_num >= 3 and (return_json or json_schema is not None):
                    response = self.query_llm(prompt, model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, return_json=False, json_schema=None, logprobs=logprobs, system_prompt_included=system_prompt_included, debug=debug)
                    prompt=f"""Turn the following text into a JSON object: {response}"""
                    json_response = self.query_llm(prompt, model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, return_json=True, json_schema=json_schema, logprobs=logprobs, system_prompt_included=False, debug=debug)
                    print("Turning text into JSON by brute force...")
                    return json_response
        return None

    def query_gpt(self, prompt: str | dict, model: str = 'gpt-4o-mini', max_tokens: int = 4000, temperature: float = 0, top_p: float = 0, logprobs: bool = False, return_json: bool = False, json_schema = None, system_prompt_included: bool = False, is_hippa: bool = False, debug: bool = False):
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

        if return_json or (json_schema is not None):
            if json_schema is None:
                api_params["response_format"] = {"type": "json_object"}
                completion = self.client.chat.completions.create(**api_params)
                response = completion.choices[0].message.content.strip()
            else:
                api_params["response_format"] = json_schema
                completion = self.client.beta.chat.completions.parse(**api_params)
                response = completion.choices[0].message.parsed
        else: 
            completion = self.client.chat.completions.create(**api_params)
            response = completion.choices[0].message.content.strip()
        if debug:
            print(f"Response: {response}")
        if logprobs:
            return response, completion.choices[0].logprobs
        else:
            return response