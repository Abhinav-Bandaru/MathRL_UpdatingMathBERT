# icl_model_wrapper.py

from openai import OpenAI
import numpy as np

class OpenAIICLModel:
    def __init__(self, api_key, model_name="gpt-4o", temperature=0.0, max_tokens=1024):
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

    def generate(self, prompt, logprobs=False):
        """
        Args:
            prompt (str): The full ICL prompt.

        Returns:
            predicted_answer (str): Model's generated output.
            mean_logprob (float): Mean log probability of generated tokens (confidence proxy).
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a Math Expert. Return just the final answer as a number, NOTHING ELSE"},
                    {"role": "user", "content": prompt}
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

