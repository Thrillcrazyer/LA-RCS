import base64
import anthropic
from typing import Any, Optional



class ClaudeService:
    def __init__(self,configs):
        self.data=configs["CLAUDE"]
        self.client = anthropic.Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key=self.data["API_KEY"]
                ) 
        self.configs=configs
        
    def chat_completion(
        self,
        messages,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ):
        model = self.data["API_MODEL"]
        temperature = self.configs["TEMPERATURE"]
        max_tokens = self.configs["MAX_TOKENS"]
        top_p = self.configs["TIMEOUT"]

        try:
            response: Any = self.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
                **kwargs
            )

            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            #OPUS 기준으로 수정
            cost = prompt_tokens / 1000 * 0.015 + completion_tokens / 1000 * 0.075

            return response.message.content[0].text, cost

        except anthropic.APITimeoutError as e:
            # Handle timeout error, e.g. retry or log
            raise Exception(f"OpenAI API request timed out: {e}")
        except anthropic.APIConnectionError as e:
            # Handle connection error, e.g. check network or log
            raise Exception(f"OpenAI API request failed to connect: {e}")
        except anthropic.BadRequestError as e:
            # Handle invalid request error, e.g. validate parameters or log
            raise Exception(f"OpenAI API request was invalid: {e}")
        except anthropic.AuthenticationError as e:
            # Handle authentication error, e.g. check credentials or log
            raise Exception(f"OpenAI API request was not authorized: {e}")
        except anthropic.PermissionDeniedError as e:
            # Handle permission error, e.g. check scope or log
            raise Exception(f"OpenAI API request was not permitted: {e}")
        except anthropic.RateLimitError as e:
            # Handle rate limit error, e.g. wait or log
            raise Exception(f"OpenAI API request exceeded rate limit: {e}")
        except anthropic.APIError as e:
            # Handle API error, e.g. retry or log
            raise Exception(f"OpenAI API returned an API Error: {e}")
