"""
OpenAI-compatible LLM client wrapper.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import openai


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    base_url: str
    model: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 4096


class LLMClient:
    """OpenAI-compatible LLM client."""

    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM client.

        Args:
            config: LLM configuration
        """
        self.config = config
        self.client = openai.OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            The generated response text
        """
        messages: List[Dict[str, str]] = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )

        return response.choices[0].message.content or ""

    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response with conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            The generated response text
        """
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )

        return response.choices[0].message.content or ""

    @classmethod
    def from_params(
        cls,
        base_url: str,
        model: str,
        api_key: str,
        **kwargs: Any,
    ) -> "LLMClient":
        """
        Create an LLM client from parameters.

        Args:
            base_url: API base URL
            model: Model name
            api_key: API key
            **kwargs: Additional config options

        Returns:
            Configured LLM client
        """
        config = LLMConfig(
            base_url=base_url,
            model=model,
            api_key=api_key,
            **kwargs,
        )
        return cls(config)
