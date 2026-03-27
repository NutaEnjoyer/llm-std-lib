"""
Basic usage example for llm_std_lib.

Demonstrates how to configure the client, attach middleware and send a
simple chat completion request through the unified provider interface.
"""

from __future__ import annotations

# from llm_std_lib import LLMClient, LLMConfig
# from llm_std_lib.middleware.builtins.logger import LoggerMiddleware
# from llm_std_lib.middleware.builtins.cost import CostTrackerMiddleware
#
# config = LLMConfig(
#     providers={"openai": {"api_key": "sk-..."}},
# )
#
# client = LLMClient(
#     config=config,
#     middleware=[LoggerMiddleware(), CostTrackerMiddleware()],
# )
#
# response = client.complete(
#     model="openai/gpt-4o-mini",
#     messages=[{"role": "user", "content": "Hello, world!"}],
# )
#
# print(response.content)
