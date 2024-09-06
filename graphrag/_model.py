import numpy as np
import os
from openai import AsyncOpenAI, APIConnectionError, RateLimitError, Timeout
from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage


async def openai_complete_if_cache(
        model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
    return response.choices[0].message.content


async def gpt_4o_complete(
        prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
        prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def deepseek_model_if_cache(
        model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Please set DEEPSEEK_API_KEY in environment variables")
    openai_async_client = AsyncOpenAI(
        api_key=api_key, base_url="https://api.deepseek.com/v1"
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    # Cache the response if having
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
    return response.choices[0].message.content


async def deepseek_chat_complete(
        prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await deepseek_model_if_cache(
        "deepseek-chat",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def ollama_model_if_cache(
        model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    import ollama
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_client = ollama.AsyncClient(host=ollama_host)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    response = await ollama_client.chat(model=model, messages=messages, **kwargs)

    result = response["message"]["content"]
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": model}})
    return result


async def ollama_model_complete(
        prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    ollama_model_name = os.getenv("OLLAMA_MODEL", "qwen2:72b")
    return await ollama_model_if_cache(
        ollama_model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    openai_async_client = AsyncOpenAI()
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])
