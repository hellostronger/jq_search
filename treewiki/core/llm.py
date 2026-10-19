from __future__ import annotations

import json
import re
from typing import Any


class LLMError(RuntimeError):
    pass


class LLM:
    """OpenAI 兼容客户端：供应商不锁定，base_url/api_key/双模型均可配置。"""

    def __init__(self, cfg: dict[str, Any]):
        from openai import OpenAI

        llm_cfg = cfg["llm"]
        if not llm_cfg.get("base_url"):
            raise LLMError("未配置 llm.base_url")
        if not llm_cfg.get("api_key"):
            raise LLMError("未配置 llm.api_key")
        self.cfg = cfg
        self._client = OpenAI(
            base_url=llm_cfg["base_url"], api_key=llm_cfg["api_key"], timeout=180
        )
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._thinking_disabled = False

    @property
    def index_model(self) -> str:
        return self.cfg["llm"]["index_model"]

    @property
    def chat_model(self) -> str:
        return self.cfg["llm"]["chat_model"]

    def usage(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        model = model or self.chat_model
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self._thinking_disabled:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        resp = self._request(kwargs)
        self._record(resp)
        content = resp.choices[0].message.content or ""
        if content:
            return content

        # 思考型模型可能把 max_tokens 全耗在 reasoning 上，导致 content 为空
        if resp.choices[0].finish_reason == "length" and _reasoning_tokens(resp):
            for disable_thinking, variant in (
                (True, {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}),
                (False, {"max_tokens": min(max(max_tokens * 3, 8192), 32768)}),
            ):
                retry = {**kwargs, **variant}
                try:
                    r2 = self._request(retry)
                except Exception:
                    continue
                self._record(r2)
                text = r2.choices[0].message.content or ""
                if text:
                    if disable_thinking:
                        self._thinking_disabled = True
                    return text
            raise LLMError(
                f"LLM 输出为空：思考耗尽 max_tokens={max_tokens}；"
                "自动关闭思考与扩大预算后仍无内容"
            )
        return content

    def _request(self, kwargs: dict[str, Any]):
        try:
            return self._client.chat.completions.create(**kwargs)
        except Exception:
            if "response_format" not in kwargs:
                raise
            # 部分兼容端点不支持 response_format，降级重试
            retry = {k: v for k, v in kwargs.items() if k != "response_format"}
            return self._client.chat.completions.create(**retry)

    def _record(self, resp) -> None:
        self.calls += 1
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    def chat_json(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> dict:
        text = self.chat(
            messages, model=model, json_mode=True,
            temperature=temperature, max_tokens=max_tokens,
        )
        data = extract_json(text)
        if data is None:
            text = self.chat(
                messages
                + [{"role": "user", "content": "上一次输出不是合法 JSON。请重新输出，只输出 JSON 本身。"}],
                model=model, json_mode=True, temperature=temperature, max_tokens=max_tokens,
            )
            data = extract_json(text)
        if not isinstance(data, dict):
            raise LLMError(f"LLM 未返回 JSON 对象: {text[:200]}")
        return data


def _reasoning_tokens(resp) -> int:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0
    n = getattr(usage, "reasoning_tokens", None)
    if n:
        return int(n)
    details = getattr(usage, "completion_tokens_details", None)
    return int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0


def extract_json(text: str | None) -> Any:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(text[s : e + 1])
        except (json.JSONDecodeError, ValueError):
            return None
    return None
