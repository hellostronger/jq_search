from __future__ import annotations

from types import SimpleNamespace

from treewiki.core.llm import LLM


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def make_llm(responses) -> LLM:
    llm = LLM.__new__(LLM)
    llm.cfg = {"llm": {"index_model": "m", "chat_model": "m"}}
    llm._client = FakeClient(responses)
    llm.calls = 0
    llm.prompt_tokens = 0
    llm.completion_tokens = 0
    llm._thinking_disabled = False
    return llm


def make_resp(content: str, finish: str = "stop", reasoning: int = 0):
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=20,
        reasoning_tokens=reasoning,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
        usage=usage,
    )


def test_retries_with_thinking_disabled_when_reasoning_exhausts_budget():
    llm = make_llm([make_resp("", "length", 2500), make_resp('{"ok":true}')])

    out = llm.chat([{"role": "user", "content": "hi"}], model="m", json_mode=True, max_tokens=2500)

    assert out == '{"ok":true}'
    assert llm.calls == 2
    calls = llm._client.chat.completions.calls
    assert calls[1]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_falls_back_to_larger_budget_when_extra_body_rejected():
    llm = make_llm(
        [make_resp("", "length", 2500), ValueError("unknown field"), make_resp('{"ok":true}')]
    )

    out = llm.chat([{"role": "user", "content": "hi"}], model="m", max_tokens=2500)

    assert out == '{"ok":true}'
    calls = llm._client.chat.completions.calls
    assert "extra_body" in calls[1]
    assert calls[2]["max_tokens"] == 8192


def test_remembers_thinking_disabled_for_later_calls():
    llm = make_llm(
        [make_resp("", "length", 2500), make_resp('{"a":1}'), make_resp('{"b":2}')]
    )

    llm.chat([{"role": "user", "content": "one"}], model="m", max_tokens=2500)
    llm.chat([{"role": "user", "content": "two"}], model="m", max_tokens=800)

    calls = llm._client.chat.completions.calls
    assert calls[2]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_empty_content_without_reasoning_is_returned_as_is():
    llm = make_llm([make_resp("", "stop", 0)])

    assert llm.chat([{"role": "user", "content": "hi"}], model="m") == ""
    assert len(llm._client.chat.completions.calls) == 1
