import json

import pytest
from pydantic import BaseModel

from ielts_tutor.config import Settings
from ielts_tutor.kg.extract import tag_item
from ielts_tutor.llm import FakeLLM, LLMError, get_llm
from ielts_tutor.llm.base import parse_json_output, strict_json_schema


class Pair(BaseModel):
    a: int
    b: list[str]


def test_strict_schema_closes_objects():
    schema = strict_json_schema(Pair)
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["a", "b"]


def test_parse_json_output_handles_fences_and_errors():
    assert parse_json_output('```json\n{"a": 1, "b": []}\n```', Pair).a == 1
    with pytest.raises(LLMError):
        parse_json_output("not json", Pair)


def test_get_llm_fake_provider():
    assert get_llm(Settings(llm_provider="fake")).name == "fake"


def _responder(payload):
    return lambda messages, system: json.dumps(payload)


def test_tag_item_accepts_valid_tags(kg):
    llm = FakeLLM(_responder({
        "topic_ids": ["topic.technology.internet"],
        "question_type": "qt.writing.t1.line_graph",
        "grammar_ids": ["gp.data_language"],
        "new_topics": [],
        "confidence": 0.9,
    }))
    item = tag_item(llm, kg, "w1", "The line graph shows internet access ...", "writing")
    assert item.reasons == []
    assert item.tags.question_type == "qt.writing.t1.line_graph"
    prompt = llm.calls[0][0][0].content
    assert "topic.technology.internet" in prompt
    assert "topic.personal.hometown" not in prompt  # speaking-only topic hidden for writing


def test_tag_item_drops_unknown_ids_and_flags_for_review(kg):
    llm = FakeLLM(_responder({
        "topic_ids": ["topic.made_up", "topic.environment.waste"],
        "question_type": "qt.nope",
        "grammar_ids": ["gp.passive"],
        "new_topics": [{"name": "Fashion", "parent_id": "topic.culture", "keywords": ["clothes"]}],
        "confidence": 0.5,
    }))
    item = tag_item(llm, kg, "x", "text", "writing")
    assert item.tags.topic_ids == ["topic.environment.waste"]
    assert item.tags.question_type is None
    assert set(item.dropped_ids) == {"topic.made_up", "qt.nope"}
    assert any("low confidence" in r for r in item.reasons)
    assert "proposes new topics" in item.reasons
