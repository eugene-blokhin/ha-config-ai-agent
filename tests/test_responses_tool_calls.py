import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ha-config-ai-agent"))
from src.agents.agent_system import AgentSystem


class FakeStream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def __aiter__(self):
        self.iterator = iter(self.events)
        return self

    async def __anext__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopAsyncIteration

    async def close(self):
        self.closed = True


class FakeResponses:
    def __init__(self, stream):
        self.stream = stream

    async def create(self, **_params):
        return self.stream


class ResponsesToolCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_multiple_tool_calls_are_emitted_together_after_completion(self):
        calls = [
            SimpleNamespace(
                type="function_call",
                call_id=f"call_{index}",
                name=f"tool_{index}",
                arguments="{}",
                model_dump=lambda exclude_none=True, index=index: {
                    "type": "function_call",
                    "call_id": f"call_{index}",
                    "name": f"tool_{index}",
                    "arguments": "{}",
                },
            )
            for index in (1, 2)
        ]
        stream = FakeStream([
            SimpleNamespace(type="response.output_item.done", item=calls[0]),
            SimpleNamespace(type="response.output_item.done", item=calls[1]),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(output=calls, usage=None),
            ),
        ])
        agent = SimpleNamespace(
            client=SimpleNamespace(responses=FakeResponses(stream)),
            model="gpt-6-luna",
            temperature=None,
            usage_tracking="disabled",
        )

        chunks = [
            chunk async for chunk in AgentSystem._stream_responses(agent, [], [])
        ]

        tool_call_chunks = [
            chunk for chunk in chunks if chunk.choices[0].delta.tool_calls
        ]
        self.assertEqual(len(tool_call_chunks), 1)
        self.assertEqual(
            [call.id for call in tool_call_chunks[0].choices[0].delta.tool_calls],
            ["call_1", "call_2"],
        )
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
