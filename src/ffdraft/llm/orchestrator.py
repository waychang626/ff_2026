"""Run the draft through Claude, with the tool surface in `tools.py`.

The model is doing the four jobs from brief section 2 and nothing else, so it
is run at low effort: parsing "bijan gone" into a player ID does not need deep
reasoning, and under a 60-second pick clock latency is the binding constraint.
The thinking happens in `simulate.py`, where it is reproducible.

Requires `pip install anthropic` (the `llm` extra). Everything else in this
package works without it - `ffdraft draft` is a complete draft-day tool with no
API dependency, which is deliberate: the network is not a thing to rely on
fifteen seconds before a pick.
"""

from __future__ import annotations

from typing import Any

from .prompt import SYSTEM_PROMPT
from .session import DraftSession
from .tools import TOOL_FUNCTIONS, bind

MODEL = "claude-opus-5"


def _require_sdk():
    try:
        import anthropic
        from anthropic import beta_tool
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SystemExit(
            "the LLM layer needs the Anthropic SDK: pip install 'ffdraft[llm]'\n"
            "(the `ffdraft draft` console works without it)"
        ) from exc
    return anthropic, beta_tool


class Orchestrator:
    """A conversation bound to one draft session."""

    def __init__(self, session: DraftSession, model: str = MODEL) -> None:
        anthropic, beta_tool = _require_sdk()
        bind(session)
        self.session = session
        self.model = model
        self.client = anthropic.Anthropic()
        self.tools = [beta_tool(fn) for fn in TOOL_FUNCTIONS]
        self.messages: list[dict[str, Any]] = []

    def send(self, user_message: str) -> str:
        """One turn. Returns the model's final text."""
        self.messages.append({"role": "user", "content": user_message})

        runner = self.client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=self.tools,
            messages=self.messages,
            # Low effort on purpose: the model parses names and reads out a
            # result. The decision was made before this call.
            output_config={"effort": "low"},
        )

        final = None
        for message in runner:
            final = message
            self.messages.append({"role": "assistant", "content": message.content})
            tool_response = runner.generate_tool_call_response()
            if tool_response is not None:
                self.messages.append(tool_response)

        if final is None:
            return ""
        if getattr(final, "stop_reason", None) == "refusal":
            details = getattr(final, "stop_details", None)
            return f"[model declined: {getattr(details, 'category', 'unknown')}]"
        return "".join(b.text for b in final.content if b.type == "text").strip()


def run_console(session: DraftSession, model: str = MODEL) -> int:
    """Interactive loop. `ffdraft draft` is the offline equivalent."""
    orchestrator = Orchestrator(session, model=model)
    print(f"{session.config.name} - seat {session.config.my_seat}, via {model}")
    print("type picks as they happen; 'quit' to stop\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break
        print(orchestrator.send(line))
        print()
    session.log.save(session.log_path)
    print(f"saved {len(session.log.picks)} picks to {session.log_path}")
    return 0
