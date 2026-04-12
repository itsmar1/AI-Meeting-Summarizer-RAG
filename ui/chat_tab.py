"""
Chat tab — ask questions about one or all past meeting sessions
using the RAG pipeline.
"""

import asyncio

import gradio as gr

from core.exceptions import MeetingSummarizerError, RAGError
from core.logger import get_logger
from rag.rag_chain import answer_across_sessions, answer_question
from storage.session_repo import list_sessions

logger = get_logger(__name__)


def _get_session_choices() -> list[tuple[str, str]]:
    """
    Return a list of (display_label, session_id) tuples for the dropdown.
    Includes a special "All sessions" option as the first entry.
    """
    try:
        sessions = asyncio.get_event_loop().run_until_complete(list_sessions(limit=30))
        choices = [("All sessions (cross-meeting search)", "__all__")]
        for s in sessions:
            label = (
                f"{s.created_at[:16].replace('T', ' ')}  "
                f"[{s.language_name}]  {s.audio_filename}"
            )
            choices.append((label, s.id))
        return choices
    except Exception as exc:
        logger.warning("Could not load session list for chat tab: %s", exc)
        return [("All sessions (cross-meeting search)", "__all__")]


def build_chat_tab() -> None:
    """
    Render the Chat tab inside the currently active gr.Blocks context.
    Must be called from within a `with gr.Blocks():` block.
    """
    gr.Markdown("## Ask questions about your meetings")
    gr.Markdown(
        "Select a specific meeting or search across all of them. "
        "Answers are grounded in the actual transcript — the model will "
        "not guess if the information isn't there."
    )

    with gr.Row():
        session_dropdown = gr.Dropdown(
            label="Meeting session",
            choices=_get_session_choices(),
            value="__all__",
            interactive=True,
            scale=4,
        )
        refresh_sessions_btn = gr.Button(
            "Refresh", variant="secondary", scale=1
        )

    question_input = gr.Textbox(
        label="Your question",
        placeholder=(
            "e.g. What was decided about the project deadline?  "
            "Who is responsible for the budget review?"
        ),
        lines=2,
    )

    with gr.Row():
        ask_btn   = gr.Button("Ask",   variant="primary",    scale=3)
        clear_btn = gr.Button("Clear", variant="secondary",  scale=1)

    answer_output = gr.Textbox(
        label="Answer",
        interactive=False,
        lines=8,
        max_lines=20,
    )

    sources_display = gr.Textbox(
        label="Source excerpts used",
        interactive=False,
        lines=5,
        max_lines=10,
        visible=True,
    )

    error_display = gr.Textbox(
        label="Error", interactive=False, visible=False, lines=2
    )

    # ── Event handlers ────────────────────────────────────────────────────────

    def on_refresh_sessions():
        return gr.update(choices=_get_session_choices())

    def on_clear():
        return "", "", "", gr.update(visible=False)

    def on_ask(session_value: str, question: str):
        """
        Generator function — yields partial answer strings for streaming.
        Also performs retrieval to populate the sources panel.
        """
        question = question.strip()
        if not question:
            yield (
                gr.update(visible=False),
                "",
                "Please enter a question.",
                gr.update(visible=False),
            )
            return

        # Retrieve source chunks first so we can show them
        try:
            from rag.retriever import retrieve, retrieve_across_sessions
            from storage.session_repo import list_sessions as _list

            if session_value == "__all__":
                sessions = asyncio.get_event_loop().run_until_complete(_list(limit=30))
                sids = [s.id for s in sessions]
                chunks = asyncio.get_event_loop().run_until_complete(
                    retrieve_across_sessions(sids, question)
                )
            else:
                chunks = asyncio.get_event_loop().run_until_complete(
                    retrieve(session_value, question)
                )

            sources_text = "\n\n---\n\n".join(
                f"[Excerpt {i+1}  |  similarity: {1 - c['distance']:.0%}]\n{c['text']}"
                for i, c in enumerate(chunks)
            ) or "No relevant excerpts found."

        except RAGError as exc:
            yield (
                gr.update(visible=False),
                "",
                "",
                gr.update(
                    value=str(exc),
                    visible=True,
                ),
            )
            return
        except Exception as exc:
            yield (
                gr.update(visible=False),
                "",
                "",
                gr.update(value=f"Retrieval error: {exc}", visible=True),
            )
            return

        # Stream the answer
        accumulated = ""
        try:
            if session_value == "__all__":
                sessions = asyncio.get_event_loop().run_until_complete(
                    list_sessions(limit=30)
                )
                sids = [s.id for s in sessions]
                gen = answer_across_sessions(question, sids)
            else:
                gen = answer_question(question, session_value)

            async def _collect():
                nonlocal accumulated
                async for token in gen:
                    accumulated += token
                    yield (
                        gr.update(visible=False),
                        sources_text,
                        accumulated,
                        gr.update(visible=False),
                    )

            loop = asyncio.get_event_loop()
            ait = _collect().__aiter__()
            while True:
                try:
                    result = loop.run_until_complete(ait.__anext__())
                    yield result
                except StopAsyncIteration:
                    break

        except MeetingSummarizerError as exc:
            yield (
                gr.update(visible=False),
                sources_text,
                accumulated or "",
                gr.update(value=str(exc), visible=True),
            )

    # ── Wire up events ────────────────────────────────────────────────────────

    refresh_sessions_btn.click(
        fn=on_refresh_sessions,
        outputs=session_dropdown,
    )

    clear_btn.click(
        fn=on_clear,
        outputs=[question_input, sources_display, answer_output, error_display],
    )

    ask_btn.click(
        fn=on_ask,
        inputs=[session_dropdown, question_input],
        outputs=[error_display, sources_display, answer_output, error_display],
    )

    question_input.submit(
        fn=on_ask,
        inputs=[session_dropdown, question_input],
        outputs=[error_display, sources_display, answer_output, error_display],
    )