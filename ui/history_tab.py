"""
History tab — browse past meeting sessions, view their details,
and delete them (which also removes the ChromaDB index).
"""

import asyncio

import gradio as gr

from core.exceptions import StorageError
from core.logger import get_logger
from rag.vector_store import delete_collection
from storage.session_repo import (
    delete_session,
    get_session,
    list_sessions,
    save_transcript_file,
)

logger = get_logger(__name__)


def _format_session_list(sessions) -> str:
    """Format session list for display in a textbox."""
    if not sessions:
        return "No meetings processed yet."
    lines = []
    for s in sessions:
        lang = f"[{s.language_name}]"
        speakers = f"{s.speaker_count} speaker(s)"
        lines.append(
            f"• {s.created_at[:16].replace('T', ' ')}  {lang}  {speakers}  "
            f"— {s.audio_filename}\n"
            f"  ID: {s.id}\n"
            f"  {s.summary[:120]}{'…' if len(s.summary) > 120 else ''}"
        )
    return "\n\n".join(lines)


def _format_session_detail(detail) -> tuple[str, str, str, str, str, str]:
    """
    Unpack a SessionDetail into display strings for each output component.
    Returns: (summary, decisions, action_items, open_questions, transcript, meta)
    """
    if detail is None:
        empty = "Session not found."
        return empty, empty, empty, empty, empty, empty

    decisions = "\n".join(f"• {d}" for d in detail.decisions) or "None recorded."
    action_items = "\n".join(
        f"• [{item.owner}] {item.task}"
        + (f"  (due: {item.due_date})" if item.due_date else "")
        for item in detail.action_items
    ) or "None recorded."
    open_questions = "\n".join(
        f"• {q}" for q in detail.open_questions
    ) or "None recorded."
    meta = (
        f"Language: {detail.language_name} | "
        f"Speakers: {detail.speaker_count} | "
        f"Created: {detail.created_at[:16].replace('T', ' ')}"
    )
    return (
        detail.summary,
        decisions,
        action_items,
        open_questions,
        detail.transcript,
        meta,
    )


def build_history_tab() -> None:
    """
    Render the History tab inside the currently active gr.Blocks context.
    Must be called from within a `with gr.Blocks():` block.
    """
    gr.Markdown("## Past meetings")
    gr.Markdown(
        "Paste a session ID from the list below into the lookup box to view "
        "its full details, re-download the transcript, or delete it."
    )

    with gr.Row():
        refresh_btn = gr.Button("Refresh list", variant="secondary", scale=1)

    session_list_display = gr.Textbox(
        label="Recent sessions (newest first)",
        interactive=False,
        lines=10,
        max_lines=15,
    )

    gr.Markdown("### Session detail")

    with gr.Row():
        session_id_input = gr.Textbox(
            label="Session ID",
            placeholder="Paste a session ID from the list above…",
            scale=3,
        )
        load_btn  = gr.Button("Load",   variant="primary",   scale=1)
        delete_btn = gr.Button("Delete", variant="stop",      scale=1)

    detail_meta = gr.Textbox(
        label="Session info", interactive=False, max_lines=1
    )

    with gr.Tabs():
        with gr.Tab("Summary"):
            detail_summary = gr.Textbox(
                interactive=False, lines=5
            )
        with gr.Tab("Decisions"):
            detail_decisions = gr.Textbox(
                interactive=False, lines=5
            )
        with gr.Tab("Action items"):
            detail_actions = gr.Textbox(
                interactive=False, lines=6
            )
        with gr.Tab("Open questions"):
            detail_questions = gr.Textbox(
                interactive=False, lines=4
            )
        with gr.Tab("Full transcript"):
            detail_transcript = gr.Textbox(
                interactive=False, lines=12, max_lines=30
            )

    download_btn = gr.Button("Prepare transcript download", variant="secondary")
    transcript_file = gr.File(label="Download transcript", interactive=False)

    status_msg = gr.Markdown(visible=False)

    # ── Event handlers ────────────────────────────────────────────────────────

    def on_refresh():
        try:
            sessions = asyncio.get_event_loop().run_until_complete(list_sessions())
            return _format_session_list(sessions)
        except StorageError as exc:
            return f"Error loading sessions: {exc}"

    def on_load(session_id: str):
        sid = session_id.strip()
        if not sid:
            empty = "Please enter a session ID."
            return empty, empty, empty, empty, empty, empty, gr.update(visible=False)
        try:
            detail = asyncio.get_event_loop().run_until_complete(get_session(sid))
            summary, decisions, actions, questions, transcript, meta = (
                _format_session_detail(detail)
            )
            return summary, decisions, actions, questions, transcript, meta, gr.update(visible=False)
        except StorageError as exc:
            msg = f"Error loading session: {exc}"
            return msg, msg, msg, msg, msg, msg, gr.update(visible=False)

    def on_delete(session_id: str):
        sid = session_id.strip()
        if not sid:
            return gr.update(value="Please enter a session ID.", visible=True), ""
        try:
            deleted = asyncio.get_event_loop().run_until_complete(delete_session(sid))
            delete_collection(sid)
            if deleted:
                sessions = asyncio.get_event_loop().run_until_complete(list_sessions())
                return (
                    gr.update(value=f"Session {sid[:8]}… deleted.", visible=True),
                    _format_session_list(sessions),
                )
            return gr.update(value="Session not found.", visible=True), ""
        except StorageError as exc:
            return gr.update(value=f"Delete failed: {exc}", visible=True), ""

    def on_download(session_id: str):
        sid = session_id.strip()
        if not sid:
            return None
        try:
            detail = asyncio.get_event_loop().run_until_complete(get_session(sid))
            if detail is None:
                return None
            path = asyncio.get_event_loop().run_until_complete(
                save_transcript_file(sid, detail.transcript, detail.audio_filename)
            )
            return str(path)
        except Exception as exc:
            logger.warning("Transcript download failed: %s", exc)
            return None

    # Wire up
    refresh_btn.click(fn=on_refresh, outputs=session_list_display)

    load_btn.click(
        fn=on_load,
        inputs=session_id_input,
        outputs=[
            detail_summary, detail_decisions, detail_actions,
            detail_questions, detail_transcript, detail_meta, status_msg,
        ],
    )

    delete_btn.click(
        fn=on_delete,
        inputs=session_id_input,
        outputs=[status_msg, session_list_display],
    )

    download_btn.click(
        fn=on_download,
        inputs=session_id_input,
        outputs=transcript_file,
    )