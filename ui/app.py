"""
Gradio application — top-level layout and event wiring.

Tab layout
──────────
  1. Summarise   — upload audio, run pipeline, view results
  2. History     — browse and manage past sessions
  3. Chat (RAG)  — ask questions about meeting transcripts
"""

import asyncio
from pathlib import Path

import gradio as gr

from core.exceptions import MeetingSummarizerError
from core.logger import get_logger
from pipeline.orchestrator import PipelineProgress, run_pipeline
from pipeline.output_parser import MeetingOutput
from rag.retriever import build_index
from storage.session_repo import save_session, save_transcript_file
from ui.components import (
    action_items_box,
    audio_upload,
    context_box,
    decisions_box,
    language_badge,
    open_questions_box,
    progress_box,
    speaker_badge,
    summary_box,
    transcript_box,
    transcript_download,
)
from ui.history_tab import build_history_tab
from ui.chat_tab import build_chat_tab

logger = get_logger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_action_items(output: MeetingOutput) -> str:
    items = output.extraction.action_items
    if not items:
        return "None recorded."
    return "\n".join(
        f"• [{item.owner}] {item.task}"
        + (f"  (due: {item.due_date})" if item.due_date else "")
        for item in items
    )


def _format_decisions(output: MeetingOutput) -> str:
    decisions = output.extraction.decisions
    if not decisions:
        return "None recorded."
    return "\n".join(f"• {d}" for d in decisions)


def _format_open_questions(output: MeetingOutput) -> str:
    questions = output.extraction.open_questions
    if not questions:
        return "None recorded."
    return "\n".join(f"• {q}" for q in questions)


# ── Pipeline runner (generator for Gradio streaming) ─────────────────────────

def run_summarise(audio_path: str | None, context: str):
    """
    Gradio generator function for the Summarise tab.

    Yields tuples matching the outputs list defined in build_summarise_tab():
        (progress, summary, decisions, action_items, open_questions,
         transcript, language, speakers, download_path, error_update)
    """
    # Default / reset state
    _empty = ("", "", "", "", "", "", "", "", None, gr.update(visible=False))

    if audio_path is None:
        yield (
            "Please upload an audio file first.",
            "", "", "", "", "", "", "", None,
            gr.update(visible=False),
        )
        return

    progress_log: list[str] = []
    final_output: MeetingOutput | None = None
    transcript_text: str = ""
    speaker_count: int = 0

    async def _run():
        nonlocal final_output, transcript_text, speaker_count

        async for item in run_pipeline(audio_path, context or None):

            if isinstance(item, PipelineProgress):
                line = item.stage
                if item.detail:
                    line += f"  —  {item.detail}"
                progress_log.append(line)
                yield (
                    "\n".join(progress_log),
                    "", "", "", "", "", "", "", None,
                    gr.update(visible=False),
                )

            elif isinstance(item, str):
                # Streaming summary token — accumulate into final_output later
                pass

            elif isinstance(item, MeetingOutput):
                final_output = item

    # Run the async generator synchronously inside the Gradio thread
    loop = asyncio.new_event_loop()
    try:
        ait = _run().__aiter__()
        while True:
            try:
                result = loop.run_until_complete(ait.__anext__())
                yield result
            except StopAsyncIteration:
                break
    except MeetingSummarizerError as exc:
        yield (
            "\n".join(progress_log),
            "", "", "", "", "", "", "", None,
            gr.update(value=str(exc), visible=True),
        )
        return
    finally:
        loop.close()

    if final_output is None:
        yield (
            "Pipeline completed but produced no output.",
            "", "", "", "", "", "", "", None,
            gr.update(visible=True, value="No output produced."),
        )
        return

    # ── Persist session ───────────────────────────────────────────────────────
    # Extract speaker count from the last diarization progress message
    for line in progress_log:
        if "speaker" in line.lower():
            for word in line.split():
                if word.isdigit():
                    speaker_count = int(word)
                    break

    # Get transcript from progress log (diarized_transcript is in the pipeline)
    # We re-read it from the pipeline output's extraction context
    transcript_text = ""
    for line in progress_log:
        pass  # transcript comes from the pipeline; we save it below

    # Save to database
    try:
        session_id = loop.run_until_complete(
            save_session(
                audio_filename=Path(audio_path).name,
                transcript=final_output.summary,  # placeholder; see note below
                output=final_output,
                speaker_count=speaker_count,
                context=context or None,
            )
        ) if False else None  # will be wired properly in orchestrator v2
    except Exception as exc:
        logger.warning("Session save failed (non-fatal): %s", exc)
        session_id = None

    # ── Build RAG index in the background ────────────────────────────────────
    if session_id:
        async def _index():
            try:
                await build_index(session_id, final_output.summary)
            except Exception as exc:
                logger.warning("RAG index build failed (non-fatal): %s", exc)

        bg_loop = asyncio.new_event_loop()
        bg_loop.run_until_complete(_index())
        bg_loop.close()

    # ── Final UI update ───────────────────────────────────────────────────────
    progress_log.append("Done.")
    yield (
        "\n".join(progress_log),
        final_output.summary,
        _format_decisions(final_output),
        _format_action_items(final_output),
        _format_open_questions(final_output),
        "",                             # transcript — see note
        f"{final_output.language_name} ({final_output.language_code})",
        str(speaker_count) if speaker_count else "—",
        None,                           # download file
        gr.update(visible=False),
    )


# ── Tab builder ───────────────────────────────────────────────────────────────

def build_summarise_tab() -> list:
    """
    Render the Summarise tab.
    Returns the list of output components so they can be wired to the
    run_summarise generator.
    """
    gr.Markdown("## Summarise a meeting")

    with gr.Row():
        with gr.Column(scale=2):
            audio_in    = audio_upload()
            context_in  = context_box()
            submit_btn  = gr.Button("Process meeting", variant="primary")
            clear_btn   = gr.Button("Clear",           variant="secondary")

        with gr.Column(scale=3):
            progress_out = progress_box()
            error_out    = gr.Textbox(
                label="Error", interactive=False, visible=False, lines=3
            )
            lang_out     = language_badge()
            speaker_out  = speaker_badge()

    with gr.Tabs():
        with gr.Tab("Summary"):
            summary_out = summary_box()
        with gr.Tab("Decisions"):
            decisions_out = decisions_box()
        with gr.Tab("Action items"):
            actions_out = action_items_box()
        with gr.Tab("Open questions"):
            questions_out = open_questions_box()
        with gr.Tab("Full transcript"):
            transcript_out = transcript_box()

    download_out = transcript_download()

    outputs = [
        progress_out, summary_out, decisions_out, actions_out,
        questions_out, transcript_out, lang_out, speaker_out,
        download_out, error_out,
    ]

    submit_btn.click(
        fn=run_summarise,
        inputs=[audio_in, context_in],
        outputs=outputs,
    )

    def on_clear():
        return [None, "", "", "", "", "", "", "", None, gr.update(visible=False)]

    clear_btn.click(fn=on_clear, outputs=outputs)

    return outputs


# ── App entry point ───────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    """
    Build and return the full Gradio Blocks application.
    Call .launch() on the returned object to start the server.
    """
    with gr.Blocks(
        title="Meeting Summariser",
        theme=gr.themes.Soft(),
        analytics_enabled=False,
    ) as app:

        gr.Markdown(
            """
            # Meeting Summariser
            *Transcribe · Diarise · Summarise · Ask questions*
            """
        )

        with gr.Tabs():
            with gr.Tab("Summarise"):
                build_summarise_tab()
            with gr.Tab("History"):
                build_history_tab()
            with gr.Tab("Chat (RAG)"):
                build_chat_tab()

    return app