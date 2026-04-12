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
from pipeline.orchestrator import PipelineProgress, PipelineResult, run_pipeline
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

    if audio_path is None:
        yield (
            "Please upload an audio file first.",
            "", "", "", "", "", "", "", None,
            gr.update(visible=False),
        )
        return

    progress_log: list[str] = []
    final_result: PipelineResult | None = None

    async def _run():
        nonlocal final_result

        async for item in run_pipeline(audio_path, context or None):

            if isinstance(item, PipelineProgress):
                line = item.stage + (f"  —  {item.detail}" if item.detail else "")

                progress_log.append(line)
                yield (
                    "\n".join(progress_log),
                    "", "", "", "", "", "", "", None,
                    gr.update(visible=False),
                )

            elif isinstance(item, str):
                # Streaming summary token — accumulate inside PipelineResult
                pass


            elif isinstance(item, PipelineResult):
                # The single final object: has output, diarized_transcript,
                # and speaker_count all in one place — no parsing required.
                final_result = item

    # Drive the async generator from Gradio's sync thread
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

    if final_result is None:
        yield (
            "Pipeline completed but produced no output.",
            "", "", "", "", "", "", "", None,
            gr.update(visible=True, value="No output produced."),
        )
        return

    output = final_result.output
    diarized_transcript = final_result.diarized_transcript
    speaker_count = final_result.speaker_count

    # ── Persist session to SQLite ─────────────────────────────────────────────
    session_id: str | None = None
    transcript_file_path = None
    save_loop = asyncio.new_event_loop()

    try:
        session_id = save_loop.run_until_complete(
            save_session(
                audio_filename=Path(audio_path).name,
                transcript=diarized_transcript,  # ← real transcript now
                output=output,
                speaker_count=speaker_count,  # ← real speaker count now
                context=context or None,
            )
        )
        logger.info("Session saved: %s", session_id)

        # Write transcript .txt file for the download button
        transcript_file_path = save_loop.run_until_complete(
            save_transcript_file(
                session_id=session_id,
                transcript=diarized_transcript,
                audio_filename=Path(audio_path).name,
            )
        )

    except Exception as exc:
        logger.warning("Session save failed (non-fatal): %s", exc)
    finally:
        save_loop.close()

    # ── Build RAG index in the background ────────────────────────────────────
    if session_id:
        index_loop = asyncio.new_event_loop()
        try:
            index_loop.run_until_complete(
                build_index(session_id, diarized_transcript)
            )
        except Exception as exc:
            logger.warning("RAG index build failed (non-fatal): %s", exc)
        finally:
            index_loop.close()

    # ── Final UI update ───────────────────────────────────────────────────────
    progress_log.append("Done.")
    yield (
        "\n".join(progress_log),
        output.summary,
        _format_decisions(output),
        _format_action_items(output),
        _format_open_questions(output),
        diarized_transcript,  # ← real transcript now
        f"{output.language_name} ({output.language_code})",
        str(speaker_count),  # ← real speaker count now
        str(transcript_file_path) if transcript_file_path else None,
        # download file
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