"""
Reusable Gradio component factories.

Keeping component definitions here rather than inline in app.py means
styling and labels are consistent across all tabs and easy to update
in one place.
"""

import gradio as gr


def audio_upload() -> gr.Audio:
    return gr.Audio(
        type="filepath",
        label="Upload meeting audio",
        show_label=True,
    )


def context_box() -> gr.Textbox:
    return gr.Textbox(
        label="Context (optional)",
        placeholder=(
            "Provide any background that helps the model — e.g. "
            "'Q3 budget review with the engineering team'"
        ),
        lines=2,
    )


def progress_box() -> gr.Textbox:
    return gr.Textbox(
        label="Pipeline progress",
        interactive=False,
        lines=4,
        max_lines=4,
    )


def summary_box() -> gr.Textbox:
    return gr.Textbox(
        label="Executive summary",
        interactive=False,
        lines=6,
        show_copy_button=True,
    )


def decisions_box() -> gr.Textbox:
    return gr.Textbox(
        label="Decisions",
        interactive=False,
        lines=5,
        show_copy_button=True,
    )


def action_items_box() -> gr.Textbox:
    return gr.Textbox(
        label="Action items",
        interactive=False,
        lines=6,
        show_copy_button=True,
    )


def open_questions_box() -> gr.Textbox:
    return gr.Textbox(
        label="Open questions",
        interactive=False,
        lines=4,
        show_copy_button=True,
    )


def transcript_box() -> gr.Textbox:
    return gr.Textbox(
        label="Full diarized transcript",
        interactive=False,
        lines=12,
        max_lines=30,
        show_copy_button=True,
    )


def language_badge() -> gr.Textbox:
    return gr.Textbox(
        label="Detected language",
        interactive=False,
        max_lines=1,
    )


def speaker_badge() -> gr.Textbox:
    return gr.Textbox(
        label="Speakers detected",
        interactive=False,
        max_lines=1,
    )


def transcript_download() -> gr.File:
    return gr.File(
        label="Download transcript (.txt)",
        interactive=False,
    )


def chat_input() -> gr.Textbox:
    return gr.Textbox(
        label="Ask a question about this meeting",
        placeholder="e.g. What did SPEAKER_00 say about the deadline?",
        lines=2,
    )


def chat_output() -> gr.Textbox:
    return gr.Textbox(
        label="Answer",
        interactive=False,
        lines=6,
        show_copy_button=True,
    )


def error_box() -> gr.Textbox:
    return gr.Textbox(
        label="Error",
        interactive=False,
        visible=False,
        lines=3,
    )


def status_md() -> gr.Markdown:
    return gr.Markdown(value="", visible=False)