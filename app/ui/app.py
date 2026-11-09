"""Small Gradio interface; backend objects never become browser state."""

from copy import deepcopy

import gradio as gr

from .service import safe_error


def document_dropdown(choices):
    return gr.Dropdown(choices=[("All indexed documents", ""), *choices], value="",
                       label="Search in", interactive=True)


def build_app(service):
    def refresh():
        try:
            return document_dropdown(service.documents()), "Document list refreshed."
        except Exception as error:
            return document_dropdown([]), safe_error(error)

    def index_files(files, progress=gr.Progress()):
        try:
            message, choices = service.index_uploads(files, progress)
            return message, document_dropdown(choices)
        except Exception as error:
            return safe_error(error), gr.skip()

    def chat(question, document_id, history):
        try:
            answer, sources = service.answer(question, document_id)
            messages = deepcopy(history or [])[-38:]
            messages.extend([{"role": "user", "content": question},
                             {"role": "assistant", "content": answer}])
            return "", messages, messages, sources, "Answer complete."
        except Exception as error:
            return question, gr.skip(), gr.skip(), "No supporting citations.", safe_error(error)

    def clear():
        return [], [], "No supporting citations.", "Chat cleared. Indexed documents are unchanged."

    with gr.Blocks(title="hujjat-ai", analytics_enabled=False, delete_cache=(3600, 3600)) as demo:
        gr.Markdown("# hujjat-ai\nAsk questions grounded in your documents.")
        gr.Markdown("Local document library. Each question is answered independently; "
                    "chat history is for display, not conversational memory.")
        history = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                uploads = gr.File(label="Documents (up to 10, 20 MB each)", file_count="multiple",
                    file_types=[".pdf", ".docx", ".txt", ".md", ".markdown"], type="filepath")
                index_button = gr.Button("Index documents", variant="primary")
                index_status = gr.Textbox(label="Indexing status", interactive=False, lines=5)
                document = gr.Dropdown(choices=[("All indexed documents", "")], value="",
                                       label="Search in", interactive=True)
                refresh_button = gr.Button("Refresh documents")
            with gr.Column(scale=2):
                conversation = gr.Chatbot(label="Document chat", render_markdown=False,
                    sanitize_html=True, allow_tags=False, allow_file_downloads=False, height=420)
                question = gr.Textbox(label="Question", placeholder="Ask a complete question about your documents")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear_button = gr.Button("Clear chat")
                status = gr.Textbox(label="Status", interactive=False)
                with gr.Accordion("Validated supporting sources", open=True):
                    sources = gr.Textbox(value="No supporting citations.", label="Citations and evidence",
                                         interactive=False, lines=6)
        shared = {"concurrency_id": "backend", "concurrency_limit": 1, "api_visibility": "private"}
        # Keep orig_name metadata: service validates the raw path against our private
        # upload root, size, filename, and link checks instead of File.preprocess().
        index_button.click(index_files, uploads, [index_status, document], preprocess=False, **shared)
        refresh_button.click(refresh, outputs=[document, status], **shared)
        demo.load(refresh, outputs=[document, status], **shared)
        for event in (send.click, question.submit):
            event(chat, [question, document, history], [question, conversation, history, sources, status], **shared)
        clear_button.click(clear, outputs=[conversation, history, sources, status], **shared)
    demo.queue(max_size=16, default_concurrency_limit=1)
    return demo
