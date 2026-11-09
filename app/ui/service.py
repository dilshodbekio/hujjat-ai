"""Local UI boundary: one worker owns Qdrant; public results contain no paths."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from hashlib import sha256
from pathlib import Path
from queue import Empty, Queue
from tempfile import TemporaryDirectory
from threading import Lock
from zipfile import ZipFile

from app.generation.gemini import GeminiAnswerGenerator
from app.generation.models import AnswerValidationError, ContextLimitError, GenerationError
from app.generation.pipeline import GenerationPipeline
from app.ingestion.chunking import chunk_document
from app.ingestion.pipeline import IngestionError, ingest_document
from app.retrieval.config import ConfigurationError
from app.retrieval.embeddings import AuthenticationError, EmbeddingError
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.vector_store import StorageError


EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_FILES = 10


class UIError(ValueError):
    """Controlled message safe for local UI display."""


def safe_error(error):
    if isinstance(error, UIError):
        return str(error)
    if isinstance(error, (ConfigurationError, AuthenticationError)):
        return "Check your Gemini key and settings in .env, then restart the app."
    if isinstance(error, IngestionError):
        return "Could not read this document. Check its format; scanned PDFs may need OCR."
    if isinstance(error, AnswerValidationError):
        return "The answer failed evidence validation. Please rephrase the question and try again."
    if isinstance(error, ContextLimitError):
        return "The document context could not fit. Try a more specific question or select one document."
    if isinstance(error, (EmbeddingError, GenerationError)):
        return "Gemini could not complete the request. Check connectivity, model access, or quota and try again."
    if isinstance(error, StorageError):
        return "The local database is unavailable or incompatible. Close other Qdrant clients and check settings."
    return "The request could not be completed. Please try again."


def original_filename(name):
    if (not isinstance(name, str) or not name.strip() or len(name) > 240 or
            any(character in name for character in '/\\:') or
            any(ord(character) < 32 for character in name) or name in {".", ".."}):
        raise UIError("The uploaded filename is invalid.")
    if Path(name).suffix.lower() not in EXTENSIONS:
        raise UIError("Upload PDF, DOCX, TXT, MD, or Markdown files only.")
    return name


class _Backend:
    def __init__(self, retrieval_config, generation_config):
        self.retrieval = RetrievalPipeline(retrieval_config)
        self.generator = None
        try:
            self.generator = GeminiAnswerGenerator(retrieval_config.api_key, generation_config)
            self.rag = GenerationPipeline(generation_config, retrieval=self.retrieval, generator=self.generator)
        except Exception:
            try:
                if self.generator is not None:
                    self.generator.close()
            finally:
                self.retrieval.close()
            raise

    def close(self):
        try:
            self.retrieval.close()
        finally:
            self.generator.close()


class UIService:
    def __init__(self, retrieval_config, generation_config, upload_root, *, backend_factory=None):
        self.upload_root = Path(upload_root).resolve()
        self.question_limit = generation_config.question_chars
        self._lock = Lock()
        self._closed = False
        self._documents = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hujjat-backend")
        factory = backend_factory or (lambda: _Backend(retrieval_config, generation_config))
        try:
            self._backend = self._executor.submit(factory).result()
        except Exception as error:
            self._executor.shutdown(wait=True)
            raise UIError(safe_error(error)) from None

    def _submit(self, function, *args):
        with self._lock:
            if self._closed:
                raise UIError("The app has shut down. Restart it to continue.")
            return self._executor.submit(function, *args)

    def _inventory(self):
        documents = self._backend.retrieval.store.list_documents()
        self._documents = {"doc_" + sha256(item["source_path"].encode()).hexdigest(): item for item in documents}
        return [(f"{Path(item['filename'].replace(chr(92), '/')).name} · {token[4:16]}", token)
                for token, item in self._documents.items()]

    def documents(self):
        return self._submit(self._inventory).result()

    def _index_one(self, upload, stages):
        if not isinstance(upload, dict):
            raise UIError("Please select a file using the upload control.")
        filename = original_filename(upload.get("orig_name"))
        stages.put((.05, "Validating upload"))
        path_value = upload.get("path")
        if not isinstance(path_value, str):
            raise UIError("The uploaded file is unavailable; please upload it again.")
        path = Path(path_value)
        resolved = path.resolve()
        if not resolved.is_relative_to(self.upload_root) or not resolved.is_file():
            raise UIError("The uploaded file is unavailable; please upload it again.")
        if any(item.is_symlink() or item.is_junction() for item in [path, *path.parents]
               if item != self.upload_root and item.is_relative_to(self.upload_root)):
            raise UIError("Linked upload files are not supported.")
        if resolved.stat().st_size > MAX_FILE_BYTES:
            raise UIError("Each file must be no larger than 20 MB.")
        # Own the processing copy, not the browser's upload cache entry.
        with TemporaryDirectory(prefix="hujjat-process-") as directory:
            staged = Path(directory) / ("document" + Path(filename).suffix.lower())
            digest = sha256()
            size = 0
            with resolved.open("rb") as source, staged.open("wb") as target:
                while block := source.read(1024 * 1024):
                    size += len(block)
                    if size > MAX_FILE_BYTES:
                        raise UIError("Each file must be no larger than 20 MB.")
                    digest.update(block)
                    target.write(block)
            if staged.suffix == ".docx":
                with ZipFile(staged) as archive:
                    if sum(item.file_size for item in archive.infolist()) > 100 * 1024 * 1024:
                        raise UIError("This DOCX expands beyond the supported processing size.")
            stages.put((.2, "Extracting text"))
            document = ingest_document(staged)
            document.filename = filename
            document.source_path = f"upload:{digest.hexdigest()}:{filename}"
            stages.put((.4, "Splitting into chunks"))
            chunks = chunk_document(document)
            missing = self._backend.retrieval.store.missing_chunks(chunks)
            if not missing:
                return f"{filename}: already indexed ({len(chunks)} chunks)."
            stages.put((.6, "Embedding and saving missing chunks"))
            self._backend.retrieval.index_chunks(missing)
            if self._backend.retrieval.store.missing_chunks(chunks):
                raise UIError("Indexing is incomplete. Re-upload to resume missing chunks.")
            warning = " Some pages have no text; scanned PDFs may need OCR." if any(
                not part.text.strip() for part in document.parts) else ""
            return f"{filename}: indexed {len(missing)} new chunks; {len(chunks)} total." + warning

    def index_uploads(self, uploads, progress=None):
        if not isinstance(uploads, list) or not uploads:
            raise UIError("Select at least one document to index.")
        if len(uploads) > MAX_FILES:
            raise UIError("Upload at most 10 files at a time.")
        messages = []
        for index, upload in enumerate(uploads):
            stages = Queue()
            future = self._submit(self._index_one, upload, stages)
            try:
                while True:
                    try:
                        while True:
                            fraction, stage = stages.get_nowait()
                            if progress:
                                progress((index + fraction) / len(uploads), desc=stage)
                    except Empty:
                        pass
                    try:
                        messages.append(future.result(timeout=.1))
                        break
                    except TimeoutError:
                        if future.done():
                            messages.append(future.result())
                            break
            except Exception as error:
                messages.append(f"File {index + 1}: " + safe_error(error))
            if progress:
                progress((index + 1) / len(uploads), desc="File finished")
        return "\n".join(messages), self.documents()

    def _answer(self, question, document_id):
        if not isinstance(question, str) or not question.strip() or len(question) > self.question_limit:
            raise UIError(f"Enter a question between 1 and {self.question_limit} characters.")
        self._inventory()
        filters = None
        if document_id:
            if document_id not in self._documents:
                raise UIError("The selected document is no longer available. Refresh the document list.")
            filters = {"source_path": self._documents[document_id]["source_path"]}
        result = self._backend.rag.answer(question, filters=filters)
        # ONLY citations/evidence that survived GenerationPipeline validation.
        sources = []
        by_id = {citation.source_id: citation for citation in result.citations}
        answer_lines = []
        for claim in result.claims:
            labels = []
            for evidence in claim.evidence:
                citation = by_id[evidence.source_id]
                filename = Path(citation.chunk.filename.replace("\\", "/")).name
                primary = citation.primary_location
                label = filename
                if primary and primary.page_number is not None:
                    label += f", page {primary.page_number}"
                elif primary and primary.page_label:
                    label += f", page label {primary.page_label}"
                line = label + "\nEvidence: " + evidence.quote
                if line not in sources:
                    sources.append(line)
                if label not in labels:
                    labels.append(label)
            answer_lines.append(claim.text.strip() + " " + " ".join(f"[{label}]" for label in labels))
        # Rebuild displayed labels from filenames only, including for legacy payloads.
        answer = "\n".join(answer_lines) if result.status == "answered" else result.answer
        return answer, "\n\n".join(sources) if sources else "No supporting citations."

    def answer(self, question, document_id=None):
        return self._submit(self._answer, question, document_id).result()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            cleanup = self._executor.submit(self._backend.close)
        try:
            cleanup.result()
        finally:
            self._executor.shutdown(wait=True)
