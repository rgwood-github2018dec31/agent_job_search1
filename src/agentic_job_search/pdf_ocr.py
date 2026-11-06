'''OCR for PDFs with no usable text layer, done entirely on the local model.

A PDF whose extracted text is shorter than PDF_MIN_TEXT_CHARS (a phone photo saved as a PDF, a
scan) has each page rendered whole to an image and transcribed by MODEL_NAME_LOCAL_OCR through the
local Ollama MCP server. The transcription is cached per file content (sha256), so a file is read
once however often it is renamed or moved.

Such a file is also CATEGORIZED locally (`categorize_locally`): the first one seen was an identity
document, and its transcription must not reach a remote model. Its quote is checked by regex only
(evidence.find_quote), never relocated by the remote evidence model, for the same reason.

The transcription call is exempt from Verify Source Reference: its output is the input text that
later calls cite, not a claim about it.
'''
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
from pathlib import Path

import pypdfium2
from utils_tools_n_agents_common.mcp_client import call_mcp_tool

from agentic_job_search import cache_io, evidence
from agentic_job_search.config import (
    LLM_LOCAL_MCP_URL,
    LLM_LOCAL_PURPOSE_PDF_CATEGORIZE,
    LLM_LOCAL_PURPOSE_PDF_OCR,
    LLM_LOCAL_SERVICE_NAME,
    LLM_MCP_CALL_TIMEOUT_SECONDS,
    MODEL_NAME_LOCAL_OCR,
    PDF_MIN_TEXT_CHARS,
    PDF_OCR_MAX_PAGES,
    PDF_OCR_RENDER_SCALE,
    PDF_OCR_TEMPERATURE,
)
from agentic_job_search.preferences import RUN_DIR
from agentic_job_search.text_budget import quote_for_error

logger = logging.getLogger(__name__)

PDF_OCR_CACHE_PATH = RUN_DIR / 'pdf_ocr_cache.yaml'

OCR_PROMPT = (
    'Transcribe all text in this image verbatim, in reading order. Output only the text, with no '
    'commentary, headings or formatting of your own. If the image holds no text, output nothing.'
)
CATEGORIZE_SYSTEM = (
    'You label a document. Use "saved_jd" for a job description or job posting, "other" for '
    'unrelated content, or a short snake_case label you derive such as "resume", "contract", '
    '"invoice", "identity_document". Also return ONE passage of the document that shows the '
    'category, copied exactly as written: no labels, no quote marks, no joining of fragments.'
)
CATEGORIZE_SCHEMA = {
    'type': 'object',
    'properties': {'category': {'type': 'string'}, 'quote': {'type': 'string'}},
    'required': ['category', 'quote'],
}


def needs_ocr(pages: list[str]) -> bool:
    '''True when the text layer is too thin to read the document from.'''
    return sum(len(page.strip()) for page in pages) < PDF_MIN_TEXT_CHARS


def file_hash_sha256(pdf: Path) -> str:
    return hashlib.sha256(pdf.read_bytes()).hexdigest()


def cached_ocr_pages(pdf: Path) -> list[str] | None:
    '''The cached transcription of this file's content, or None if it was never transcribed.'''
    entry = cache_io.load(PDF_OCR_CACHE_PATH, 'PDF OCR cache').get(file_hash_sha256(pdf))
    pages = entry.get('pages') if isinstance(entry, dict) else None
    return pages if isinstance(pages, list) and all(isinstance(p, str) for p in pages) else None


def render_pages_png_base64(pdf: Path) -> tuple[list[str], int]:
    '''Each page rendered whole, as base64 PNG, capped at PDF_OCR_MAX_PAGES. Returns (images, page count).'''
    document = pypdfium2.PdfDocument(pdf)
    try:
        page_count = len(document)
        images = []
        for index in range(min(page_count, PDF_OCR_MAX_PAGES)):
            buffer = io.BytesIO()
            document[index].render(scale=PDF_OCR_RENDER_SCALE).to_pil().save(buffer, format='PNG')
            images.append(base64.b64encode(buffer.getvalue()).decode('ascii'))
        return images, page_count
    finally:
        document.close()


def _chat_content(raw: str, what: str) -> str:
    '''The message text of a local chat reply, raising with the whole reply on any malformed answer.'''
    try:
        data = json.loads(raw)
    except ValueError as ex:
        raise ValueError(f'{what}: reply is not JSON ({ex}): {quote_for_error(raw, what)!r}') from ex
    if not isinstance(data, dict):
        raise ValueError(f'{what}: reply is not an object: {quote_for_error(raw, what)!r}')
    if data.get('error') or data.get('ok') is False:
        raise RuntimeError(f'{what}: server reported an error: {quote_for_error(raw, what)!r}')
    if data.get('done_reason') == 'length':
        raise RuntimeError(f'{what}: reply truncated (done_reason=length): {quote_for_error(raw, what)!r}')
    content = (data.get('message') or {}).get('content') if isinstance(data.get('message'), dict) else None
    if not isinstance(content, str):
        raise ValueError(f'{what}: reply has no message.content string: {quote_for_error(raw, what)!r}')
    return content


async def _chat(messages: list[dict], purpose_slug: str, what: str, **extra) -> str:
    args = {
        'model': MODEL_NAME_LOCAL_OCR, 'messages': messages, 'temperature': PDF_OCR_TEMPERATURE,
        'think': False, 'service_name': LLM_LOCAL_SERVICE_NAME, 'purpose_slug': purpose_slug, **extra,
    }
    try:
        raw = await call_mcp_tool(LLM_LOCAL_MCP_URL, 'chat', args, timeout_seconds=LLM_MCP_CALL_TIMEOUT_SECONDS)
    except Exception as ex:
        raise RuntimeError(f'{what}: local chat at {LLM_LOCAL_MCP_URL} failed (model={MODEL_NAME_LOCAL_OCR}): '
                           f'{type(ex).__name__}: {ex}') from ex
    return _chat_content(raw, what)


async def ocr_pdf_pages(pdf: Path) -> list[str]:
    '''The text of each page as the local vision model reads it; cached per file content.

    Raises on any failure (server down, malformed reply), naming the file, model and page; the
    caller decides what one file's failure means.
    '''
    if (cached := cached_ocr_pages(pdf)) is not None:
        logger.info(f'PDF OCR: {pdf.name} read from the cache ({sum(map(len, cached))} chars)')
        return cached
    images, page_count = render_pages_png_base64(pdf)
    if page_count > len(images):
        logger.warning(f'PDF OCR: {pdf.name} has {page_count} pages; only the first {len(images)} are '
                       f'transcribed (PDF_OCR_MAX_PAGES) and {page_count - len(images)} are left unread')
    pages = []
    for number, image in enumerate(images, start=1):
        what = f'PDF OCR of {pdf.name} page {number}/{page_count}'
        pages.append((await _chat([{'role': 'user', 'content': OCR_PROMPT, 'images': [image]}],
                                  LLM_LOCAL_PURPOSE_PDF_OCR, what)).strip())
    cache = cache_io.load(PDF_OCR_CACHE_PATH, 'PDF OCR cache')
    cache[file_hash_sha256(pdf)] = {'file': pdf.name, 'model': MODEL_NAME_LOCAL_OCR,
                                    'pages_total': page_count, 'pages': pages}
    cache_io.save(PDF_OCR_CACHE_PATH, cache, 'PDF OCR cache')
    logger.info(f'PDF OCR: {pdf.name} transcribed by {MODEL_NAME_LOCAL_OCR}: {len(pages)} page(s), '
                f'{sum(map(len, pages))} chars')
    return pages


async def categorize_locally(text: str, filename: str) -> str | None:
    '''A category for OCR'd text, from the local model only. None when its quote is not in the text.'''
    what = f'local PDF categorization of {filename}'
    content = await _chat([{'role': 'system', 'content': CATEGORIZE_SYSTEM},
                           {'role': 'user', 'content': f'Filename: {filename}\n\nContent:\n{text}'}],
                          LLM_LOCAL_PURPOSE_PDF_CATEGORIZE, what, output_format=CATEGORIZE_SCHEMA)
    try:
        answer = json.loads(content)
    except ValueError as ex:
        raise ValueError(f'{what}: content is not JSON ({ex}): {quote_for_error(content, what)!r}') from ex
    category = str(answer.get('category') or '').strip() if isinstance(answer, dict) else ''
    quote = str(answer.get('quote') or '') if isinstance(answer, dict) else ''
    if not category:
        raise ValueError(f'{what}: no category in {quote_for_error(content, what)!r}')
    if evidence.find_quote(quote, text) is None:
        # Regex only: relocating the quote would mean sending this text to the remote evidence model.
        logger.warning(f'Hallucination rejected (pdf_categorize_local, {filename}): category={category!r} '
                       f'cited {quote!r}, which is not in the transcription ({len(text)} chars) — left '
                       'uncategorized, asked again next run')
        return None
    return category
