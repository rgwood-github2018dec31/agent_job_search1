'''PDF OCR: a scan is read and labelled on the local model only, once per file content.'''
import json

import agentic_job_search.tools_generic as tools
import pypdf
import pytest
from agentic_job_search import pdf_ocr

LIVE_PAGE_WIDTH, LIVE_PAGE_HEIGHT, LIVE_TEXT_MARGIN, LIVE_FONT_SIZE = 800, 300, 40, 48
JD_TEXT = 'About the job\nSenior AI Engineer\nYou will build agentic systems in Python.'


@pytest.fixture
def scanned_pdf(tmp_path, monkeypatch):
    '''A real two-page PDF with no text layer, and the OCR cache redirected into tmp_path.'''
    monkeypatch.setattr(pdf_ocr, 'PDF_OCR_CACHE_PATH', tmp_path / 'pdf_ocr_cache.yaml')
    writer = pypdf.PdfWriter()
    for _ in range(2):
        writer.add_blank_page(width=200, height=200)
    pdf = tmp_path / 'IMG_1.pdf'
    with pdf.open('wb') as handle:
        writer.write(handle)
    return pdf


def _fake_local_chat(calls, ocr_text=JD_TEXT, category='saved_jd', quote='Senior AI Engineer'):
    async def fake_call(url, tool, args, timeout_seconds=None):
        calls.append(args)
        assert url == pdf_ocr.LLM_LOCAL_MCP_URL and tool == 'chat'
        if args['purpose_slug'] == pdf_ocr.LLM_LOCAL_PURPOSE_PDF_OCR:
            assert args['messages'][0]['images'], 'every OCR call carries the page image'
            content = ocr_text
        else:
            content = json.dumps({'category': category, 'quote': quote})
        return json.dumps({'message': {'role': 'assistant', 'content': content}, 'done_reason': 'stop'})
    return fake_call


async def _no_remote(*args, **kwargs):
    raise AssertionError('a scanned PDF must never reach the remote categorizer')


async def test_scanned_pdf_is_ocrd_once_and_categorized_locally(scanned_pdf, monkeypatch):
    calls = []
    monkeypatch.setattr(pdf_ocr, 'call_mcp_tool', _fake_local_chat(calls))
    monkeypatch.setattr(tools, '_categorize_pdf_text', _no_remote)

    await tools.categorize_save_dir_pdfs(save_dir=scanned_pdf.parent)

    assert (scanned_pdf.parent / 'cat-saved_jd-IMG_1.pdf').exists()
    ocr_calls = [c for c in calls if c['purpose_slug'] == pdf_ocr.LLM_LOCAL_PURPOSE_PDF_OCR]
    assert len(ocr_calls) == 2  # one per page
    assert all(c['model'] == pdf_ocr.MODEL_NAME_LOCAL_OCR for c in calls)

    # The transcription is cached per content: the renamed file is read from it, with no model call.
    calls.clear()
    assert await pdf_ocr.ocr_pdf_pages(scanned_pdf.parent / 'cat-saved_jd-IMG_1.pdf') == [JD_TEXT, JD_TEXT]
    assert calls == []


async def test_a_pdf_with_a_text_layer_is_never_ocrd(tmp_path, monkeypatch):
    class TextPage:
        def extract_text(self):
            return JD_TEXT * 3

    class FakeReader:
        pages = [TextPage()]
        def __init__(self, path): pass

    async def no_ocr(*args, **kwargs):
        raise AssertionError('a PDF with a text layer needs no OCR')

    async def remote(text, filename):
        return 'saved_jd'

    (tmp_path / 'jd.pdf').write_bytes(b'%PDF fake')
    monkeypatch.setattr(tools.pypdf, 'PdfReader', FakeReader)
    monkeypatch.setattr(pdf_ocr, 'call_mcp_tool', no_ocr)
    monkeypatch.setattr(tools, '_categorize_pdf_text', remote)
    await tools.categorize_save_dir_pdfs(save_dir=tmp_path)
    assert (tmp_path / 'cat-saved_jd-jd.pdf').exists()


async def test_server_down_leaves_the_file_for_the_next_run(scanned_pdf, monkeypatch, caplog):
    async def down(*args, **kwargs):
        raise ConnectionError('All connection attempts failed')

    monkeypatch.setattr(pdf_ocr, 'call_mcp_tool', down)
    monkeypatch.setattr(tools, '_categorize_pdf_text', _no_remote)
    await tools.categorize_save_dir_pdfs(save_dir=scanned_pdf.parent)

    assert scanned_pdf.exists()
    assert not pdf_ocr.PDF_OCR_CACHE_PATH.exists()
    message = next(r.message for r in caplog.records if 'Local OCR categorization failed' in r.message)
    assert 'IMG_1.pdf' in message and pdf_ocr.LLM_LOCAL_MCP_URL in message and 'ConnectionError' in message


async def test_an_uncited_local_category_leaves_the_file_uncategorized(scanned_pdf, monkeypatch, caplog):
    calls = []
    monkeypatch.setattr(pdf_ocr, 'call_mcp_tool', _fake_local_chat(calls, quote='Words not on the page'))
    monkeypatch.setattr(tools, '_categorize_pdf_text', _no_remote)
    await tools.categorize_save_dir_pdfs(save_dir=scanned_pdf.parent)

    assert scanned_pdf.exists()
    assert any('Hallucination rejected (pdf_categorize_local' in r.message for r in caplog.records)


async def test_a_page_with_no_text_even_after_ocr_is_labelled_no_text(scanned_pdf, monkeypatch):
    calls = []
    monkeypatch.setattr(pdf_ocr, 'call_mcp_tool', _fake_local_chat(calls, ocr_text=''))
    monkeypatch.setattr(tools, '_categorize_pdf_text', _no_remote)
    await tools.categorize_save_dir_pdfs(save_dir=scanned_pdf.parent)

    assert (scanned_pdf.parent / 'cat-no_text-IMG_1.pdf').exists()


@pytest.mark.parametrize('reply, fragment', [
    ('not json', 'not JSON'),
    (json.dumps({'ok': False, 'error': 'model not found'}), 'server reported an error'),
    (json.dumps({'message': {'content': 'cut'}, 'done_reason': 'length'}), 'truncated'),
    (json.dumps({'done_reason': 'stop'}), 'no message.content'),
])
def test_malformed_local_replies_raise_with_the_reply(reply, fragment):
    with pytest.raises((ValueError, RuntimeError), match=fragment) as caught:
        pdf_ocr._chat_content(reply, 'PDF OCR of x.pdf page 1/1')
    assert 'x.pdf' in str(caught.value)


@pytest.mark.live
async def test_live_local_ocr_reads_a_rendered_page(tmp_path, monkeypatch):
    '''Against the real :8002 server and MODEL_NAME_LOCAL_OCR; skipped when it is down.'''
    from PIL import Image, ImageDraw, ImageFont

    monkeypatch.setattr(pdf_ocr, 'PDF_OCR_CACHE_PATH', tmp_path / 'pdf_ocr_cache.yaml')
    pdf = tmp_path / 'photo.pdf'  # an image saved as a PDF: no text layer, like a phone scan
    image = Image.new('RGB', (LIVE_PAGE_WIDTH, LIVE_PAGE_HEIGHT), 'white')
    ImageDraw.Draw(image).text((LIVE_TEXT_MARGIN, LIVE_TEXT_MARGIN), 'Senior AI Engineer', fill='black',
                               font=ImageFont.load_default(size=LIVE_FONT_SIZE))
    image.save(pdf, 'PDF')
    try:
        pages = await pdf_ocr.ocr_pdf_pages(pdf)
    except RuntimeError as ex:
        if 'ConnectError' in str(ex) or 'ConnectionError' in str(ex):
            pytest.skip(f'local LLM server is down: {ex}')
        raise
    assert 'Senior AI Engineer' in pages[0]
