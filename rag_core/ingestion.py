"""
rag_core.ingestion — load the guideline PDF and split it into section-aware,
cross-page chunks.

Lifted from the notebook's Sections 2-3. Copy new chunking ideas back here
once they've been tried in the notebook; `build_index.py` is what actually
re-runs this against the persisted index.
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def _section_for_page(page_number: int) -> str:
    section = config.SECTION_STARTS[0][1]
    for start_page, name in config.SECTION_STARTS:
        if page_number >= start_page:
            section = name
        else:
            break
    return section


def _load_pages(pdf_path: Path):
    if not pdf_path.exists():
        matches = list(Path(".").glob("*prostate*cancer*.pdf"))
        if matches:
            pdf_path = matches[0]

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Guideline PDF not found at {pdf_path}. Set GUIDELINE_PDF_PATH in "
            "your .env, or place the PDF at the project root."
        )

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    for p in pages:
        p.metadata["document_id"] = config.DOC_ID
        p.metadata["document_title"] = config.DOC_TITLE
        p.metadata["version"] = config.DOC_VERSION
        p.metadata["page_number"] = p.metadata.get("page", 0) + 1

    return pages


def _pages_by_section(pages):
    """Group (page_number, page_text) tuples by guideline section, in page order."""
    grouped = {}
    for page in pages:
        page_number = int(page.metadata["page_number"])
        section = _section_for_page(page_number)
        grouped.setdefault(section, []).append((page_number, page.page_content))
    for section in grouped:
        grouped[section].sort(key=lambda t: t[0])
    return grouped


def load_and_chunk_pdf(
    pdf_path: Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Load the guideline PDF and split it into section-aware, cross-page
    chunks. Each chunk's metadata carries document_id, section, page_start,
    page_end, pages (comma-joined), and a stable chunk_id.
    """
    pdf_path = Path(pdf_path or config.PDF_PATH)
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    pages = _load_pages(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
        add_start_index=True,
    )

    all_chunks: list[Document] = []
    grouped = _pages_by_section(pages)

    for section_name, page_list in grouped.items():
        combined_text = ""
        page_offsets = []
        for page_number, text in page_list:
            start = len(combined_text)
            combined_text += text + "\n"
            end = len(combined_text)
            page_offsets.append((start, end, page_number))

        section_doc = Document(
            page_content=combined_text,
            metadata={"section": section_name, "document_id": config.DOC_ID},
        )

        pieces = splitter.split_documents([section_doc])

        for piece in pieces:
            start_idx = piece.metadata.get("start_index", 0)
            end_idx = start_idx + len(piece.page_content)

            spanned_pages = sorted(
                {pn for (s, e, pn) in page_offsets if s < end_idx and e > start_idx}
            )
            if not spanned_pages:
                nearest = min(page_offsets, key=lambda t: abs(t[0] - start_idx))
                spanned_pages = [nearest[2]]

            piece.metadata["section"] = section_name
            piece.metadata["document_id"] = config.DOC_ID
            piece.metadata["page_start"] = spanned_pages[0]
            piece.metadata["page_end"] = spanned_pages[-1]
            piece.metadata["page_number"] = spanned_pages[0]
            piece.metadata["pages"] = ",".join(str(p) for p in spanned_pages)
            piece.metadata.pop("start_index", None)
            all_chunks.append(piece)

    for i, chunk in enumerate(all_chunks, 1):
        chunk.metadata["chunk_id"] = f"{config.DOC_ID}-CH-{i:04d}"

    return all_chunks
