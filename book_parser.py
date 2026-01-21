"""
Book Parser Module for Audiobook Generation.

Supports: PDF, EPUB, TXT
Features:
- Chapter-aware parsing for EPUB (keeps chapters intact)
- Word-based chunking for non-chapter content
- Metadata extraction (title, author, chapters)

Logic:
- EPUB with chapters: Each chapter = 1 chunk (regardless of word count)
- EPUB without chapters / PDF / TXT: Use 2500 word chunking
"""

import re
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict

# PDF parsing
import fitz  # PyMuPDF

# EPUB parsing
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup


@dataclass
class TextChunk:
    """A chunk of text for audio generation."""
    index: int
    text: str
    word_count: int
    title: str  # Chapter title or "Part X"
    section_type: str = "body"  # "front", "body", or "back"


@dataclass
class BookData:
    """Parsed book data."""
    title: str
    author: str
    source_file: str
    format: str  # 'pdf', 'epub', 'txt'
    chunks: List[TextChunk]
    total_words: int
    has_chapters: bool

    def to_dict(self) -> dict:
        return {
            'title': self.title,
            'author': self.author,
            'source_file': self.source_file,
            'format': self.format,
            'chunks': [asdict(ch) for ch in self.chunks],
            'total_words': self.total_words,
            'has_chapters': self.has_chapters
        }


class BookParser:
    """Parse PDF, EPUB, and TXT files into structured book data."""

    SUPPORTED_FORMATS = {'.pdf', '.epub', '.txt'}
    DEFAULT_CHUNK_SIZE = 2500  # words

    @classmethod
    def parse(cls, file_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> BookData:
        """
        Parse a book file and extract structured data.

        Args:
            file_path: Path to PDF, EPUB, or TXT file
            chunk_size: Words per chunk for non-chapter content (default 2500)

        Returns:
            BookData with title, author, chunks

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in cls.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}. Supported: {cls.SUPPORTED_FORMATS}")

        if ext == '.epub':
            return cls._parse_epub(path, chunk_size)
        elif ext == '.pdf':
            return cls._parse_pdf(path, chunk_size)
        else:  # .txt
            return cls._parse_txt(path, chunk_size)

    @classmethod
    def _parse_epub(cls, path: Path, chunk_size: int) -> BookData:
        """
        Parse EPUB file.

        If chapters found: Keep each chapter intact (1 audio per chapter)
        If no chapters: Use word-based chunking
        """
        book = epub.read_epub(str(path))

        # Extract metadata
        title = book.get_metadata('DC', 'title')
        title = title[0][0] if title else path.stem

        author = book.get_metadata('DC', 'creator')
        author = author[0][0] if author else 'Unknown'

        # Extract chapters in SPINE ORDER (correct reading order)
        chapters = []
        for idref, linear in book.spine:
            item = book.get_item_with_id(idref)
            if not item or item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue

            # Parse HTML content
            soup = BeautifulSoup(item.get_content(), 'html.parser')

            # Try to find chapter title from headings
            chapter_title = None
            for tag in ['h1', 'h2', 'h3', 'title']:
                header = soup.find(tag)
                if header:
                    chapter_title = header.get_text(strip=True)
                    break

            # Detect section type and fallback title from epub:type
            section_type = "body"
            for elem in [soup.find('body'), soup.find('section')]:
                if elem and elem.get('epub:type'):
                    epub_type = elem.get('epub:type')
                    # Detect front matter
                    if 'frontmatter' in epub_type or epub_type in [
                        'titlepage', 'imprint', 'preface', 'introduction',
                        'foreword', 'dedication', 'epigraph', 'prologue', 'preamble'
                    ]:
                        section_type = "front"
                    # Detect back matter
                    elif 'backmatter' in epub_type or epub_type in [
                        'colophon', 'afterword', 'endnotes', 'appendix',
                        'epilogue', 'conclusion', 'uncopyright', 'copyright-page'
                    ]:
                        section_type = "back"
                    # Use epub:type as fallback title if no heading found
                    if not chapter_title:
                        chapter_title = epub_type.replace('-', ' ').title()
                    break

            # Extract text content
            text = soup.get_text(separator=' ', strip=True)
            text = cls._clean_text(text)

            # Lower threshold: 10 words (was 50) to include intro/preface/short sections
            if text and len(text.split()) > 10:
                word_count = len(text.split())
                chapters.append({
                    'title': chapter_title,
                    'text': text,
                    'word_count': word_count,
                    'section_type': section_type
                })

        # Determine if we have real chapters
        # Consider it "has chapters" if we have multiple sections with titles
        titled_chapters = [ch for ch in chapters if ch['title']]
        has_chapters = len(titled_chapters) >= 2

        if has_chapters:
            # Keep chapters intact
            chunks = []
            for i, ch in enumerate(chapters):
                chunk_title = ch['title'] or f"Section {i + 1}"
                chunks.append(TextChunk(
                    index=i,
                    text=ch['text'],
                    word_count=ch['word_count'],
                    title=chunk_title,
                    section_type=ch.get('section_type', 'body')
                ))
        else:
            # Merge all text and chunk by word count
            full_text = ' '.join(ch['text'] for ch in chapters)
            chunks = cls._chunk_by_words(full_text, chunk_size)

        total_words = sum(ch.word_count for ch in chunks)

        return BookData(
            title=title,
            author=author,
            source_file=str(path),
            format='epub',
            chunks=chunks,
            total_words=total_words,
            has_chapters=has_chapters
        )

    @classmethod
    def _parse_pdf(cls, path: Path, chunk_size: int) -> BookData:
        """Parse PDF file using word-based chunking."""
        doc = fitz.open(str(path))

        # Extract all text
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"

        full_text = cls._clean_text(full_text)
        chunks = cls._chunk_by_words(full_text, chunk_size)
        total_words = sum(ch.word_count for ch in chunks)

        return BookData(
            title=path.stem,
            author='Unknown',
            source_file=str(path),
            format='pdf',
            chunks=chunks,
            total_words=total_words,
            has_chapters=False
        )

    @classmethod
    def _parse_txt(cls, path: Path, chunk_size: int) -> BookData:
        """Parse TXT file using word-based chunking."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        text = cls._clean_text(text)
        chunks = cls._chunk_by_words(text, chunk_size)
        total_words = sum(ch.word_count for ch in chunks)

        return BookData(
            title=path.stem,
            author='Unknown',
            source_file=str(path),
            format='txt',
            chunks=chunks,
            total_words=total_words,
            has_chapters=False
        )

    @classmethod
    def _chunk_by_words(cls, text: str, max_words: int) -> List[TextChunk]:
        """
        Split text into word-based chunks at sentence boundaries.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # If adding this sentence exceeds limit, save current chunk
            if current_word_count + sentence_words > max_words and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(TextChunk(
                    index=len(chunks),
                    text=chunk_text,
                    word_count=current_word_count,
                    title=f"Part {len(chunks) + 1}"
                ))
                current_chunk = []
                current_word_count = 0

            current_chunk.append(sentence)
            current_word_count += sentence_words

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(TextChunk(
                index=len(chunks),
                text=chunk_text,
                word_count=current_word_count,
                title=f"Part {len(chunks) + 1}"
            ))

        return chunks

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text - preserve all characters, only fix whitespace."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Fix spacing around punctuation (remove space before punctuation)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text.strip()


def test_parser():
    """Test the parser with sample EPUB."""
    import sys

    epub_path = "/Users/promise/Desktop/h-rider-haggard_allan-quatermain-stories.epub"

    print(f"Parsing: {epub_path}")
    book = BookParser.parse(epub_path)

    print(f"\nTitle: {book.title}")
    print(f"Author: {book.author}")
    print(f"Format: {book.format}")
    print(f"Has chapters: {book.has_chapters}")
    print(f"Total words: {book.total_words:,}")
    print(f"Total chunks: {len(book.chunks)}")

    print("\nChunks:")
    for chunk in book.chunks[:10]:  # Show first 10
        print(f"  {chunk.index + 1}. {chunk.title} ({chunk.word_count:,} words)")

    if len(book.chunks) > 10:
        print(f"  ... and {len(book.chunks) - 10} more")


if __name__ == "__main__":
    test_parser()
