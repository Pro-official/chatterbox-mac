"""
Audiobook Generator with Progress Tracking and Resume Support.

Features:
- Generates audio for each text chunk
- Saves progress to progress.json for resume capability
- Organized output folder structure
- Real-time progress updates via generator
"""

import json
import re
import time
from pathlib import Path
from typing import Generator, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import torch
import torchaudio

from book_parser import BookParser, BookData, TextChunk


def split_text_into_tts_chunks(text: str, max_chars: int = 250) -> List[str]:
    """
    Split text into smaller chunks suitable for TTS generation.
    Tries to split on sentence boundaries, then commas, then words.
    """
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            parts = re.split(r'(?<=,)\s+', sentence)
            for part in parts:
                if len(part) > max_chars:
                    words = part.split()
                    word_chunk = ""
                    for word in words:
                        if len(word_chunk + " " + word) <= max_chars:
                            word_chunk += " " + word if word_chunk else word
                        else:
                            if word_chunk:
                                chunks.append(word_chunk.strip())
                            word_chunk = word
                    if word_chunk:
                        chunks.append(word_chunk.strip())
                else:
                    if len(current_chunk + " " + part) <= max_chars:
                        current_chunk += " " + part if current_chunk else part
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part
        else:
            if len(current_chunk + " " + sentence) <= max_chars:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [chunk for chunk in chunks if chunk.strip()]


@dataclass
class ProgressState:
    """Tracks audiobook generation progress with detailed chunk info."""
    total_chunks: int
    completed_chunks: List[dict]  # List of dicts with detailed chunk info
    failed_chunks: List[dict]     # List of dicts with error info
    current_chunk: Optional[int]
    started_at: str
    updated_at: str
    status: str  # 'in_progress', 'completed', 'paused', 'failed'
    total_elapsed_seconds: float = 0.0  # Total generation time

    def to_dict(self) -> dict:
        return asdict(self)

    def get_completed_indices(self) -> List[int]:
        """Get list of completed chunk indices (for backwards compatibility)."""
        return [c['index'] if isinstance(c, dict) else c for c in self.completed_chunks]

    @classmethod
    def from_dict(cls, data: dict) -> 'ProgressState':
        # Handle backwards compatibility - if total_elapsed_seconds not present
        if 'total_elapsed_seconds' not in data:
            data['total_elapsed_seconds'] = 0.0
        return cls(**data)

    @classmethod
    def new(cls, total_chunks: int) -> 'ProgressState':
        now = datetime.now().isoformat()
        return cls(
            total_chunks=total_chunks,
            completed_chunks=[],
            failed_chunks=[],
            current_chunk=None,
            started_at=now,
            updated_at=now,
            status='in_progress',
            total_elapsed_seconds=0.0
        )


@dataclass
class GenerationProgress:
    """Progress update yielded during generation."""
    current_chunk: int
    total_chunks: int
    chunk_title: str
    chunk_text_preview: str  # First 100 chars
    status: str  # 'generating', 'completed', 'failed', 'skipped'
    audio_path: Optional[str]
    elapsed_seconds: float
    estimated_remaining_seconds: Optional[float]
    error_message: Optional[str] = None
    # Sub-chunk progress (for TTS generation within a chapter)
    current_subchunk: int = 0
    total_subchunks: int = 0


class AudiobookGenerator:
    """Generate audiobook from parsed book data."""

    def __init__(
        self,
        model,  # ChatterboxTurboTTS
        output_base_dir: str = "audiobooks",
    ):
        self.model = model
        self.output_base_dir = Path(output_base_dir)
        # Use the model's sample rate instead of hardcoding
        self.sample_rate = model.sr

    def generate(
        self,
        book_file: str,
        reference_audio: str,
        chunk_size: int = 2500,
        resume: bool = True,
        max_chunks: Optional[int] = None,
        max_words: Optional[int] = None,
        selected_chapters: Optional[List[int]] = None,
        edited_texts: Optional[dict] = None
    ) -> Generator[GenerationProgress, None, str]:
        """
        Generate audiobook from book file.

        Args:
            book_file: Path to PDF/EPUB/TXT file
            reference_audio: Path to reference voice audio
            chunk_size: Words per chunk for non-chapter content (default 2500)
            resume: If True, resume from progress.json if exists
            max_chunks: If set, only process this many chunks (for testing)
            max_words: If set, truncate each chunk to this many words (for testing)
            selected_chapters: If set, only process chapters with these indices
            edited_texts: Dict mapping chapter indices to edited text content

        Yields:
            GenerationProgress updates

        Returns:
            Path to output folder
        """
        start_time = time.time()

        # 1. Parse book
        book = BookParser.parse(book_file, chunk_size=chunk_size)
        chunks = book.chunks

        # Apply edited texts if provided
        if edited_texts:
            for chunk in chunks:
                if chunk.index in edited_texts:
                    chunk.text = edited_texts[chunk.index]
                    chunk.word_count = len(chunk.text.split())

        # Filter to selected chapters only
        if selected_chapters is not None:
            chunks = [ch for ch in chunks if ch.index in selected_chapters]

        # Apply test mode limits (after chapter selection)
        if max_chunks is not None:
            chunks = chunks[:max_chunks]

        # Truncate chunk text if max_words is set
        if max_words is not None:
            for chunk in chunks:
                words = chunk.text.split()
                if len(words) > max_words:
                    chunk.text = ' '.join(words[:max_words])
                    chunk.word_count = max_words

        # 2. Create output folder
        folder_name = self._sanitize_filename(book.title)
        output_folder = self.output_base_dir / folder_name
        output_folder.mkdir(parents=True, exist_ok=True)

        # Create chunks subfolder for text files
        chunks_folder = output_folder / "chunks"
        chunks_folder.mkdir(exist_ok=True)

        # 3. Save lean metadata (without full text - text is saved in chunks/ folder)
        metadata_path = output_folder / "metadata.json"
        metadata = {
            'title': book.title,
            'author': book.author,
            'format': book.format,
            'total_words': book.total_words,
            'has_chapters': book.has_chapters,
            'chapters': [
                {'index': ch.index, 'title': ch.title, 'word_count': ch.word_count}
                for ch in chunks
            ],
            'chunk_size': chunk_size,
            'reference_audio': str(reference_audio),
            'created_at': datetime.now().isoformat()
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # 4. Save chunk text files (for reference)
        for chunk in chunks:
            chunk_file = chunks_folder / f"chunk_{chunk.index + 1:03d}.txt"
            with open(chunk_file, 'w') as f:
                f.write(f"# {chunk.title}\n")
                f.write(f"# Words: {chunk.word_count}\n\n")
                f.write(chunk.text)

        # 5. Load or create progress state
        progress_path = output_folder / "progress.json"
        if resume and progress_path.exists():
            with open(progress_path, 'r') as f:
                progress = ProgressState.from_dict(json.load(f))
            # Validate progress matches current book
            if progress.total_chunks != len(chunks):
                # Book changed, start fresh
                progress = ProgressState.new(total_chunks=len(chunks))
        else:
            progress = ProgressState.new(total_chunks=len(chunks))

        # Track timing for estimates
        chunk_times = []

        # 6. Generate audio for each chunk
        for chunk in chunks:
            chunk_idx = chunk.index
            audio_filename = f"audio_{chunk_idx + 1:03d}.wav"
            audio_path = output_folder / audio_filename

            # Skip if already completed (use helper for backwards compatibility)
            if chunk_idx in progress.get_completed_indices():
                yield GenerationProgress(
                    current_chunk=chunk_idx + 1,
                    total_chunks=len(chunks),
                    chunk_title=chunk.title,
                    chunk_text_preview=chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                    status='skipped',
                    audio_path=str(audio_path),
                    elapsed_seconds=time.time() - start_time,
                    estimated_remaining_seconds=None
                )
                continue

            # Update progress state
            progress.current_chunk = chunk_idx
            progress.updated_at = datetime.now().isoformat()
            self._save_progress(progress_path, progress)

            # Calculate time estimate
            elapsed = time.time() - start_time
            if chunk_times:
                avg_time = sum(chunk_times) / len(chunk_times)
                remaining_chunks = len(chunks) - chunk_idx - 1
                remaining = avg_time * remaining_chunks
            else:
                remaining = None

            # Yield "generating" status
            yield GenerationProgress(
                current_chunk=chunk_idx + 1,
                total_chunks=len(chunks),
                chunk_title=chunk.title,
                chunk_text_preview=chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                status='generating',
                audio_path=None,
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=remaining
            )

            chunk_start = time.time()

            try:
                # Split chapter text into small TTS-friendly chunks (~250 chars)
                tts_chunks = split_text_into_tts_chunks(chunk.text, max_chars=250)
                total_subchunks = len(tts_chunks)

                generated_wavs = []
                subchunk_times = []  # Track timing for remaining time calculation
                remaining_chapters = len(chunks) - chunk_idx - 1

                for tts_idx, tts_text in enumerate(tts_chunks):
                    # Generate audio for each small chunk (with timing)
                    subchunk_start = time.time()
                    wav = self.model.generate(tts_text, audio_prompt_path=reference_audio)
                    subchunk_time = time.time() - subchunk_start
                    subchunk_times.append(subchunk_time)

                    # Calculate remaining time based on sub-chunk progress
                    avg_subchunk_time = sum(subchunk_times) / len(subchunk_times)
                    remaining_subchunks_in_chapter = total_subchunks - (tts_idx + 1)
                    # Estimate remaining: current chapter's remaining + future chapters
                    if remaining_chapters > 0:
                        # Assume similar subchunks per chapter for estimation
                        estimated_future_subchunks = remaining_chapters * total_subchunks
                    else:
                        estimated_future_subchunks = 0
                    remaining = avg_subchunk_time * (remaining_subchunks_in_chapter + estimated_future_subchunks)

                    # Yield progress AFTER TTS completes (with updated timing)
                    elapsed = time.time() - start_time
                    yield GenerationProgress(
                        current_chunk=chunk_idx + 1,
                        total_chunks=len(chunks),
                        chunk_title=chunk.title,
                        chunk_text_preview=tts_text[:100] + "..." if len(tts_text) > 100 else tts_text,
                        status='generating',
                        audio_path=None,
                        elapsed_seconds=elapsed,
                        estimated_remaining_seconds=remaining,
                        current_subchunk=tts_idx + 1,
                        total_subchunks=total_subchunks
                    )

                    # Convert to tensor if needed
                    if isinstance(wav, np.ndarray):
                        wav = torch.from_numpy(wav).unsqueeze(0)
                    elif isinstance(wav, torch.Tensor) and wav.dim() == 1:
                        wav = wav.unsqueeze(0)

                    generated_wavs.append(wav)

                # Concatenate all TTS chunks with short silence gaps
                if len(generated_wavs) > 1:
                    silence_samples = int(0.3 * self.sample_rate)  # 0.3s silence
                    first_wav = generated_wavs[0]
                    target_device = first_wav.device
                    target_dtype = first_wav.dtype

                    silence = torch.zeros(1, silence_samples, dtype=target_dtype, device=target_device)

                    final_wav = generated_wavs[0]
                    for wav_chunk in generated_wavs[1:]:
                        wav_chunk = wav_chunk.to(target_device)
                        final_wav = torch.cat([final_wav, silence, wav_chunk], dim=1)
                else:
                    final_wav = generated_wavs[0]

                # Save concatenated audio using torchaudio (same as working TTS)
                torchaudio.save(str(audio_path), final_wav.cpu(), self.sample_rate)

                # Track timing
                chunk_time = time.time() - chunk_start
                chunk_times.append(chunk_time)

                # Calculate audio duration from wav tensor
                audio_duration = final_wav.shape[1] / self.sample_rate

                # Update progress with detailed chunk info
                progress.completed_chunks.append({
                    'index': chunk_idx,
                    'title': chunk.title,
                    'audio_file': audio_filename,
                    'duration_seconds': round(audio_duration, 2),
                    'generation_time_seconds': round(chunk_time, 2),
                    'completed_at': datetime.now().isoformat()
                })
                progress.updated_at = datetime.now().isoformat()
                progress.total_elapsed_seconds = time.time() - start_time
                self._save_progress(progress_path, progress)

                # Recalculate estimate
                elapsed = time.time() - start_time
                avg_time = sum(chunk_times) / len(chunk_times)
                remaining_chunks = len(chunks) - chunk_idx - 1
                remaining = avg_time * remaining_chunks if remaining_chunks > 0 else 0

                # Yield completion
                yield GenerationProgress(
                    current_chunk=chunk_idx + 1,
                    total_chunks=len(chunks),
                    chunk_title=chunk.title,
                    chunk_text_preview=chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                    status='completed',
                    audio_path=str(audio_path),
                    elapsed_seconds=elapsed,
                    estimated_remaining_seconds=remaining
                )

            except Exception as e:
                # Track failure with detailed error info but continue
                progress.failed_chunks.append({
                    'index': chunk_idx,
                    'title': chunk.title,
                    'error': str(e),
                    'failed_at': datetime.now().isoformat()
                })
                progress.updated_at = datetime.now().isoformat()
                progress.total_elapsed_seconds = time.time() - start_time
                self._save_progress(progress_path, progress)

                yield GenerationProgress(
                    current_chunk=chunk_idx + 1,
                    total_chunks=len(chunks),
                    chunk_title=chunk.title,
                    chunk_text_preview=chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                    status='failed',
                    audio_path=None,
                    elapsed_seconds=time.time() - start_time,
                    estimated_remaining_seconds=None,
                    error_message=str(e)
                )

        # 7. Mark as complete
        if len(progress.failed_chunks) == 0:
            progress.status = 'completed'
        else:
            progress.status = 'completed_with_errors'
        progress.current_chunk = None
        progress.updated_at = datetime.now().isoformat()
        progress.total_elapsed_seconds = time.time() - start_time
        self._save_progress(progress_path, progress)

        return str(output_folder)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Convert string to safe filename."""
        safe = re.sub(r'[^\w\s-]', '', name)
        safe = re.sub(r'[-\s]+', '_', safe)
        return safe[:100]

    @staticmethod
    def _save_progress(path: Path, progress: ProgressState):
        """Save progress state to JSON file."""
        with open(path, 'w') as f:
            json.dump(progress.to_dict(), f, indent=2)

    @classmethod
    def get_progress(cls, output_folder: str) -> Optional[ProgressState]:
        """Load progress state from an existing audiobook folder."""
        progress_path = Path(output_folder) / "progress.json"
        if progress_path.exists():
            with open(progress_path, 'r') as f:
                return ProgressState.from_dict(json.load(f))
        return None
