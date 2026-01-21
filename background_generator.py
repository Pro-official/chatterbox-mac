"""
Background Generation Manager for Audiobook Generation.

Runs audiobook generation in a background thread, decoupled from
Gradio UI. This ensures generation continues even if the browser
disconnects (monitor off, peripherals unplugged, etc.).

The UI polls progress.json for status updates instead of relying
on generator yields through WebSocket.
"""

import json
import threading
import uuid
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime

from audiobook_generator import AudiobookGenerator, ProgressState
from book_parser import BookParser


@dataclass
class GenerationJob:
    """Represents a background generation job."""
    job_id: str
    output_folder: str
    thread: threading.Thread
    started_at: str
    book_title: str
    status: str  # 'running', 'completed', 'failed'


class BackgroundGenerationManager:
    """
    Manages background audiobook generation tasks.

    Uses singleton pattern to track all running generations.
    Generation runs in a background thread and writes progress to progress.json.
    UI polls progress.json for status updates.
    """

    _instance: Optional['BackgroundGenerationManager'] = None
    _lock = threading.Lock()

    def __init__(self):
        self._jobs: Dict[str, GenerationJob] = {}
        self._folder_to_job: Dict[str, str] = {}  # Map output folder -> job_id

    @classmethod
    def get_instance(cls) -> 'BackgroundGenerationManager':
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def start_generation(
        self,
        model,
        book_file: str,
        reference_audio: str,
        chunk_size: int = 2500,
        selected_chapters: Optional[List[int]] = None,
        edited_texts: Optional[dict] = None,
        max_chunks: Optional[int] = None,
        max_words: Optional[int] = None,
        output_base_dir: str = "audiobooks"
    ) -> tuple[str, str]:
        """
        Start background generation.

        Args:
            model: ChatterboxTurboTTS model instance
            book_file: Path to book file
            reference_audio: Path to reference voice audio
            chunk_size: Words per chunk for non-chapter content
            selected_chapters: List of chapter indices to convert (None = all)
            edited_texts: Dict mapping chapter indices to edited text content
            max_chunks: Max chunks to generate (for test mode)
            max_words: Max words per chunk (for test mode)
            output_base_dir: Base directory for output

        Returns:
            Tuple of (job_id, output_folder)
        """
        # Parse book to get title for output folder
        book = BookParser.parse(book_file, chunk_size=chunk_size)
        folder_name = AudiobookGenerator._sanitize_filename(book.title)
        output_folder = str(Path(output_base_dir) / folder_name)

        # Check if already generating this book
        if output_folder in self._folder_to_job:
            existing_job_id = self._folder_to_job[output_folder]
            existing_job = self._jobs.get(existing_job_id)
            if existing_job and existing_job.thread.is_alive():
                # Already running, return existing job
                return existing_job_id, output_folder

        # Create generator
        generator = AudiobookGenerator(model, output_base_dir=output_base_dir)

        # Generate job ID
        job_id = str(uuid.uuid4())[:8]

        # Create and start thread
        thread = threading.Thread(
            target=self._run_generation,
            args=(
                job_id,
                generator,
                book_file,
                reference_audio,
                chunk_size,
                selected_chapters,
                edited_texts,
                max_chunks,
                max_words,
                output_folder
            ),
            daemon=True,
            name=f"audiobook-gen-{job_id}"
        )

        # Create job record
        job = GenerationJob(
            job_id=job_id,
            output_folder=output_folder,
            thread=thread,
            started_at=datetime.now().isoformat(),
            book_title=book.title,
            status='running'
        )

        self._jobs[job_id] = job
        self._folder_to_job[output_folder] = job_id

        # Start the thread
        thread.start()

        return job_id, output_folder

    def _run_generation(
        self,
        job_id: str,
        generator: AudiobookGenerator,
        book_file: str,
        reference_audio: str,
        chunk_size: int,
        selected_chapters: Optional[List[int]],
        edited_texts: Optional[dict],
        max_chunks: Optional[int],
        max_words: Optional[int],
        output_folder: str
    ):
        """
        Runs in background thread.

        Consumes the generator, which saves progress to progress.json.
        """
        try:
            # Consume the generator - progress is saved to progress.json automatically
            for _ in generator.generate(
                book_file=book_file,
                reference_audio=reference_audio,
                chunk_size=chunk_size,
                resume=True,
                max_chunks=max_chunks,
                max_words=max_words,
                selected_chapters=selected_chapters,
                edited_texts=edited_texts
            ):
                # Just consume - progress is written by generator
                pass

            # Mark job as completed
            if job_id in self._jobs:
                self._jobs[job_id].status = 'completed'

        except Exception as e:
            # Update progress.json with error status
            progress_path = Path(output_folder) / "progress.json"
            if progress_path.exists():
                try:
                    with open(progress_path, 'r') as f:
                        progress_data = json.load(f)

                    progress_data['status'] = 'failed'
                    progress_data['error'] = str(e)
                    progress_data['updated_at'] = datetime.now().isoformat()

                    with open(progress_path, 'w') as f:
                        json.dump(progress_data, f, indent=2)
                except Exception:
                    pass  # Best effort

            # Mark job as failed
            if job_id in self._jobs:
                self._jobs[job_id].status = 'failed'

    def get_status(self, output_folder: str) -> Optional[ProgressState]:
        """
        Read current status from progress.json.

        Args:
            output_folder: Path to the audiobook output folder

        Returns:
            ProgressState or None if not found
        """
        return AudiobookGenerator.get_progress(output_folder)

    def get_job_by_folder(self, output_folder: str) -> Optional[GenerationJob]:
        """Get job info by output folder."""
        job_id = self._folder_to_job.get(output_folder)
        if job_id:
            return self._jobs.get(job_id)
        return None

    def is_running(self, output_folder: str) -> bool:
        """Check if generation is currently running for this folder."""
        job = self.get_job_by_folder(output_folder)
        if job:
            return job.thread.is_alive()
        return False

    def stop_generation(self, output_folder: str) -> bool:
        """
        Request to stop generation.

        Note: This sets a flag but doesn't forcibly kill the thread.
        The generator would need to check this flag periodically.
        For now, we just mark status as 'paused' in progress.json.

        Returns:
            True if request was processed, False if no job found
        """
        progress_path = Path(output_folder) / "progress.json"
        if progress_path.exists():
            try:
                with open(progress_path, 'r') as f:
                    progress_data = json.load(f)

                # Only update if currently in_progress
                if progress_data.get('status') == 'in_progress':
                    progress_data['status'] = 'paused'
                    progress_data['updated_at'] = datetime.now().isoformat()

                    with open(progress_path, 'w') as f:
                        json.dump(progress_data, f, indent=2)
                    return True
            except Exception:
                pass
        return False

    def list_active_jobs(self) -> List[GenerationJob]:
        """Get list of currently running jobs."""
        return [job for job in self._jobs.values() if job.thread.is_alive()]

    def cleanup_completed_jobs(self):
        """Remove completed/failed jobs from tracking."""
        to_remove = []
        for job_id, job in self._jobs.items():
            if not job.thread.is_alive():
                to_remove.append(job_id)

        for job_id in to_remove:
            job = self._jobs.pop(job_id, None)
            if job:
                self._folder_to_job.pop(job.output_folder, None)
