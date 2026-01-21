#!/usr/bin/env python3
"""
Chatterbox-TTS Gradio App with Model Selection
Supports both Standard (500M) and Turbo (350M) models.
Default: Turbo for faster generation.

Includes Audiobook Generator for PDF/EPUB/TXT files.
"""

import random
import numpy as np
import torch
import gradio as gr
import logging
from pathlib import Path
import sys
import re
from typing import List, Optional, Generator
import time

from book_parser import BookParser
from audiobook_generator import AudiobookGenerator, GenerationProgress, ProgressState
from background_generator import BackgroundGenerationManager
from config import DEV_MODE, get_test_mode_limits

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Monkey patch torch.load to handle device mapping
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, **kwargs):
    if map_location is None:
        map_location = 'cpu'
    return original_torch_load(f, map_location=map_location, **kwargs)

torch.load = patched_torch_load
if 'torch' in sys.modules:
    sys.modules['torch'].load = patched_torch_load

logger.info("✅ Applied torch.load device mapping patch")

# Fix Perth watermarker for macOS
import perth
perth.PerthImplicitWatermarker = perth.DummyWatermarker
logger.info("✅ Applied Perth watermarker fix for macOS")

# Device detection
if torch.cuda.is_available():
    DEVICE = "cuda"
    logger.info("🚀 Running on CUDA GPU")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    logger.info("🍎 Apple Silicon detected - using MPS GPU")
else:
    DEVICE = "cpu"
    logger.info("🚀 Running on CPU")

print(f"🚀 Running on device: {DEVICE}")

# Model cache
MODELS = {
    "turbo": None,
    "standard": None
}
CURRENT_MODEL_TYPE = "turbo"  # Default


def load_turbo_model():
    """Load Turbo model (350M, faster)."""
    global MODELS
    if MODELS["turbo"] is None:
        logger.info("Loading Turbo model...")
        from huggingface_hub import snapshot_download
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        local_path = snapshot_download(
            repo_id='ResembleAI/chatterbox-turbo',
            token=None,
            allow_patterns=['*.safetensors', '*.json', '*.txt', '*.pt', '*.model']
        )
        MODELS["turbo"] = ChatterboxTurboTTS.from_local(local_path, DEVICE)
        logger.info("✅ Turbo model loaded!")
    return MODELS["turbo"]


def load_standard_model():
    """Load Standard model (500M, more features)."""
    global MODELS
    if MODELS["standard"] is None:
        logger.info("Loading Standard model...")
        from chatterbox.tts import ChatterboxTTS

        model = ChatterboxTTS.from_pretrained(device="cpu")

        # Move to device
        if DEVICE != "cpu":
            if hasattr(model, 't3') and model.t3 is not None:
                model.t3 = model.t3.to(DEVICE)
            if hasattr(model, 's3gen') and model.s3gen is not None:
                model.s3gen = model.s3gen.to(DEVICE)
            if hasattr(model, 've') and model.ve is not None:
                model.ve = model.ve.to(DEVICE)
            model.device = DEVICE

        MODELS["standard"] = model
        logger.info("✅ Standard model loaded!")
    return MODELS["standard"]


def get_model(model_type: str):
    """Get model by type."""
    if model_type == "turbo":
        return load_turbo_model()
    else:
        return load_standard_model()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def split_text_into_chunks(text: str, max_chars: int = 250) -> List[str]:
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


def generate_tts_audio(
    text_input: str,
    audio_prompt_path_input: str,
    model_type: str,
    exaggeration_input: float,
    temperature_input: float,
    seed_num_input: int,
    cfgw_input: float,
    chunk_size: int = 250
) -> tuple[int, np.ndarray]:
    """Generate TTS audio with model selection."""
    try:
        # Get the selected model
        current_model = get_model(model_type)

        if current_model is None:
            raise RuntimeError("TTS model is not loaded.")

        if seed_num_input != 0:
            set_seed(int(seed_num_input))

        # Split text into chunks
        text_chunks = split_text_into_chunks(text_input, chunk_size)
        logger.info(f"Processing {len(text_chunks)} chunk(s) with {model_type} model")

        generated_wavs = []
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        for i, chunk in enumerate(text_chunks):
            logger.info(f"Generating chunk {i+1}/{len(text_chunks)}: '{chunk[:50]}...'")

            # Generate audio - Turbo has fewer options
            if model_type == "turbo":
                if audio_prompt_path_input:
                    wav = current_model.generate(chunk, audio_prompt_path=audio_prompt_path_input)
                else:
                    wav = current_model.generate(chunk)
            else:
                # Standard model with full options
                wav = current_model.generate(
                    chunk,
                    audio_prompt_path=audio_prompt_path_input,
                    exaggeration=exaggeration_input,
                    temperature=temperature_input,
                    cfg_weight=cfgw_input,
                )

            generated_wavs.append(wav)

            if len(text_chunks) > 1:
                chunk_path = output_dir / f"chunk_{i+1}_{random.randint(1000, 9999)}.wav"
                import torchaudio
                torchaudio.save(str(chunk_path), wav.cpu(), current_model.sr)

        # Concatenate chunks
        if len(generated_wavs) > 1:
            silence_samples = int(0.3 * current_model.sr)
            first_wav = generated_wavs[0]
            target_device = first_wav.device
            target_dtype = first_wav.dtype

            silence = torch.zeros(1, silence_samples, dtype=target_dtype)
            silence = silence.to(target_device)

            final_wav = generated_wavs[0]
            for wav_chunk in generated_wavs[1:]:
                final_wav = torch.cat([final_wav, silence, wav_chunk], dim=1)
        else:
            final_wav = generated_wavs[0]

        logger.info("✅ Audio generation complete.")

        output_path = output_dir / f"generated_{model_type}_{random.randint(1000, 9999)}.wav"
        import torchaudio
        torchaudio.save(str(output_path), final_wav.cpu(), current_model.sr)
        logger.info(f"Saved to: {output_path}")

        return (current_model.sr, final_wav.squeeze(0).cpu().numpy())

    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        raise gr.Error(f"Generation failed: {str(e)}")


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    if seconds is None:
        return "--:--"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def preview_book(book_file):
    """Parse book and show preview info + chapter selector/dropdown choices."""
    if book_file is None:
        return (
            "Upload a book file to see preview.",
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=None),
            {}  # Reset edited texts
        )

    try:
        book = BookParser.parse(book_file.name)
        preview = f"## 📚 {book.title}\n"
        preview += f"**Author:** {book.author}\n"
        preview += f"**Format:** {book.format.upper()}\n"
        preview += f"**Total Words:** {book.total_words:,}\n"
        preview += f"**Chapters/Parts:** {len(book.chunks)}\n\n"

        if book.has_chapters:
            preview += "### Chapters:\n"
        else:
            preview += "### Parts (word-based chunks):\n"

        for chunk in book.chunks[:15]:  # Show first 15
            section_label = f"[{chunk.section_type.upper()}] " if chunk.section_type != "body" else ""
            preview += f"- {section_label}**{chunk.title}** ({chunk.word_count:,} words)\n"

        if len(book.chunks) > 15:
            preview += f"\n*...and {len(book.chunks) - 15} more*\n"

        # Build chapter choices for CheckboxGroup (show section type for non-body)
        chapter_choices = []
        for ch in book.chunks:
            if ch.section_type != "body":
                label = f"[{ch.section_type.upper()}] {ch.title} ({ch.word_count:,} words)"
            else:
                label = f"{ch.title} ({ch.word_count:,} words)"
            chapter_choices.append((label, ch.index))

        # Only select body chapters by default (not front/back matter)
        body_indices = [ch.index for ch in book.chunks if ch.section_type == "body"]

        # Build dropdown choices (same format)
        dropdown_choices = [(f"{ch.title}", ch.index) for ch in book.chunks]

        return (
            preview,
            gr.update(choices=chapter_choices, value=body_indices),
            gr.update(choices=dropdown_choices, value=None),
            {}  # Reset edited texts for new book
        )
    except Exception as e:
        return (
            f"❌ Error parsing book: {str(e)}",
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=None),
            {}
        )


def toggle_select_all(select_all, book_file):
    """Toggle all chapters selected/unselected."""
    if book_file is None:
        return gr.update(value=[])

    if select_all:
        # Select all chapters
        book = BookParser.parse(book_file.name)
        all_indices = [ch.index for ch in book.chunks]
        return gr.update(value=all_indices)
    else:
        # Unselect all
        return gr.update(value=[])


def load_chapter_text(book_file, chapter_idx, edited_texts):
    """Load chapter text from book or edited_texts cache."""
    if book_file is None or chapter_idx is None:
        return ""

    # Check if we have edited text for this chapter
    if edited_texts and chapter_idx in edited_texts:
        return edited_texts[chapter_idx]

    try:
        book = BookParser.parse(book_file.name)
        for chunk in book.chunks:
            if chunk.index == chapter_idx:
                return chunk.text
        return "Chapter not found."
    except Exception as e:
        return f"Error loading chapter: {str(e)}"


def save_chapter_text(chapter_idx, text_content, edited_texts):
    """Save edited chapter text to state."""
    if chapter_idx is None:
        return edited_texts, "⚠️ No chapter selected"

    # Update the edited texts dict
    new_texts = edited_texts.copy() if edited_texts else {}
    new_texts[chapter_idx] = text_content
    return new_texts, f"✅ Saved changes to chapter {chapter_idx + 1}"


# Global background generation manager
background_manager = BackgroundGenerationManager.get_instance()


def build_progress_html_from_state(progress: ProgressState, output_folder: str) -> str:
    """Build progress HTML from ProgressState (read from progress.json)."""
    # Calculate progress percentage
    completed_count = len(progress.completed_chunks)
    total = progress.total_chunks
    pct = (completed_count / total) * 100 if total > 0 else 0

    # Get current chapter info
    current_chunk = progress.current_chunk
    if current_chunk is not None:
        current_display = current_chunk + 1  # 1-indexed for display
    else:
        current_display = completed_count

    # Status display
    status = progress.status
    status_emoji = {
        'in_progress': '⏳',
        'completed': '✅',
        'completed_with_errors': '⚠️',
        'failed': '❌',
        'paused': '⏸️'
    }.get(status, '•')

    status_color = {
        'in_progress': '#ffaa00',
        'completed': '#00ff88',
        'completed_with_errors': '#ffaa00',
        'failed': '#ff4444',
        'paused': '#888888'
    }.get(status, '#ffffff')

    # Time info
    elapsed_str = format_time(progress.total_elapsed_seconds)

    # Estimate remaining time based on average chunk time
    remaining_str = "--:--"
    if completed_count > 0 and progress.total_elapsed_seconds > 0:
        avg_time_per_chunk = progress.total_elapsed_seconds / completed_count
        remaining_chunks = total - completed_count
        remaining_seconds = avg_time_per_chunk * remaining_chunks
        remaining_str = format_time(remaining_seconds)

    # Get current chapter title from the last completed chunk or metadata
    current_title = "Processing..."
    if progress.completed_chunks:
        last_completed = progress.completed_chunks[-1]
        if isinstance(last_completed, dict):
            current_title = last_completed.get('title', 'Chapter')

    # Build HTML
    html = f"""
    <div style="padding: 20px; background: #1a1a2e; border-radius: 12px; margin: 10px 0; border: 1px solid #333;">
        <h3 style="margin: 0 0 15px 0; color: #00d4ff;">📊 Generation Progress</h3>

        <!-- Background mode indicator -->
        <div style="background: #1e3a5f; padding: 8px 12px; border-radius: 6px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
            <span style="color: #00d4ff;">🔄</span>
            <span style="color: #88ccff; font-size: 12px;">Running in background - safe to close browser</span>
        </div>

        <!-- Main Progress Bar -->
        <div style="background: #333; border-radius: 8px; height: 28px; overflow: hidden; margin-bottom: 15px;">
            <div style="background: linear-gradient(90deg, #00d4ff, #00ff88); height: 100%; width: {pct:.1f}%; transition: width 0.3s; display: flex; align-items: center; justify-content: center;">
                <span style="color: #000; font-weight: bold; font-size: 12px;">{pct:.1f}%</span>
            </div>
        </div>

        <!-- Status Grid -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;">
            <div style="background: #252540; padding: 12px; border-radius: 8px;">
                <div style="color: #888; font-size: 11px; text-transform: uppercase;">Chapters</div>
                <div style="color: #fff; font-size: 18px; font-weight: bold;">{completed_count} / {total}</div>
            </div>
            <div style="background: #252540; padding: 12px; border-radius: 8px;">
                <div style="color: #888; font-size: 11px; text-transform: uppercase;">Status</div>
                <div style="color: {status_color}; font-size: 18px; font-weight: bold;">{status_emoji} {status.upper().replace('_', ' ')}</div>
            </div>
        </div>

        <!-- Time Info -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;">
            <div style="background: #252540; padding: 12px; border-radius: 8px;">
                <div style="color: #888; font-size: 11px; text-transform: uppercase;">⏱️ Elapsed</div>
                <div style="color: #fff; font-size: 16px;">{elapsed_str}</div>
            </div>
            <div style="background: #252540; padding: 12px; border-radius: 8px;">
                <div style="color: #888; font-size: 11px; text-transform: uppercase;">⏳ Remaining</div>
                <div style="color: #fff; font-size: 16px;">~{remaining_str}</div>
            </div>
        </div>

        <!-- Current/Last Chapter -->
        <div style="background: #252540; padding: 12px; border-radius: 8px;">
            <div style="color: #888; font-size: 11px; text-transform: uppercase; margin-bottom: 5px;">📖 {'Last Completed' if status != 'in_progress' else 'Currently Processing'}</div>
            <div style="color: #00d4ff; font-size: 14px; font-weight: bold;">{current_title}</div>
        </div>
    </div>
    """
    return html


def build_audio_list_html_from_state(progress: ProgressState, output_folder: str) -> str:
    """Build audio file list HTML from ProgressState."""
    completed = progress.completed_chunks
    count = len(completed)

    html = f"""
    <div style="background: #1a1a2e; border-radius: 12px; padding: 15px; border: 1px solid #333;">
        <h4 style="color: #00d4ff; margin: 0 0 10px 0;">🎵 Generated Audio Files ({count})</h4>
        <div style="max-height: 300px; overflow-y: auto;">
    """

    # Show last 15 files
    display_chunks = completed[-15:] if len(completed) > 15 else completed

    if len(completed) > 15:
        html += f"<p style='color: #888; font-size: 12px;'><em>Showing last 15 of {count} files...</em></p>"

    for chunk in display_chunks:
        if isinstance(chunk, dict):
            idx = chunk.get('index', 0) + 1
            title = chunk.get('title', f'Chapter {idx}')
            audio_file = chunk.get('audio_file', f'audio_{idx:03d}.wav')
            duration = chunk.get('duration_seconds', 0)
            duration_str = format_time(duration) if duration else ""
        else:
            # Backwards compatibility
            idx = chunk + 1 if isinstance(chunk, int) else 0
            title = f'Chapter {idx}'
            audio_file = f'audio_{idx:03d}.wav'
            duration_str = ""

        html += f"""
        <div style="padding: 10px; margin: 5px 0; background: #252540; border-radius: 6px; border-left: 3px solid #00ff88;">
            <span style="color: #00ff88;">✅</span>
            <span style="color: #fff; font-weight: bold;">{audio_file}</span>
            <span style="color: #888;"> - {title}</span>
            {f'<span style="color: #666; font-size: 11px;"> ({duration_str})</span>' if duration_str else ''}
        </div>
        """

    html += "</div></div>"
    return html


def build_completion_html(progress: ProgressState, output_folder: str) -> str:
    """Build completion status HTML."""
    completed_count = len(progress.completed_chunks)
    failed_count = len(progress.failed_chunks)
    elapsed_str = format_time(progress.total_elapsed_seconds)

    # Calculate total audio duration
    total_duration = 0
    for chunk in progress.completed_chunks:
        if isinstance(chunk, dict):
            total_duration += chunk.get('duration_seconds', 0)
    duration_str = format_time(total_duration)

    status_color = '#00ff88' if progress.status == 'completed' else '#ffaa00'
    status_text = 'COMPLETE' if progress.status == 'completed' else 'COMPLETE WITH ERRORS'

    html = f"""
    <div style="padding: 20px; background: #1a1a2e; border-radius: 12px; margin: 10px 0; border: 2px solid {status_color};">
        <h3 style="margin: 0 0 15px 0; color: {status_color};">✅ Audiobook Generation {status_text}!</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px; margin-bottom: 15px;">
            <div style="background: #252540; padding: 12px; border-radius: 8px;">
                <div style="color: #888; font-size: 11px; text-transform: uppercase;">Files</div>
                <div style="color: #00ff88; font-size: 20px; font-weight: bold;">{completed_count}</div>
            </div>
            <div style="background: #252540; padding: 12px; border-radius: 8px;">
                <div style="color: #888; font-size: 11px; text-transform: uppercase;">Duration</div>
                <div style="color: #00d4ff; font-size: 20px; font-weight: bold;">{duration_str}</div>
            </div>
            <div style="background: #252540; padding: 12px; border-radius: 8px;">
                <div style="color: #888; font-size: 11px; text-transform: uppercase;">Gen Time</div>
                <div style="color: #fff; font-size: 20px; font-weight: bold;">{elapsed_str}</div>
            </div>
            <div style="background: #252540; padding: 12px; border-radius: 8px;">
                <div style="color: #888; font-size: 11px; text-transform: uppercase;">Failed</div>
                <div style="color: {'#ff4444' if failed_count > 0 else '#00ff88'}; font-size: 20px; font-weight: bold;">{failed_count}</div>
            </div>
        </div>
        <div style="background: #252540; padding: 12px; border-radius: 8px;">
            <div style="color: #888; font-size: 11px; text-transform: uppercase;">Output Folder</div>
            <div style="color: #00d4ff; font-size: 14px;">{output_folder}</div>
        </div>
    </div>
    """
    return html


def start_background_generation(
    book_file,
    reference_audio,
    chunk_size: int,
    selected_chapters: List[int],
    test_mode: bool,
    edited_texts: dict
) -> tuple[str, str, str, str]:
    """
    Start background generation.

    Returns:
        Tuple of (output_folder, initial_progress_html, initial_audio_html, None)
    """
    if book_file is None:
        raise gr.Error("Please upload a book file (PDF, EPUB, or TXT)")

    if reference_audio is None:
        raise gr.Error("Please upload reference audio for voice cloning")

    if selected_chapters is not None and len(selected_chapters) == 0:
        raise gr.Error("Please select at least one chapter to convert")

    # Get the turbo model
    model = load_turbo_model()

    # Test mode settings
    max_chunks, max_words = get_test_mode_limits(test_mode)

    # Start background generation
    job_id, output_folder = background_manager.start_generation(
        model=model,
        book_file=book_file.name,
        reference_audio=reference_audio,
        chunk_size=chunk_size,
        selected_chapters=selected_chapters,
        edited_texts=edited_texts,
        max_chunks=max_chunks,
        max_words=max_words
    )

    logger.info(f"Started background generation: job_id={job_id}, folder={output_folder}")

    # Initial status HTML
    initial_html = f"""
    <div style="padding: 20px; background: #1a1a2e; border-radius: 12px; margin: 10px 0; border: 1px solid #00d4ff;">
        <h3 style="margin: 0 0 15px 0; color: #00d4ff;">🚀 Generation Started</h3>
        <div style="background: #1e3a5f; padding: 12px; border-radius: 8px; margin-bottom: 15px;">
            <div style="color: #88ccff; font-size: 13px;">
                <strong>Background Mode Active</strong> - Generation will continue even if you close this browser tab.
            </div>
        </div>
        <div style="background: #252540; padding: 12px; border-radius: 8px;">
            <div style="color: #888; font-size: 11px; text-transform: uppercase;">Output Folder</div>
            <div style="color: #00d4ff; font-size: 14px;">{output_folder}</div>
        </div>
    </div>
    """

    initial_audio_html = """
    <div style="padding: 15px; background: #1a1a2e; border-radius: 12px; text-align: center; border: 1px solid #333;">
        <p style="color: #888;">🎵 Audio files will appear here as they are generated...</p>
    </div>
    """

    return output_folder, initial_html, initial_audio_html, None


def poll_generation_status(output_folder: str) -> tuple[str, str, str, bool]:
    """
    Poll progress.json for status updates.

    Returns:
        Tuple of (progress_html, audio_list_html, preview_audio_path, is_complete)
    """
    if not output_folder:
        return gr.update(), gr.update(), gr.update(), False

    progress = background_manager.get_status(output_folder)

    if progress is None:
        # No progress file yet, generation just started
        return gr.update(), gr.update(), gr.update(), False

    # Check if complete
    is_complete = progress.status in ['completed', 'completed_with_errors', 'failed']

    if is_complete:
        # Build completion HTML
        progress_html = build_completion_html(progress, output_folder)
    else:
        # Build in-progress HTML
        progress_html = build_progress_html_from_state(progress, output_folder)

    # Build audio list
    audio_html = build_audio_list_html_from_state(progress, output_folder)

    # Get first audio file for preview if available
    preview_audio = None
    if progress.completed_chunks:
        first_chunk = progress.completed_chunks[0]
        if isinstance(first_chunk, dict):
            audio_file = first_chunk.get('audio_file')
            if audio_file:
                preview_path = Path(output_folder) / audio_file
                if preview_path.exists():
                    preview_audio = str(preview_path)

    return progress_html, audio_html, preview_audio, is_complete


# Create Gradio interface
with gr.Blocks(
    title="🎙️ Chatterbox-TTS",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container { max-width: 1200px; margin: auto; }
    .model-info { padding: 10px; border-radius: 8px; margin: 10px 0; }
    .turbo-info { background: linear-gradient(135deg, #e8f5e9, #c8e6c9); border: 1px solid #a5d6a7; }
    .standard-info { background: linear-gradient(135deg, #fff3e0, #ffe0b2); border: 1px solid #ffcc80; }
    """
) as demo:

    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1>🎙️ Chatterbox-TTS</h1>
        <p style="font-size: 18px; color: #666;">
            High-quality voice cloning with model selection<br>
            <strong>Running on Apple Silicon MPS GPU</strong>
        </p>
    </div>
    """)

    with gr.Tabs():
        # =============== TAB 1: TEXT TO SPEECH ===============
        with gr.TabItem("🎤 Text to Speech"):
            with gr.Row():
                with gr.Column():
                    # MODEL SELECTOR - Prominent at the top
                    model_selector = gr.Radio(
                        choices=["turbo", "standard"],
                        value="turbo",
                        label="🚀 Model Selection",
                        info="Turbo (350M) = Fast (~3x RTF) | Standard (500M) = More features but slower (~10x RTF)"
                    )

                    text = gr.Textbox(
                        value="Hello! This is a test of Chatterbox TTS running on Apple Silicon.",
                        label="Text to synthesize",
                        max_lines=10,
                        lines=4
                    )

                    ref_wav = gr.Audio(
                        type="filepath",
                        label="Reference Audio (Optional - 6+ seconds)",
                        sources=["upload", "microphone"]
                    )

                    with gr.Row():
                        exaggeration = gr.Slider(
                            0.25, 2, step=0.05,
                            label="Exaggeration (Standard model only)",
                            value=0.5
                        )
                        cfg_weight = gr.Slider(
                            0.2, 1, step=0.05,
                            label="CFG/Pace (Standard model only)",
                            value=0.5
                        )

                    with gr.Accordion("⚙️ Advanced Options", open=False):
                        chunk_size = gr.Slider(
                            100, 400, step=25,
                            label="Chunk Size (chars)",
                            value=250
                        )
                        seed_num = gr.Number(
                            value=0,
                            label="Seed (0 = random)",
                            precision=0
                        )
                        temp = gr.Slider(
                            0.05, 5, step=0.05,
                            label="Temperature (Standard model only)",
                            value=0.8
                        )

                    run_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")

                with gr.Column():
                    audio_output = gr.Audio(label="Generated Speech")

                    gr.HTML(f"""
                    <div style="background: #f5f5f5; border: 1px solid #ccc; border-radius: 8px; padding: 15px; margin: 10px 0;">
                        <h4 style="margin: 0 0 10px 0; color: #333;">💻 System Status</h4>
                        <p style="margin: 5px 0; color: #222;"><strong>Device:</strong> {DEVICE.upper()} {'🚀' if DEVICE == 'mps' else '💻'}</p>
                        <p style="margin: 5px 0; color: #222;"><strong>PyTorch:</strong> {torch.__version__}</p>
                        <p style="margin: 5px 0; color: #222;"><strong>MPS Available:</strong> {'✅ Yes' if torch.backends.mps.is_available() else '❌ No'}</p>
                    </div>
                    """)

            run_btn.click(
                fn=generate_tts_audio,
                inputs=[
                    text,
                    ref_wav,
                    model_selector,
                    exaggeration,
                    temp,
                    seed_num,
                    cfg_weight,
                    chunk_size,
                ],
                outputs=[audio_output],
                show_progress=True
            )

            gr.Examples(
                examples=[
                    ["Hello! Testing the Chatterbox TTS system."],
                    ["The quick brown fox jumps over the lazy dog."],
                    ["Welcome to voice synthesis! This technology uses neural networks to generate natural speech."],
                ],
                inputs=[text],
                label="📝 Example Texts"
            )

        # =============== TAB 2: AUDIOBOOK GENERATOR ===============
        with gr.TabItem("📚 Audiobook Generator"):
            gr.Markdown("""
            ### Generate audiobooks from PDF, EPUB, or TXT files

            **How it works:**
            - **EPUB with chapters**: Each chapter becomes one audio file (regardless of length)
            - **PDF/TXT or EPUB without chapters**: Text is split into ~2500 word chunks

            **Features:**
            - **Background generation** - continues even if you close the browser
            - Progress saved automatically - resume if interrupted
            - Audio files saved to `audiobooks/[Book Title]/`
            """)

            with gr.Row():
                with gr.Column():
                    book_file = gr.File(
                        label="📄 Upload Book",
                        file_types=[".pdf", ".epub", ".txt"],
                        type="filepath"
                    )

                    book_preview = gr.Markdown(
                        value="Upload a book file to see preview.",
                        label="Book Info"
                    )

                    # Chapter selection
                    gr.Markdown("### 📑 Select Chapters to Convert")
                    select_all_chapters = gr.Checkbox(
                        label="✅ Select All / Unselect All",
                        value=True,
                        interactive=True,
                        info="Toggle to select or unselect all chapters"
                    )
                    chapter_selector = gr.CheckboxGroup(
                        label="Chapters",
                        choices=[],
                        value=[],
                        interactive=True
                    )

                    # Text preview/edit
                    with gr.Accordion("📝 Text Preview & Edit", open=False):
                        chapter_dropdown = gr.Dropdown(
                            label="Select Chapter to View/Edit",
                            choices=[],
                            value=None,
                            interactive=True
                        )
                        text_preview = gr.Textbox(
                            label="Chapter Text (Editable)",
                            lines=15,
                            max_lines=30,
                            interactive=True,
                            placeholder="Select a chapter from the dropdown to view and edit its text..."
                        )
                        with gr.Row():
                            save_text_btn = gr.Button("💾 Save Text Changes", variant="secondary")
                            save_status = gr.Markdown("")
                        # State to store edited texts
                        edited_texts = gr.State({})

                    audiobook_ref_audio = gr.Audio(
                        type="filepath",
                        label="🎤 Reference Voice (6+ seconds)",
                        sources=["upload", "microphone"]
                    )

                    with gr.Accordion("⚙️ Settings", open=True):
                        test_mode = gr.Checkbox(
                            label="🧪 Test Mode (First chapter only, max 1000 words)",
                            value=DEV_MODE,
                            info="Enable for quick testing - disable for full audiobook generation"
                        )

                        audiobook_chunk_size = gr.Slider(
                            1500, 3500, step=100,
                            label="Chunk Size (words) - Only for non-chapter content",
                            value=2500,
                            info="For PDFs or EPUBs without chapters"
                        )

                    generate_audiobook_btn = gr.Button(
                        "🚀 Generate Audiobook",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column():
                    progress_display = gr.HTML(
                        value="""
                        <div style="padding: 20px; background: #1a1a2e; border-radius: 12px; text-align: center; border: 1px solid #333;">
                            <h3 style="color: #00d4ff; margin: 0 0 10px 0;">📊 Generation Status</h3>
                            <p style="color: #888;">Upload a book and click Generate to start.</p>
                        </div>
                        """
                    )

                    audio_files_display = gr.HTML(
                        value="""
                        <div style="padding: 15px; background: #1a1a2e; border-radius: 12px; text-align: center; border: 1px solid #333;">
                            <p style="color: #888;">🎵 Generated files will appear here</p>
                        </div>
                        """
                    )

                    audiobook_preview = gr.Audio(
                        label="Preview (First Chapter/Part)",
                        visible=True
                    )

            # Hidden state for tracking generation
            generation_folder = gr.State(None)
            is_generation_complete = gr.State(False)

            # Timer for polling (inactive by default, polls every 3 seconds when active)
            poll_timer = gr.Timer(value=3, active=False)

            # Connect book file upload to preview, chapter selector, and dropdown
            book_file.change(
                fn=preview_book,
                inputs=[book_file],
                outputs=[book_preview, chapter_selector, chapter_dropdown, edited_texts]
            )

            # Toggle select all chapters
            select_all_chapters.change(
                fn=toggle_select_all,
                inputs=[select_all_chapters, book_file],
                outputs=[chapter_selector]
            )

            # Load chapter text when dropdown selection changes
            chapter_dropdown.change(
                fn=load_chapter_text,
                inputs=[book_file, chapter_dropdown, edited_texts],
                outputs=[text_preview]
            )

            # Save edited text
            save_text_btn.click(
                fn=save_chapter_text,
                inputs=[chapter_dropdown, text_preview, edited_texts],
                outputs=[edited_texts, save_status]
            )

            # Connect generate button - starts background generation
            generate_audiobook_btn.click(
                fn=start_background_generation,
                inputs=[book_file, audiobook_ref_audio, audiobook_chunk_size, chapter_selector, test_mode, edited_texts],
                outputs=[generation_folder, progress_display, audio_files_display, audiobook_preview]
            ).then(
                fn=lambda: gr.Timer(active=True),  # Activate polling timer
                outputs=poll_timer
            )

            # Timer polls for status updates
            poll_timer.tick(
                fn=poll_generation_status,
                inputs=[generation_folder],
                outputs=[progress_display, audio_files_display, audiobook_preview, is_generation_complete]
            ).then(
                fn=lambda done: gr.Timer(active=not done),  # Stop polling when complete
                inputs=[is_generation_complete],
                outputs=poll_timer
            )


def main():
    try:
        logger.info("Pre-loading Turbo model...")
        load_turbo_model()
        logger.info("✅ Turbo model ready!")

        demo.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            show_error=True
        )

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        demo.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            show_error=True
        )


if __name__ == "__main__":
    main()
