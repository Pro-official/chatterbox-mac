#!/usr/bin/env python3
"""
Chatterbox-TTS Gradio App with Model Selection
Supports both Standard (500M) and Turbo (350M) models.
Default: Turbo for faster generation.
"""

import random
import numpy as np
import torch
import gradio as gr
import logging
from pathlib import Path
import sys
import re
from typing import List, Optional

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
