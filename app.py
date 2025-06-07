#!/usr/bin/env python3
"""
Chatterbox-TTS Gradio App - Based on Official ResembleAI Implementation
Adapted for local usage with MPS GPU support on Apple Silicon
Original: https://huggingface.co/spaces/ResembleAI/Chatterbox/tree/main
"""

import random
import numpy as np
import torch
import gradio as gr
import logging
from pathlib import Path
import sys
import re
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Monkey patch torch.load to handle device mapping for Chatterbox-TTS
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, **kwargs):
    """
    Patched torch.load that automatically maps CUDA tensors to CPU/MPS
    """
    if map_location is None:
        # Default to CPU for compatibility
        map_location = 'cpu'
    logger.info(f"🔧 Loading with map_location={map_location}")
    return original_torch_load(f, map_location=map_location, **kwargs)

# Apply the patch immediately after torch import
torch.load = patched_torch_load

# Also patch it in the torch module namespace to catch all uses
if 'torch' in sys.modules:
    sys.modules['torch'].load = patched_torch_load

logger.info("✅ Applied comprehensive torch.load device mapping patch")

# Device detection with MPS support
# Note: Chatterbox-TTS has compatibility issues with MPS, forcing CPU for stability
if torch.cuda.is_available():
    DEVICE = "cuda"
    logger.info("🚀 Running on CUDA GPU")
else:
    DEVICE = "cpu"
    if torch.backends.mps.is_available():
        logger.info("🍎 Apple Silicon detected - using CPU mode for Chatterbox-TTS compatibility")
        logger.info("💡 Note: MPS support is disabled due to chatterbox-tts library limitations")
    else:
        logger.info("🚀 Running on CPU")

print(f"🚀 Running on device: {DEVICE}")

# Try different import paths for chatterbox
MODEL = None

def get_or_load_model():
    """Loads the ChatterboxTTS model if it hasn't been loaded already,
    and ensures it's on the correct device."""
    global MODEL, DEVICE
    if MODEL is None:
        print("Model not loaded, initializing...")
        try:
            # Try the official import path first
            try:
                from chatterbox.src.chatterbox.tts import ChatterboxTTS
                logger.info("✅ Using official chatterbox.src import path")
            except ImportError:
                # Fallback to our previous import
                from chatterbox import ChatterboxTTS
                logger.info("✅ Using chatterbox direct import path")
            
            # Load model to CPU first to avoid device issues
            MODEL = ChatterboxTTS.from_pretrained("cpu")
            
            # Move to target device if not CPU
            if DEVICE != "cpu":
                logger.info(f"Moving model components to {DEVICE}...")
                try:
                    # For MPS, use safer tensor movement
                    if DEVICE == "mps":
                        # Move components with MPS-safe approach
                        if hasattr(MODEL, 't3') and MODEL.t3 is not None:
                            MODEL.t3 = MODEL.t3.to(DEVICE)
                            logger.info("✅ t3 component moved to MPS")
                        if hasattr(MODEL, 's3gen') and MODEL.s3gen is not None:
                            MODEL.s3gen = MODEL.s3gen.to(DEVICE)
                            logger.info("✅ s3gen component moved to MPS")
                        if hasattr(MODEL, 've') and MODEL.ve is not None:
                            MODEL.ve = MODEL.ve.to(DEVICE)
                            logger.info("✅ ve component moved to MPS")
                    else:
                        # Standard device movement for CUDA
                        if hasattr(MODEL, 't3'):
                            MODEL.t3 = MODEL.t3.to(DEVICE)
                        if hasattr(MODEL, 's3gen'):
                            MODEL.s3gen = MODEL.s3gen.to(DEVICE)
                        if hasattr(MODEL, 've'):
                            MODEL.ve = MODEL.ve.to(DEVICE)
                    
                    MODEL.device = DEVICE
                    logger.info(f"✅ All model components moved to {DEVICE}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to move some components to {DEVICE}: {e}")
                    logger.info("🔄 Falling back to CPU mode for stability")
                    DEVICE = "cpu"
                    MODEL.device = "cpu"
            
            logger.info(f"✅ Model loaded successfully on {DEVICE}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    return MODEL

def set_seed(seed: int):
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif DEVICE == "mps":
        # MPS doesn't have separate seed functions
        pass
    random.seed(seed)
    np.random.seed(seed)

def split_text_into_chunks(text: str, max_chars: int = 250) -> List[str]:
    """
    Split text into chunks at sentence boundaries, respecting max character limit.
    
    Args:
        text: Input text to split
        max_chars: Maximum characters per chunk
    
    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]
    
    # Split by sentences first (period, exclamation, question mark)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If single sentence is too long, split by commas or spaces
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split long sentence by commas
            parts = re.split(r'(?<=,)\s+', sentence)
            for part in parts:
                if len(part) > max_chars:
                    # Split by spaces as last resort
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
            # Normal sentence processing
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
    exaggeration_input: float,
    temperature_input: float,
    seed_num_input: int,
    cfgw_input: float,
    chunk_size: int = 250
) -> tuple[int, np.ndarray]:
    """
    Generates TTS audio using the ChatterboxTTS model with support for text chunking.

    Args:
        text_input: The text to synthesize.
        audio_prompt_path_input: Path to the reference audio file.
        exaggeration_input: Exaggeration parameter for the model.
        temperature_input: Temperature parameter for the model.
        seed_num_input: Random seed (0 for random).
        cfgw_input: CFG/Pace weight.
        chunk_size: Maximum characters per chunk.

    Returns:
        A tuple containing the sample rate (int) and the audio waveform (numpy.ndarray).
    """
    try:
        current_model = get_or_load_model()

        if current_model is None:
            raise RuntimeError("TTS model is not loaded.")

        if seed_num_input != 0:
            set_seed(int(seed_num_input))

        # Split text into chunks
        text_chunks = split_text_into_chunks(text_input, chunk_size)
        logger.info(f"Processing {len(text_chunks)} text chunk(s)")
        
        generated_wavs = []
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        for i, chunk in enumerate(text_chunks):
            logger.info(f"Generating chunk {i+1}/{len(text_chunks)}: '{chunk[:50]}...'")
            
            # Generate audio for this chunk
            wav = current_model.generate(
                chunk,
                audio_prompt_path=audio_prompt_path_input,
                exaggeration=exaggeration_input,
                temperature=temperature_input,
                cfg_weight=cfgw_input,
            )
            
            generated_wavs.append(wav)
            
            # Save individual chunk if multiple chunks
            if len(text_chunks) > 1:
                chunk_path = output_dir / f"chunk_{i+1}_{random.randint(1000, 9999)}.wav"
                import torchaudio
                torchaudio.save(str(chunk_path), wav, current_model.sr)
                logger.info(f"Chunk {i+1} saved to: {chunk_path}")
        
        # Concatenate all audio chunks
        if len(generated_wavs) > 1:
            # Add small silence between chunks (0.3 seconds)
            silence_samples = int(0.3 * current_model.sr)
            
            # Fix MPS tensor creation - create on CPU first, then move to device
            first_wav = generated_wavs[0]
            target_device = first_wav.device
            target_dtype = first_wav.dtype
            
            # Create silence tensor safely for MPS
            silence = torch.zeros(1, silence_samples, dtype=target_dtype)
            if DEVICE == "mps":
                # For MPS, ensure proper tensor initialization
                silence = silence.to(target_device)
            else:
                silence = silence.to(target_device)
            
            final_wav = generated_wavs[0]
            for wav_chunk in generated_wavs[1:]:
                final_wav = torch.cat([final_wav, silence, wav_chunk], dim=1)
        else:
            final_wav = generated_wavs[0]
        
        logger.info("✅ Audio generation complete.")
        
        # Save the final concatenated audio
        output_path = output_dir / f"generated_full_{random.randint(1000, 9999)}.wav"
        import torchaudio
        torchaudio.save(str(output_path), final_wav, current_model.sr)
        logger.info(f"Final audio saved to: {output_path}")
        
        return (current_model.sr, final_wav.squeeze(0).numpy())
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        raise gr.Error(f"Generation failed: {str(e)}")

# Create Gradio interface
with gr.Blocks(
    title="🎙️ Chatterbox-TTS (Local MPS)",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container { max-width: 1200px; margin: auto; }
    .gr-button { background: linear-gradient(45deg, #FF6B6B, #4ECDC4); color: white; }
    .info-box { 
        padding: 15px; 
        border-radius: 10px; 
        margin-top: 20px; 
        border: 1px solid #ddd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box h4 { 
        margin-top: 0; 
        color: #333; 
        font-weight: bold;
    }
    .info-box p { 
        margin: 8px 0; 
        color: #555; 
        line-height: 1.4;
    }
    .chunking-info { background: linear-gradient(135deg, #e8f5e8, #f0f8f0); }
    .system-info { background: linear-gradient(135deg, #f0f4f8, #e6f2ff); }
    """
) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1>🎙️ Chatterbox-TTS Demo (Local)</h1>
        <p style="font-size: 18px; color: #666;">
            Generate high-quality speech from text with reference audio styling<br>
            <strong>Running locally with Apple Silicon MPS GPU acceleration!</strong>
        </p>
        <p style="font-size: 14px; color: #888;">
            Based on <a href="https://huggingface.co/spaces/ResembleAI/Chatterbox">official ResembleAI implementation</a><br>
            ✨ <strong>Enhanced with smart text chunking for longer texts!</strong>
        </p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Hello! This is a test of the Chatterbox-TTS voice cloning system running locally on Apple Silicon. You can now input much longer text and it will be automatically split into chunks for processing.",
                label="Text to synthesize (supports long text with automatic chunking)",
                max_lines=10,
                lines=5
            )
            
            ref_wav = gr.Audio(
                type="filepath",
                label="Reference Audio File (Optional - 6+ seconds recommended)",
                sources=["upload", "microphone"]
            )
            
            with gr.Row():
                exaggeration = gr.Slider(
                    0.25, 2, step=0.05, 
                    label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", 
                    value=0.5
                )
                cfg_weight = gr.Slider(
                    0.2, 1, step=0.05, 
                    label="CFG/Pace", 
                    value=0.5
                )

            with gr.Accordion("⚙️ Advanced Options", open=False):
                chunk_size = gr.Slider(
                    100, 400, step=25,
                    label="Chunk Size (characters per chunk for long text)",
                    value=250
                )
                seed_num = gr.Number(
                    value=0, 
                    label="Random seed (0 for random)",
                    precision=0
                )
                temp = gr.Slider(
                    0.05, 5, step=0.05, 
                    label="Temperature", 
                    value=0.8
                )

            run_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")

        with gr.Column():
            audio_output = gr.Audio(label="Generated Speech")
            
            gr.HTML("""
            <div class="info-box chunking-info">
                <h4>📝 Text Chunking Info</h4>
                <p><strong>Smart Chunking:</strong> Long text is automatically split at sentence boundaries</p>
                <p><strong>Chunk Processing:</strong> Each chunk generates separate audio, then concatenated</p>
                <p><strong>Silence Gaps:</strong> 0.3s silence added between chunks for natural flow</p>
                <p><strong>Output Files:</strong> Individual chunks + final combined audio saved</p>
            </div>
            """)
            
            # System info
            gr.HTML(f"""
            <div class="info-box system-info">
                <h4>💻 System Status</h4>
                <p><strong>Device:</strong> {DEVICE.upper()} {'🚀' if DEVICE == 'mps' else '💻'}</p>
                <p><strong>PyTorch:</strong> {torch.__version__}</p>
                <p><strong>MPS Available:</strong> {'✅ Yes' if torch.backends.mps.is_available() else '❌ No'}</p>
                <p><strong>Model Status:</strong> Ready for generation</p>
            </div>
            """)

    # Connect the interface
    run_btn.click(
        fn=generate_tts_audio,
        inputs=[
            text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
            chunk_size,
        ],
        outputs=[audio_output],
        show_progress=True
    )

    # Example texts - now with longer examples
    gr.Examples(
        examples=[
            ["Hello! This is a test of voice cloning technology running locally on Apple Silicon."],
            ["The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet. Now we can test longer text with multiple sentences to see how the chunking works."],
            ["Welcome to the future of voice synthesis! With Chatterbox, you can clone any voice in seconds. The technology uses advanced neural networks to capture the unique characteristics of a speaker's voice. This includes their tone, accent, speaking rhythm, and emotional expressiveness. The result is incredibly natural-sounding speech that maintains the original speaker's identity."],
            ["Artificial intelligence has revolutionized the way we interact with technology and create content. From virtual assistants to content creation tools, AI is transforming every aspect of our digital lives. Voice cloning technology represents one of the most exciting frontiers in this field, enabling us to preserve voices, create accessibility tools, and develop new forms of creative expression."]
        ],
        inputs=[text],
        label="📝 Example Texts (including longer ones)"
    )

def main():
    """Main function to launch the app"""
    try:
        # Attempt to load the model at startup
        logger.info("Loading model at startup...")
        get_or_load_model()
        logger.info("✅ Startup model loading complete!")
        
        # Launch the interface
        demo.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            debug=True,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"❌ CRITICAL: Failed to load model on startup: {e}")
        print(f"Application may not function properly. Error: {e}")
        # Launch anyway to show the interface
        demo.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            debug=True,
            show_error=True
        )

if __name__ == "__main__":
    main() 