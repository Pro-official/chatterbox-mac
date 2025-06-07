#!/usr/bin/env python3
"""
Chatterbox-TTS Apple Silicon Gradio Interface
Full web interface for local usage with Apple Silicon compatibility

Install gradio first: pip install gradio
Then run: python app_gradio.py
"""

import gradio as gr
from app import (
    get_or_load_model, 
    generate_audio, 
    DEVICE, 
    split_text_into_chunks,
    logger
)
import torch
import tempfile
import os

def gradio_generate_audio(
    text_input: str,
    audio_prompt_input,
    exaggeration_input: float,
    temperature_input: float,
    seed_input: int,
    cfg_weight_input: float,
    chunk_size_input: int = 250
):
    """Gradio wrapper for audio generation"""
    try:
        # Handle audio prompt
        audio_prompt_path = None
        if audio_prompt_input is not None:
            if isinstance(audio_prompt_input, tuple):
                # Gradio audio format: (sample_rate, audio_data)
                audio_prompt_path = audio_prompt_input
            elif isinstance(audio_prompt_input, str):
                # File path
                audio_prompt_path = audio_prompt_input
        
        # Generate audio using our main function
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = generate_audio(
                text=text_input,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration_input,
                temperature=temperature_input,
                seed=seed_input if seed_input != 0 else None,
                cfg_weight=cfg_weight_input,
                chunk_size=chunk_size_input,
                output_path=tmp_file.name
            )
            
            return output_path
            
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")

# Create Gradio interface
with gr.Blocks(
    title="🎙️ Chatterbox-TTS (Apple Silicon)",
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
        <h1>🎙️ Chatterbox-TTS Apple Silicon</h1>
        <p style="font-size: 18px; color: #666;">
            Generate high-quality speech from text with voice cloning<br>
            <strong>Optimized for Apple Silicon compatibility!</strong>
        </p>
        <p style="font-size: 14px; color: #888;">
            Based on <a href="https://huggingface.co/spaces/ResembleAI/Chatterbox">official ResembleAI implementation</a><br>
            ✨ <strong>Enhanced with smart text chunking and Apple Silicon support!</strong>
        </p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Hello! This is a test of the Chatterbox-TTS voice cloning system running locally on Apple Silicon.",
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
                    label="Exaggeration (Neutral = 0.5)", 
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
            </div>
            """)
            
            # System info
            gr.HTML(f"""
            <div class="info-box system-info">
                <h4>💻 System Status</h4>
                <p><strong>Device:</strong> {DEVICE.upper()} {'🍎' if torch.backends.mps.is_available() else '💻'}</p>
                <p><strong>PyTorch:</strong> {torch.__version__}</p>
                <p><strong>MPS Available:</strong> {'✅ Yes' if torch.backends.mps.is_available() else '❌ No'}</p>
                <p><strong>Compatibility:</strong> CPU mode for stability</p>
            </div>
            """)

    # Connect the interface
    run_btn.click(
        fn=gradio_generate_audio,
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

    # Example texts
    gr.Examples(
        examples=[
            ["Hello! This is a test of voice cloning running on Apple Silicon."],
            ["The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet."],
            ["Welcome to the future of voice synthesis! With Chatterbox, you can clone any voice in seconds."],
        ],
        inputs=[text],
        label="📝 Example Texts"
    )

def main():
    """Launch the Gradio interface"""
    try:
        print("🍎 Starting Chatterbox-TTS Gradio Interface")
        print(f"Device: {DEVICE}")
        
        # Pre-load model
        print("Loading model...")
        get_or_load_model()
        print("✅ Model loaded!")
        
        # Launch interface
        demo.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            debug=True,
            show_error=True
        )
        
    except ImportError as e:
        print("❌ Missing dependency!")
        print("Install with: pip install gradio")
        print("Then run: python app_gradio.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 