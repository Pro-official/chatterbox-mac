"""
MLX-Hybrid Chatterbox TTS

This implementation uses:
- MLX for VoiceEncoder (speaker embedding) - ~10x faster
- MLX for T3 (text-to-tokens) - ~5x faster
- PyTorch/MPS for S3Gen (tokens-to-mel) - efficient on MPS
- PyTorch/MPS for HiFiGAN (mel-to-wav) - efficient on MPS

The T3 model is the main bottleneck (30 transformer layers),
so MLX acceleration there provides the biggest performance gains.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import mlx.core as mx

# MLX components
from .voice_encoder import VoiceEncoder, VoiceEncConfig, load_from_pytorch as load_ve
from .t3 import T3, T3Config, T3Cond, load_from_pytorch as load_t3

# PyTorch components (S3Gen, Tokenizer)
from chatterbox.models.s3gen.s3gen import S3Token2Wav
from chatterbox.models.s3gen.const import S3GEN_SR
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.tokenizers import EnTokenizer

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for TTS generation."""
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 1000
    repetition_penalty: float = 1.2
    max_tokens: int = 1000
    n_cfm_steps: int = 10


def punc_norm(text: str) -> str:
    """Quick cleanup func for punctuation."""
    if len(text) == 0:
        return "You need to add some text for me to talk."

    if text[0].islower():
        text = text[0].upper() + text[1:]

    text = " ".join(text.split())

    punc_to_replace = [
        ("...", ", "), ("…", ", "), (":", ","), (" - ", ", "),
        (";", ", "), ("—", "-"), ("–", "-"), (" ,", ","),
        (""", "\""), (""", "\""), ("'", "'"), ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


class ChatterboxMLX:
    """
    Hybrid MLX-PyTorch Chatterbox TTS.

    Uses MLX for the compute-intensive T3 model and
    PyTorch/MPS for S3Gen and HiFiGAN.
    """

    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "mps",
        use_turbo: bool = True,
    ):
        """
        Initialize Chatterbox with MLX acceleration.

        Args:
            model_path: Path to model weights (or None to download)
            device: PyTorch device for S3Gen/HiFiGAN ("mps" or "cpu")
            use_turbo: Whether to use turbo model variant
        """
        self.device = device
        self.use_turbo = use_turbo
        self.sample_rate = S3GEN_SR

        # Load all components
        self._load_components()

    def _load_components(self):
        """Load MLX and PyTorch components."""
        from huggingface_hub import hf_hub_download

        if self.use_turbo:
            repo_id = "ResembleAI/chatterbox-turbo"
            t3_filename = "t3_turbo_v1.safetensors"
            s3gen_filename = "s3gen_meanflow.safetensors"
        else:
            repo_id = "ResembleAI/chatterbox"
            t3_filename = "t3_cfg.safetensors"
            s3gen_filename = "s3gen.safetensors"

        # Download all weight files
        logger.info(f"Loading from {repo_id}...")
        ve_path = hf_hub_download(repo_id=repo_id, filename="ve.safetensors", token=None)
        t3_path = hf_hub_download(repo_id=repo_id, filename=t3_filename, token=None)
        s3gen_path = hf_hub_download(repo_id=repo_id, filename=s3gen_filename, token=None)

        # Get tokenizer path (different for turbo vs standard)
        if self.use_turbo:
            # Turbo uses HuggingFace tokenizer format
            tokenizer_path = hf_hub_download(repo_id=repo_id, filename="vocab.json", token=None)
            tokenizer_path = Path(tokenizer_path).parent  # Get directory for HF tokenizer
        else:
            tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", token=None)

        # Load MLX VoiceEncoder
        logger.info("Loading MLX VoiceEncoder...")
        self.voice_encoder = load_ve(ve_path)

        # Load MLX T3
        logger.info("Loading MLX T3...")
        self.t3 = load_t3(t3_path)

        # Load PyTorch S3Gen
        logger.info("Loading PyTorch S3Gen...")
        from safetensors.torch import load_file
        self.s3gen = S3Token2Wav(meanflow=self.use_turbo)
        self.s3gen.load_state_dict(load_file(s3gen_path), strict=False)
        self.s3gen.to(self.device).eval()

        # Load tokenizer
        logger.info("Loading tokenizer...")
        if self.use_turbo:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            self._use_hf_tokenizer = True
        else:
            self.tokenizer = EnTokenizer(str(tokenizer_path))
            self._use_hf_tokenizer = False

        logger.info("All components loaded!")

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        audio_prompt: Union[str, np.ndarray, torch.Tensor],
        audio_prompt_sr: Optional[int] = None,
        config: Optional[GenerationConfig] = None,
    ) -> np.ndarray:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            audio_prompt: Reference audio (path or array)
            audio_prompt_sr: Sample rate of audio_prompt (if array)
            config: Generation configuration

        Returns:
            Generated audio as numpy array (24kHz)
        """
        config = config or GenerationConfig()

        t_start = time.perf_counter()

        # Load audio prompt
        if isinstance(audio_prompt, str):
            audio_prompt, audio_prompt_sr = librosa.load(audio_prompt, sr=None)

        if isinstance(audio_prompt, torch.Tensor):
            audio_prompt = audio_prompt.cpu().numpy()

        audio_prompt_sr = audio_prompt_sr or S3GEN_SR

        # Resample reference to required rates
        ref_16k = librosa.resample(audio_prompt, orig_sr=audio_prompt_sr, target_sr=S3_SR)
        ref_24k = librosa.resample(audio_prompt, orig_sr=audio_prompt_sr, target_sr=S3GEN_SR)

        # Limit reference length
        ref_16k = ref_16k[:self.ENC_COND_LEN]
        ref_24k = ref_24k[:self.DEC_COND_LEN]

        # Get speaker embedding using MLX VoiceEncoder
        t_ve_start = time.perf_counter()
        speaker_emb = self.voice_encoder.embed_wav(ref_16k, sample_rate=S3_SR)
        t_ve_end = time.perf_counter()
        logger.debug(f"VoiceEncoder (MLX): {t_ve_end - t_ve_start:.3f}s")

        # Tokenize text
        text = punc_norm(text)
        if self._use_hf_tokenizer:
            # HuggingFace tokenizer for turbo
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            text_tokens = torch.tensor(tokens).unsqueeze(0)
        else:
            # EnTokenizer for standard
            text_tokens = self.tokenizer.text_to_tokens(text)

        # Add start/stop tokens
        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        # Get speech prompt tokens for T3 conditioning
        speech_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            speech_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k], max_len=plen)
            speech_cond_prompt_tokens = torch.atleast_2d(speech_cond_prompt_tokens)

        # Prepare T3 conditioning (MLX)
        t3_cond = T3Cond(
            speaker_emb=mx.array(speaker_emb),
            cond_prompt_speech_tokens=mx.array(speech_cond_prompt_tokens.cpu().numpy()) if speech_cond_prompt_tokens is not None else None,
            emotion_adv=config.exaggeration,
        )

        # Convert text tokens to MLX
        text_tokens_mx = mx.array(text_tokens.numpy())

        # Generate speech tokens using MLX T3
        t_t3_start = time.perf_counter()
        speech_tokens = self.t3.inference_turbo(
            t3_cond=t3_cond,
            text_tokens=text_tokens_mx,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            max_gen_len=config.max_tokens,
        )
        mx.eval(speech_tokens)
        t_t3_end = time.perf_counter()
        logger.debug(f"T3 (MLX): {t_t3_end - t_t3_start:.2f}s, {int(speech_tokens.shape[1])} tokens")

        # Convert tokens to PyTorch and filter invalid
        speech_tokens_np = np.array(speech_tokens).flatten()
        speech_tokens_np = speech_tokens_np[speech_tokens_np < 6561]  # Remove special tokens
        speech_tokens_pt = torch.from_numpy(speech_tokens_np).long().to(self.device)

        # Prepare S3Gen reference
        t_s3_start = time.perf_counter()
        ref_dict = self.s3gen.embed_ref(
            torch.from_numpy(ref_24k).float(),
            S3GEN_SR,
            device=self.device,
        )

        # Generate waveform
        n_cfm_steps = 2 if self.use_turbo else config.n_cfm_steps
        output_wav, _ = self.s3gen.inference(
            speech_tokens_pt,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_steps,
        )
        t_s3_end = time.perf_counter()
        logger.debug(f"S3Gen+HiFiGAN: {t_s3_end - t_s3_start:.2f}s")

        # Convert to numpy
        output_wav = output_wav.squeeze().cpu().numpy()

        t_total = time.perf_counter() - t_start
        audio_duration = len(output_wav) / self.sample_rate
        rtf = t_total / audio_duration
        logger.info(f"Generated {audio_duration:.1f}s audio in {t_total:.1f}s (RTF: {rtf:.2f}x)")

        return output_wav

    @classmethod
    def from_pretrained(
        cls,
        device: str = "mps",
        turbo: bool = True,
    ) -> "ChatterboxMLX":
        """
        Load Chatterbox from pretrained weights.

        Args:
            device: Device for PyTorch components
            turbo: Whether to use turbo variant

        Returns:
            ChatterboxMLX instance
        """
        return cls(device=device, use_turbo=turbo)


def test_chatterbox_mlx():
    """Test the MLX Chatterbox."""
    import tempfile
    import soundfile as sf

    print("Testing ChatterboxMLX...")

    # Create a dummy reference audio
    dummy_audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1

    # Initialize model
    model = ChatterboxMLX.from_pretrained(turbo=True)

    # Generate
    print("\nGenerating speech...")
    output = model.generate(
        text="Hello, this is a test of the MLX-accelerated Chatterbox.",
        audio_prompt=dummy_audio,
        audio_prompt_sr=16000,
    )

    print(f"Output shape: {output.shape}")
    print(f"Output duration: {len(output) / 24000:.2f}s")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, output, 24000)
        print(f"Saved to: {f.name}")

    print("\n✅ ChatterboxMLX test passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_chatterbox_mlx()
