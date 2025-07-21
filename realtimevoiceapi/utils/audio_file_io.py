# utils/audio_file_io.py

"""
Audio File I/O Utilities

Handles loading and saving audio files in various formats.
Optimized for both fast and quality processing modes.
"""

import wave
import struct
import logging
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any, BinaryIO
import json
import csv
from datetime import datetime
from contextlib import contextmanager

# Optional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    AudioSegment = None

from audioengine.audioengine.audio_types import (
    AudioBytes, AudioFormat, AudioConfig, AudioConstants,
    AudioMetadata, ProcessingMode
)
from audioengine.audioengine.audio_processor import AudioProcessor
from ..core.exceptions import AudioError, AudioErrorType


class AudioFileIO:
    """
    Handles audio file I/O operations.
    
    Features:
    - WAV file support (native)
    - Multi-format support (with pydub)
    - Streaming file operations
    - Metadata preservation
    - Batch processing
    """
    
    def __init__(
        self,
        config: AudioConfig = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config or AudioConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.processor = AudioProcessor(config, ProcessingMode.BALANCED)
        
    # ============== WAV File Operations ==============
    
    def load_wav(
        self,
        file_path: Union[str, Path],
        convert_to_config: bool = True
    ) -> Tuple[AudioBytes, Dict[str, Any]]:
        """
        Load WAV file with metadata.
        
        Args:
            file_path: Path to WAV file
            convert_to_config: Convert to config format if True
            
        Returns:
            Tuple of (audio_bytes, metadata_dict)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise AudioError(f"File not found: {file_path}", AudioErrorType.FORMAT_ERROR)
        
        try:
            with wave.open(str(file_path), 'rb') as wav_file:
                # Extract metadata
                metadata = {
                    'filename': file_path.name,
                    'channels': wav_file.getnchannels(),
                    'sample_width': wav_file.getsampwidth(),
                    'frame_rate': wav_file.getframerate(),
                    'frame_count': wav_file.getnframes(),
                    'duration_seconds': wav_file.getnframes() / wav_file.getframerate(),
                    'file_size_bytes': file_path.stat().st_size
                }
                
                # Read audio data
                audio_data = wav_file.readframes(wav_file.getnframes())
                
                # Convert if requested
                if convert_to_config:
                    audio_data = self._convert_to_config_format(
                        audio_data,
                        metadata['channels'],
                        metadata['sample_width'],
                        metadata['frame_rate']
                    )
                    metadata['converted'] = True
                else:
                    metadata['converted'] = False
                
                self.logger.info(
                    f"Loaded {file_path.name}: "
                    f"{metadata['duration_seconds']:.2f}s @ {metadata['frame_rate']}Hz"
                )
                
                return audio_data, metadata
                
        except wave.Error as e:
            raise AudioError(f"Invalid WAV file: {e}", AudioErrorType.FORMAT_ERROR)
        except Exception as e:
            raise AudioError(f"Failed to load WAV: {e}", AudioErrorType.FORMAT_ERROR)
    
    def save_wav(
        self,
        audio_bytes: AudioBytes,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        create_dirs: bool = True
    ) -> Path:
        """
        Save audio as WAV file with optional metadata.
        
        Args:
            audio_bytes: Audio data to save
            file_path: Output file path
            metadata: Optional metadata to embed
            create_dirs: Create parent directories if True
            
        Returns:
            Path to saved file
        """
        file_path = Path(file_path)
        
        # Create directories if needed
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with wave.open(str(file_path), 'wb') as wav_file:
                wav_file.setnchannels(self.config.channels)
                wav_file.setsampwidth(self.config.bit_depth // 8)
                wav_file.setframerate(self.config.sample_rate)
                wav_file.writeframes(audio_bytes)
            
            # Save metadata separately if provided
            if metadata:
                self._save_metadata(file_path, metadata)
            
            file_size = file_path.stat().st_size
            self.logger.info(
                f"Saved {file_path.name}: {len(audio_bytes)} bytes "
                f"({file_size} bytes on disk)"
            )
            
            return file_path
            
        except Exception as e:
            # Clean up on failure
            if file_path.exists():
                file_path.unlink()
            raise AudioError(f"Failed to save WAV: {e}")
    
    # ============== Multi-Format Support ==============
    
    def load_any_format(
        self,
        file_path: Union[str, Path],
        format_hint: Optional[str] = None
    ) -> Tuple[AudioBytes, Dict[str, Any]]:
        """
        Load audio file in any supported format.
        
        Uses pydub if available, falls back to WAV-only.
        """
        file_path = Path(file_path)
        
        # Check extension
        ext = file_path.suffix.lower()
        
        # Try WAV first (native support)
        if ext == '.wav':
            return self.load_wav(file_path)
        
        # Need pydub for other formats
        if not HAS_PYDUB:
            if ext in ['.mp3', '.flac', '.ogg', '.m4a']:
                raise AudioError(
                    f"Format {ext} requires pydub: pip install pydub",
                    AudioErrorType.UNSUPPORTED_OPERATION
                )
            else:
                # Try as WAV anyway
                return self.load_wav(file_path)
        
        # Use pydub
        try:
            # Load with format detection
            if format_hint:
                audio = AudioSegment.from_file(str(file_path), format=format_hint)
            else:
                audio = AudioSegment.from_file(str(file_path))
            
            # Extract metadata
            metadata = {
                'filename': file_path.name,
                'format': audio.format,
                'channels': audio.channels,
                'sample_width': audio.sample_width,
                'frame_rate': audio.frame_rate,
                'frame_count': len(audio.get_array_of_samples()),
                'duration_seconds': len(audio) / 1000.0,
                'file_size_bytes': file_path.stat().st_size,
                'bitrate': getattr(audio, 'bitrate', None)
            }
            
            # Convert to target format
            audio = audio.set_frame_rate(self.config.sample_rate)
            audio = audio.set_channels(self.config.channels)
            audio = audio.set_sample_width(self.config.bit_depth // 8)
            
            # Get raw data
            audio_bytes = audio.raw_data
            
            self.logger.info(
                f"Loaded {file_path.name} ({metadata['format']}): "
                f"{metadata['duration_seconds']:.2f}s"
            )
            
            return audio_bytes, metadata
            
        except Exception as e:
            raise AudioError(f"Failed to load {ext} file: {e}")
    
    def save_any_format(
        self,
        audio_bytes: AudioBytes,
        file_path: Union[str, Path],
        format: Optional[str] = None,
        bitrate: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save audio in any supported format.
        
        Args:
            audio_bytes: Audio data
            file_path: Output path
            format: Output format (mp3, flac, etc)
            bitrate: Bitrate for lossy formats
            metadata: Metadata to embed
        """
        file_path = Path(file_path)
        
        # Determine format
        if format is None:
            ext = file_path.suffix.lower()[1:]  # Remove dot
            format = ext if ext else 'wav'
        
        # WAV can use native support
        if format == 'wav':
            return self.save_wav(audio_bytes, file_path, metadata)
        
        # Need pydub for other formats
        if not HAS_PYDUB:
            raise AudioError(
                f"Format {format} requires pydub: pip install pydub",
                AudioErrorType.UNSUPPORTED_OPERATION
            )
        
        try:
            # Create AudioSegment
            audio = AudioSegment(
                audio_bytes,
                sample_width=self.config.bit_depth // 8,
                frame_rate=self.config.sample_rate,
                channels=self.config.channels
            )
            
            # Prepare export parameters
            export_params = {'format': format}
            if bitrate:
                export_params['bitrate'] = bitrate
            
            # Add metadata if supported
            if metadata and format in ['mp3', 'mp4', 'm4a']:
                tags = {}
                if 'title' in metadata:
                    tags['title'] = metadata['title']
                if 'artist' in metadata:
                    tags['artist'] = metadata['artist']
                if 'album' in metadata:
                    tags['album'] = metadata['album']
                export_params['tags'] = tags
            
            # Export
            file_path.parent.mkdir(parents=True, exist_ok=True)
            audio.export(str(file_path), **export_params)
            
            self.logger.info(
                f"Saved {file_path.name} as {format} "
                f"({file_path.stat().st_size} bytes)"
            )
            
            return file_path
            
        except Exception as e:
            if file_path.exists():
                file_path.unlink()
            raise AudioError(f"Failed to save {format}: {e}")
    
    # ============== Streaming Operations ==============
    
    @contextmanager
    def stream_wav_reader(
        self,
        file_path: Union[str, Path],
        chunk_duration_ms: int = 100
    ):
        """
        Context manager for streaming WAV file reading.
        
        Yields chunks of audio data without loading entire file.
        """
        file_path = Path(file_path)
        wav_file = None
        
        try:
            wav_file = wave.open(str(file_path), 'rb')
            
            # Calculate chunk size
            chunk_samples = int(
                chunk_duration_ms * wav_file.getframerate() / 1000
            )
            chunk_bytes = (
                chunk_samples * wav_file.getnchannels() * wav_file.getsampwidth()
            )
            
            def read_chunk():
                """Read next chunk"""
                frames = wav_file.readframes(chunk_samples)
                if not frames:
                    return None
                    
                # Convert if needed
                if (wav_file.getnchannels() != self.config.channels or
                    wav_file.getframerate() != self.config.sample_rate):
                    frames = self._convert_chunk(
                        frames,
                        wav_file.getnchannels(),
                        wav_file.getsampwidth(),
                        wav_file.getframerate()
                    )
                
                return frames
            
            yield read_chunk
            
        finally:
            if wav_file:
                wav_file.close()
    
    @contextmanager
    def stream_wav_writer(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for streaming WAV file writing.
        
        Allows writing audio in chunks.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        wav_file = None
        bytes_written = 0
        
        try:
            wav_file = wave.open(str(file_path), 'wb')
            wav_file.setnchannels(self.config.channels)
            wav_file.setsampwidth(self.config.bit_depth // 8)
            wav_file.setframerate(self.config.sample_rate)
            
            def write_chunk(audio_bytes: AudioBytes):
                """Write audio chunk"""
                nonlocal bytes_written
                wav_file.writeframes(audio_bytes)
                bytes_written += len(audio_bytes)
                return bytes_written
            
            yield write_chunk
            
            # Save metadata after successful write
            if metadata:
                metadata['bytes_written'] = bytes_written
                self._save_metadata(file_path, metadata)
            
        except Exception as e:
            # Clean up on error
            if file_path.exists():
                file_path.unlink()
            raise AudioError(f"Streaming write failed: {e}")
        finally:
            if wav_file:
                wav_file.close()
    
    # ============== Batch Operations ==============
    
    def batch_convert(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        input_format: str = 'wav',
        output_format: str = 'wav',
        recursive: bool = False,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Batch convert audio files.
        
        Returns:
            Dictionary with conversion statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        if not input_dir.is_dir():
            raise AudioError(f"Input directory not found: {input_dir}")
        
        # Find files
        pattern = f"**/*.{input_format}" if recursive else f"*.{input_format}"
        input_files = list(input_dir.glob(pattern))
        
        if not input_files:
            return {
                'files_found': 0,
                'files_converted': 0,
                'errors': []
            }
        
        # Process files
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'files_found': len(input_files),
            'files_converted': 0,
            'files_skipped': 0,
            'total_duration': 0.0,
            'total_size_in': 0,
            'total_size_out': 0,
            'errors': []
        }
        
        for input_file in input_files:
            try:
                # Determine output path
                relative_path = input_file.relative_to(input_dir)
                output_file = output_dir / relative_path.with_suffix(f'.{output_format}')
                
                # Skip if exists and not overwriting
                if output_file.exists() and not overwrite:
                    stats['files_skipped'] += 1
                    continue
                
                # Load and convert
                self.logger.info(f"Converting {input_file.name}...")
                audio_bytes, metadata = self.load_any_format(input_file)
                
                # Save in new format
                output_file.parent.mkdir(parents=True, exist_ok=True)
                self.save_any_format(audio_bytes, output_file, output_format)
                
                # Update stats
                stats['files_converted'] += 1
                stats['total_duration'] += metadata.get('duration_seconds', 0)
                stats['total_size_in'] += input_file.stat().st_size
                stats['total_size_out'] += output_file.stat().st_size
                
            except Exception as e:
                error_msg = f"{input_file.name}: {str(e)}"
                stats['errors'].append(error_msg)
                self.logger.error(f"Failed to convert {error_msg}")
        
        return stats
    
    # ============== Metadata Operations ==============
    
    def _save_metadata(self, audio_file: Path, metadata: Dict[str, Any]):
        """Save metadata to companion JSON file"""
        metadata_file = audio_file.with_suffix('.json')
        
        # Add timestamp
        metadata['saved_at'] = datetime.now().isoformat()
        metadata['audio_file'] = audio_file.name
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save metadata: {e}")
    
    def load_metadata(self, audio_file: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Load metadata from companion JSON file"""
        audio_file = Path(audio_file)
        metadata_file = audio_file.with_suffix('.json')
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load metadata: {e}")
            return None
    
    # ============== Helper Methods ==============
    
    def _convert_to_config_format(
        self,
        audio_data: AudioBytes,
        channels: int,
        sample_width: int,
        frame_rate: int
    ) -> AudioBytes:
        """Convert audio to configured format"""
        # Use processor for conversion
        if channels > 1:
            audio_data = self.processor.ensure_mono(audio_data, channels)
        
        if frame_rate != self.config.sample_rate:
            audio_data = self.processor.resample(
                audio_data, frame_rate, self.config.sample_rate
            )
        
        if sample_width != 2:
            # Handle sample width conversion
            audio_data = self._convert_sample_width(
                audio_data, sample_width, 2
            )
        
        return audio_data
    
    def _convert_chunk(
        self,
        chunk: AudioBytes,
        channels: int,
        sample_width: int,
        frame_rate: int
    ) -> AudioBytes:
        """Convert a single chunk"""
        # Simplified conversion for streaming
        if channels > 1:
            chunk = self.processor.ensure_mono(chunk, channels)
        
        # Note: Resampling chunks individually can cause artifacts
        # For production, use a stateful resampler
        if frame_rate != self.config.sample_rate:
            self.logger.warning(
                "Chunk-wise resampling may cause artifacts. "
                "Consider loading entire file for resampling."
            )
        
        return chunk
    
    def _convert_sample_width(
        self,
        audio_data: AudioBytes,
        from_width: int,
        to_width: int
    ) -> AudioBytes:
        """Convert between sample widths"""
        if from_width == to_width:
            return audio_data
        
        if from_width == 1 and to_width == 2:
            # 8-bit to 16-bit
            result = bytearray()
            for byte in audio_data:
                sample = (byte - 128) * 256
                result.extend(struct.pack('<h', sample))
            return bytes(result)
        
        # Other conversions would go here
        raise AudioError(
            f"Sample width conversion {from_width}->{to_width} not implemented"
        )


# ============== Convenience Functions ==============

def load_audio_file(
    file_path: Union[str, Path],
    target_config: Optional[AudioConfig] = None
) -> AudioBytes:
    """Quick function to load any audio file"""
    io = AudioFileIO(target_config)
    audio_bytes, _ = io.load_any_format(file_path)
    return audio_bytes


def save_audio_file(
    audio_bytes: AudioBytes,
    file_path: Union[str, Path],
    config: Optional[AudioConfig] = None
):
    """Quick function to save audio file"""
    io = AudioFileIO(config)
    ext = Path(file_path).suffix.lower()[1:]
    io.save_any_format(audio_bytes, file_path, format=ext)


def convert_audio_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_config: Optional[AudioConfig] = None
) -> Dict[str, Any]:
    """Convert single audio file"""
    io = AudioFileIO(target_config)
    
    # Load
    audio_bytes, metadata = io.load_any_format(input_path)
    
    # Save
    io.save_any_format(audio_bytes, output_path)
    
    return {
        'input': str(input_path),
        'output': str(output_path),
        'duration': metadata.get('duration_seconds', 0),
        'original_format': metadata.get('format', 'unknown')
    }