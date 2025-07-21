# utils/audio_analysis.py

"""
Audio Analysis Utilities

Advanced audio analysis and quality assessment tools.
Provides both real-time and detailed analysis capabilities.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics

# Conditional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import scipy.signal
    import scipy.fft
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy = None

from audioengine.audioengine.audio_types import (
    AudioBytes, AudioConfig, AudioConstants, AudioMetadata,
    AmplitudeFloat, ProcessingMode
)
from ..core.exceptions import AudioError


class AnalysisLevel(Enum):
    """Analysis detail levels"""
    BASIC = "basic"          # Fast, minimal analysis
    STANDARD = "standard"    # Standard analysis
    DETAILED = "detailed"    # Full analysis with spectral info
    FORENSIC = "forensic"    # Deep analysis for debugging


@dataclass
class AudioAnalysisResult:
    """Complete audio analysis results"""
    
    # Basic metrics
    duration_ms: float
    sample_count: int
    
    # Amplitude analysis
    peak_amplitude: float
    rms_amplitude: float
    dynamic_range_db: float
    crest_factor_db: float
    
    # Voice/Speech detection
    is_speech_likely: bool
    speech_confidence: float
    voice_activity_ratio: float
    
    # Quality metrics
    signal_to_noise_ratio_db: Optional[float] = None
    total_harmonic_distortion: Optional[float] = None
    
    # Frequency analysis
    dominant_frequency_hz: Optional[float] = None
    spectral_centroid_hz: Optional[float] = None
    spectral_rolloff_hz: Optional[float] = None
    
    # Issues detected
    has_clipping: bool = False
    has_dc_offset: bool = False
    is_silent: bool = False
    
    # Advanced metrics
    zero_crossing_rate: Optional[float] = None
    spectral_flux: Optional[float] = None
    mfcc_features: Optional[List[float]] = None
    
    # Timing
    analysis_time_ms: float = 0.0
    analysis_level: AnalysisLevel = AnalysisLevel.STANDARD
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'duration_ms': self.duration_ms,
            'peak_amplitude': self.peak_amplitude,
            'rms_amplitude': self.rms_amplitude,
            'is_speech_likely': self.is_speech_likely,
            'speech_confidence': self.speech_confidence,
            'quality_issues': self.get_quality_issues(),
            'analysis_level': self.analysis_level.value
        }
    
    def get_quality_issues(self) -> List[str]:
        """Get list of detected quality issues"""
        issues = []
        if self.has_clipping:
            issues.append("clipping")
        if self.has_dc_offset:
            issues.append("dc_offset")
        if self.is_silent:
            issues.append("silent")
        if self.signal_to_noise_ratio_db and self.signal_to_noise_ratio_db < 10:
            issues.append("low_snr")
        return issues


class AudioAnalyzer:
    """
    Advanced audio analysis tool.
    
    Provides multiple analysis levels from basic to forensic.
    """
    
    def __init__(
        self,
        config: AudioConfig = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config or AudioConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Check available libraries
        self.has_numpy = HAS_NUMPY
        self.has_scipy = HAS_SCIPY
        
        if not self.has_numpy:
            self.logger.warning("NumPy not available - analysis limited")
        if not self.has_scipy:
            self.logger.warning("SciPy not available - spectral analysis disabled")
    
    def analyze(
        self,
        audio_bytes: AudioBytes,
        level: AnalysisLevel = AnalysisLevel.STANDARD
    ) -> AudioAnalysisResult:
        """
        Perform audio analysis at specified level.
        
        Args:
            audio_bytes: Audio data to analyze
            level: Analysis detail level
            
        Returns:
            AudioAnalysisResult with analysis data
        """
        start_time = time.time()
        
        if not self.has_numpy:
            # Fallback to basic analysis
            result = self._analyze_basic_only(audio_bytes)
        else:
            # Full analysis based on level
            if level == AnalysisLevel.BASIC:
                result = self._analyze_basic(audio_bytes)
            elif level == AnalysisLevel.STANDARD:
                result = self._analyze_standard(audio_bytes)
            elif level == AnalysisLevel.DETAILED:
                result = self._analyze_detailed(audio_bytes)
            else:  # FORENSIC
                result = self._analyze_forensic(audio_bytes)
        
        # Record analysis time
        result.analysis_time_ms = (time.time() - start_time) * 1000
        result.analysis_level = level
        
        return result
    
    def _analyze_basic_only(self, audio_bytes: AudioBytes) -> AudioAnalysisResult:
        """Basic analysis without NumPy"""
        sample_count = len(audio_bytes) // 2
        duration_ms = sample_count / self.config.sample_rate * 1000
        
        # Sample-based analysis
        max_val = 0
        sum_val = 0
        
        for i in range(0, min(len(audio_bytes) - 2, 10000), 2):
            sample = abs(struct.unpack('<h', audio_bytes[i:i+2])[0])
            max_val = max(max_val, sample)
            sum_val += sample
        
        samples_checked = min(5000, sample_count)
        avg_val = sum_val / samples_checked if samples_checked > 0 else 0
        
        return AudioAnalysisResult(
            duration_ms=duration_ms,
            sample_count=sample_count,
            peak_amplitude=max_val / 32767.0,
            rms_amplitude=avg_val / 32767.0,
            dynamic_range_db=20 * math.log10(max_val / (avg_val + 1)) if avg_val > 0 else 0,
            crest_factor_db=0,
            is_speech_likely=avg_val > 100,
            speech_confidence=min(1.0, avg_val / 1000),
            voice_activity_ratio=0.5,
            is_silent=max_val < 100,
            has_clipping=max_val > 32000
        )
    
    def _analyze_basic(self, audio_bytes: AudioBytes) -> AudioAnalysisResult:
        """Basic NumPy analysis - fast"""
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        float_samples = samples.astype(np.float32) / 32768.0
        
        # Basic metrics
        peak = float(np.max(np.abs(float_samples)))
        rms = float(np.sqrt(np.mean(float_samples ** 2)))
        
        # Dynamic range
        if rms > 0:
            dynamic_range_db = 20 * np.log10(peak / rms)
            crest_factor_db = dynamic_range_db
        else:
            dynamic_range_db = 0
            crest_factor_db = 0
        
        # Simple speech detection
        energy_threshold = 0.01
        is_speech = rms > energy_threshold and peak < 0.95
        
        return AudioAnalysisResult(
            duration_ms=len(samples) / self.config.sample_rate * 1000,
            sample_count=len(samples),
            peak_amplitude=peak,
            rms_amplitude=rms,
            dynamic_range_db=dynamic_range_db,
            crest_factor_db=crest_factor_db,
            is_speech_likely=is_speech,
            speech_confidence=min(1.0, rms * 10),
            voice_activity_ratio=0.5 if is_speech else 0.0,
            is_silent=peak < 0.001,
            has_clipping=peak > 0.95,
            has_dc_offset=abs(np.mean(float_samples)) > 0.01
        )
    
    def _analyze_standard(self, audio_bytes: AudioBytes) -> AudioAnalysisResult:
        """Standard analysis with more metrics"""
        # Start with basic analysis
        result = self._analyze_basic(audio_bytes)
        
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        float_samples = samples.astype(np.float32) / 32768.0
        
        # Zero-crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(float_samples)))) / 2
        result.zero_crossing_rate = zero_crossings / len(float_samples)
        
        # Voice activity detection
        result.voice_activity_ratio = self._calculate_vad_ratio(float_samples)
        
        # SNR estimation
        result.signal_to_noise_ratio_db = self._estimate_snr(float_samples)
        
        # Refine speech detection
        result.is_speech_likely = (
            result.voice_activity_ratio > 0.3 and
            result.zero_crossing_rate > 0.02 and
            result.zero_crossing_rate < 0.15 and
            result.rms_amplitude > 0.01
        )
        
        result.speech_confidence = min(1.0, result.voice_activity_ratio * 2)
        
        return result
    
    def _analyze_detailed(self, audio_bytes: AudioBytes) -> AudioAnalysisResult:
        """Detailed analysis with spectral features"""
        # Start with standard analysis
        result = self._analyze_standard(audio_bytes)
        
        if not self.has_scipy:
            return result
        
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        float_samples = samples.astype(np.float32) / 32768.0
        
        # Spectral analysis
        spectral_features = self._calculate_spectral_features(float_samples)
        result.dominant_frequency_hz = spectral_features.get('dominant_freq')
        result.spectral_centroid_hz = spectral_features.get('centroid')
        result.spectral_rolloff_hz = spectral_features.get('rolloff')
        result.spectral_flux = spectral_features.get('flux')
        
        # THD calculation
        result.total_harmonic_distortion = self._calculate_thd(float_samples)
        
        return result
    
    def _analyze_forensic(self, audio_bytes: AudioBytes) -> AudioAnalysisResult:
        """Forensic level analysis - comprehensive"""
        # Start with detailed analysis
        result = self._analyze_detailed(audio_bytes)
        
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        float_samples = samples.astype(np.float32) / 32768.0
        
        # MFCC features (simplified)
        if self.has_scipy:
            result.mfcc_features = self._calculate_mfcc(float_samples)
        
        # Additional forensic checks
        # - Check for encoding artifacts
        # - Detect potential tampering
        # - Analyze noise profile
        
        return result
    
    # ============== Helper Methods ==============
    
    def _calculate_vad_ratio(self, samples: np.ndarray) -> float:
        """Calculate voice activity ratio"""
        frame_size = int(0.02 * self.config.sample_rate)  # 20ms frames
        energy_threshold = 0.01
        
        active_frames = 0
        total_frames = len(samples) // frame_size
        
        for i in range(total_frames):
            frame = samples[i * frame_size:(i + 1) * frame_size]
            frame_energy = np.sqrt(np.mean(frame ** 2))
            if frame_energy > energy_threshold:
                active_frames += 1
        
        return active_frames / total_frames if total_frames > 0 else 0
    
    def _estimate_snr(self, samples: np.ndarray) -> float:
        """Estimate signal-to-noise ratio"""
        if len(samples) < self.config.sample_rate:
            return 0.0
        
        # Use first 100ms as noise reference
        noise_samples = samples[:int(0.1 * self.config.sample_rate)]
        noise_power = np.mean(noise_samples ** 2)
        
        # Use middle portion as signal
        mid_start = len(samples) // 4
        mid_end = 3 * len(samples) // 4
        signal_samples = samples[mid_start:mid_end]
        signal_power = np.mean(signal_samples ** 2)
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
            return float(max(0, min(60, snr_db)))  # Clamp to reasonable range
        
        return 60.0  # Assume very high SNR if no noise
    
    def _calculate_spectral_features(
        self,
        samples: np.ndarray
    ) -> Dict[str, float]:
        """Calculate spectral features using FFT"""
        # Window the signal
        window = scipy.signal.windows.hann(len(samples))
        windowed = samples * window
        
        # Compute FFT
        fft = scipy.fft.rfft(windowed)
        magnitude = np.abs(fft)
        freqs = scipy.fft.rfftfreq(len(samples), 1 / self.config.sample_rate)
        
        # Dominant frequency
        dominant_idx = np.argmax(magnitude[1:]) + 1  # Skip DC
        dominant_freq = float(freqs[dominant_idx])
        
        # Spectral centroid
        centroid = float(np.sum(freqs * magnitude) / np.sum(magnitude))
        
        # Spectral rolloff (95% of energy)
        cumsum = np.cumsum(magnitude)
        rolloff_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0][0]
        rolloff = float(freqs[rolloff_idx])
        
        # Spectral flux (simplified)
        if len(samples) > 2048:
            prev_magnitude = magnitude[:-1]
            curr_magnitude = magnitude[1:]
            flux = float(np.mean(np.abs(curr_magnitude - prev_magnitude)))
        else:
            flux = 0.0
        
        return {
            'dominant_freq': dominant_freq,
            'centroid': centroid,
            'rolloff': rolloff,
            'flux': flux
        }
    
    def _calculate_thd(self, samples: np.ndarray) -> float:
        """Calculate total harmonic distortion"""
        if len(samples) < 2048:
            return 0.0
        
        # Use middle portion
        mid = len(samples) // 2
        segment = samples[mid - 1024:mid + 1024]
        
        # FFT
        fft = scipy.fft.rfft(segment)
        magnitude = np.abs(fft)
        
        # Find fundamental
        fundamental_idx = np.argmax(magnitude[1:100]) + 1
        fundamental_power = magnitude[fundamental_idx] ** 2
        
        # Sum harmonics (2nd through 5th)
        harmonic_power = 0
        for h in range(2, 6):
            harmonic_idx = fundamental_idx * h
            if harmonic_idx < len(magnitude):
                harmonic_power += magnitude[harmonic_idx] ** 2
        
        if fundamental_power > 0:
            thd = np.sqrt(harmonic_power / fundamental_power)
            return float(min(1.0, thd))
        
        return 0.0
    
    def _calculate_mfcc(self, samples: np.ndarray) -> List[float]:
        """Calculate simplified MFCC features"""
        # This is a simplified version
        # Real MFCC would use mel filterbanks
        
        # Take short segment
        segment_size = min(2048, len(samples))
        segment = samples[:segment_size]
        
        # FFT
        fft = scipy.fft.rfft(segment)
        magnitude = np.abs(fft)
        
        # Log magnitude
        log_magnitude = np.log(magnitude + 1e-10)
        
        # DCT to get cepstral coefficients
        dct = scipy.fft.dct(log_magnitude, type=2, norm='ortho')
        
        # Return first 13 coefficients
        return [float(x) for x in dct[:13]]


class RealtimeAnalyzer:
    """
    Lightweight analyzer for real-time audio monitoring.
    
    Maintains running statistics without storing full audio.
    """
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.sample_count = 0
        self.sum_squares = 0.0
        self.max_amplitude = 0.0
        self.zero_crossings = 0
        self.last_sample = 0
        
        # Circular buffer for recent samples
        self.buffer_size = 1024
        self.recent_samples = np.zeros(self.buffer_size) if HAS_NUMPY else []
        self.buffer_pos = 0
    
    def process_chunk(self, audio_bytes: AudioBytes) -> Dict[str, float]:
        """
        Process audio chunk and return current statistics.
        
        Returns:
            Dictionary with current metrics
        """
        if HAS_NUMPY:
            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            return self._process_numpy(samples)
        else:
            return self._process_basic(audio_bytes)
    
    def _process_numpy(self, samples: np.ndarray) -> Dict[str, float]:
        """Process with NumPy"""
        float_samples = samples.astype(np.float32) / 32768.0
        
        # Update running stats
        self.sample_count += len(samples)
        self.sum_squares += np.sum(float_samples ** 2)
        self.max_amplitude = max(self.max_amplitude, float(np.max(np.abs(float_samples))))
        
        # Zero crossings
        if self.last_sample != 0:
            # Check crossing from last chunk
            if np.sign(self.last_sample) != np.sign(float_samples[0]):
                self.zero_crossings += 1
        
        crossings = np.sum(np.abs(np.diff(np.sign(float_samples)))) / 2
        self.zero_crossings += crossings
        self.last_sample = float_samples[-1]
        
        # Update circular buffer
        for sample in float_samples[-self.buffer_size:]:
            self.recent_samples[self.buffer_pos] = sample
            self.buffer_pos = (self.buffer_pos + 1) % self.buffer_size
        
        # Calculate current metrics
        rms = np.sqrt(self.sum_squares / self.sample_count)
        zcr = self.zero_crossings / self.sample_count
        
        # Recent energy
        recent_rms = float(np.sqrt(np.mean(self.recent_samples ** 2)))
        
        return {
            'sample_count': self.sample_count,
            'duration_ms': self.sample_count / self.config.sample_rate * 1000,
            'current_rms': recent_rms,
            'overall_rms': float(rms),
            'peak_amplitude': self.max_amplitude,
            'zero_crossing_rate': float(zcr),
            'is_speech_likely': recent_rms > 0.01 and zcr > 0.02 and zcr < 0.15
        }
    
    def _process_basic(self, audio_bytes: AudioBytes) -> Dict[str, float]:
        """Process without NumPy"""
        # Basic implementation
        sample_count = len(audio_bytes) // 2
        self.sample_count += sample_count
        
        # Simple peak detection
        for i in range(0, len(audio_bytes) - 2, 2):
            sample = abs(struct.unpack('<h', audio_bytes[i:i+2])[0]) / 32768.0
            self.max_amplitude = max(self.max_amplitude, sample)
        
        return {
            'sample_count': self.sample_count,
            'duration_ms': self.sample_count / self.config.sample_rate * 1000,
            'peak_amplitude': self.max_amplitude,
            'is_speech_likely': self.max_amplitude > 0.01
        }


# ============== Quality Monitor ==============

class AudioQualityMonitor:
    """
    Monitor audio quality over time.
    
    Tracks quality metrics and detects issues.
    """
    
    def __init__(
        self,
        config: AudioConfig = None,
        window_size_ms: int = 1000
    ):
        self.config = config or AudioConfig()
        self.window_size_ms = window_size_ms
        self.analyzer = RealtimeAnalyzer(config)
        
        # History tracking
        self.history: List[Dict[str, float]] = []
        self.max_history = 100
        
        # Issue tracking
        self.issues_detected: List[Tuple[float, str]] = []
        
    def add_chunk(self, audio_bytes: AudioBytes) -> Dict[str, Any]:
        """
        Add audio chunk and check quality.
        
        Returns:
            Dictionary with quality status and any issues
        """
        # Analyze chunk
        metrics = self.analyzer.process_chunk(audio_bytes)
        
        # Add to history
        self.history.append(metrics)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Check for issues
        issues = self._check_quality_issues(metrics)
        
        # Track issues
        if issues:
            timestamp = metrics['duration_ms']
            for issue in issues:
                self.issues_detected.append((timestamp, issue))
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(metrics)
        
        return {
            'metrics': metrics,
            'quality_score': quality_score,
            'current_issues': issues,
            'total_issues': len(self.issues_detected)
        }
    
    def _check_quality_issues(self, metrics: Dict[str, float]) -> List[str]:
        """Check for quality issues"""
        issues = []
        
        # Check for clipping
        if metrics.get('peak_amplitude', 0) > 0.95:
            issues.append('clipping')
        
        # Check for silence
        if metrics.get('current_rms', 1) < 0.001:
            issues.append('silence')
        
        # Check for noise (high frequency content)
        zcr = metrics.get('zero_crossing_rate', 0)
        if zcr > 0.3:
            issues.append('high_frequency_noise')
        
        # Check for DC offset (would need more analysis)
        
        return issues
    
    def _calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score (0-1)"""
        score = 1.0
        
        # Penalize clipping
        peak = metrics.get('peak_amplitude', 0)
        if peak > 0.95:
            score *= 0.7
        elif peak > 0.9:
            score *= 0.9
        
        # Penalize very low level
        rms = metrics.get('current_rms', 0)
        if rms < 0.001:
            score *= 0.5
        elif rms < 0.01:
            score *= 0.8
        
        # Good dynamic range
        if peak > 0 and rms > 0:
            dynamic_range = peak / rms
            if dynamic_range > 10 and dynamic_range < 100:
                score *= 1.1  # Bonus for good dynamics
        
        return min(1.0, max(0.0, score))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get quality monitoring summary"""
        if not self.history:
            return {
                'status': 'no_data',
                'average_quality': 0.0,
                'issues_detected': []
            }
        
        # Calculate averages
        avg_rms = statistics.mean(h.get('overall_rms', 0) for h in self.history)
        avg_peak = statistics.mean(h.get('peak_amplitude', 0) for h in self.history)
        
        # Issue summary
        issue_counts = {}
        for _, issue in self.issues_detected:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        return {
            'status': 'ok' if not self.issues_detected else 'issues_detected',
            'samples_analyzed': self.analyzer.sample_count,
            'duration_ms': self.analyzer.sample_count / self.config.sample_rate * 1000,
            'average_rms': avg_rms,
            'average_peak': avg_peak,
            'total_issues': len(self.issues_detected),
            'issue_summary': issue_counts
        }


# ============== Convenience Functions ==============

def analyze_audio(
    audio_bytes: AudioBytes,
    level: str = "standard"
) -> AudioAnalysisResult:
    """Quick function to analyze audio"""
    analyzer = AudioAnalyzer()
    analysis_level = AnalysisLevel[level.upper()]
    return analyzer.analyze(audio_bytes, analysis_level)


def check_audio_quality(
    audio_bytes: AudioBytes
) -> Dict[str, Any]:
    """Quick quality check"""
    result = analyze_audio(audio_bytes, "standard")
    
    return {
        'is_good_quality': (
            not result.has_clipping and
            not result.is_silent and
            result.signal_to_noise_ratio_db > 20
        ),
        'issues': result.get_quality_issues(),
        'speech_detected': result.is_speech_likely,
        'confidence': result.speech_confidence
    }


def create_quality_monitor(
    sample_rate: int = 24000
) -> AudioQualityMonitor:
    """Create quality monitor for real-time use"""
    config = AudioConfig(sample_rate=sample_rate)
    return AudioQualityMonitor(config)