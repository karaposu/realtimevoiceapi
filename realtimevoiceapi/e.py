# In direct_audio_capture.py, update the __init__ method:

class DirectAudioCapture:
    """Direct audio capture with minimal latency"""
    
    def __init__(
        self,
        device: Optional[int] = None,
        config: Optional[AudioConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.device = device
        self.config = config or AudioConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Audio stream
        self.pyaudio = None
        self.stream = None
        
        # Try to initialize PyAudio with better error handling
        try:
            import pyaudio
            self.pyaudio = pyaudio.PyAudio()
            
            # If no device specified, try to find default input
            if self.device is None:
                self.device = self._find_default_input_device()
            
            # Get device info
            if self.device is not None:
                device_info = self.pyaudio.get_device_info_by_index(self.device)
                self.logger.info(f"Using audio device: {device_info.get('name', 'Unknown')}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PyAudio: {e}")
            # Don't fail here - we'll fail when trying to start capture
        
        # ... rest of the implementation ...
    
    def _find_default_input_device(self) -> Optional[int]:
        """Find the default input device"""
        try:
            # Get default input device
            default_device = self.pyaudio.get_default_input_device_info()
            return default_device['index']
        except Exception as e:
            self.logger.warning(f"No default input device found: {e}")
            
            # Try to find any input device
            for i in range(self.pyaudio.get_device_count()):
                try:
                    info = self.pyaudio.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0:
                        self.logger.info(f"Found input device: {info['name']}")
                        return i
                except:
                    continue
            
            return None
    
    def start_capture(self, callback: Optional[Callable[[AudioBytes], None]] = None) -> None:
        """Start audio capture"""
        if not self.pyaudio:
            raise AudioError("PyAudio not initialized - check microphone permissions")
        
        try:
            # Try different formats if the default doesn't work
            formats_to_try = [
                pyaudio.paInt16,
                pyaudio.paInt32,
                pyaudio.paFloat32
            ]
            
            last_error = None
            for format_type in formats_to_try:
                try:
                    self.stream = self.pyaudio.open(
                        format=format_type,
                        channels=self.config.channels,
                        rate=self.config.sample_rate,
                        input=True,
                        input_device_index=self.device,
                        frames_per_buffer=self.chunk_size // 2,  # /2 for int16
                        stream_callback=self._audio_callback if callback else None
                    )
                    self.logger.info(f"Audio stream opened with format {format_type}")
                    break
                except Exception as e:
                    last_error = e
                    continue
            
            if not self.stream:
                raise AudioError(f"Failed to open audio stream: {last_error}")
                
        except Exception as e:
            raise AudioError(f"Error starting audio capture: {e}")