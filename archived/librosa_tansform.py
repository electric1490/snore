import librosa
import numpy as np

def prepare_audio_for_yamnet(audio_file_path):
    """
    Loads an audio file, resamples it to 16kHz, and ensures it's a 1-second chunk.

    Args:
        audio_file_path: Path to the audio file.

    Returns:
        A NumPy array representing the audio data, resampled and trimmed/padded to 1 second,
        or None if an error occurs during loading.  Also returns the sample rate.
    """
    try:
        # Load the audio file using librosa.  It handles various formats.
        audio, sr = librosa.load(audio_file_path, sr=None)  # sr=None preserves original sample rate

        # Resample to 16kHz if necessary. YAMNET was trained on 16kHz audio.
        if sr != 16000:
            audio = librosa.resample(audio, sr, 16000)
            sr = 16000

        # Ensure 1-second length.  YAMNET expects fixed-length input.
        target_samples = sr  # 1 second * sample rate
        if len(audio) > target_samples:
            audio = audio[:target_samples]  # Trim if longer
        elif len(audio) < target_samples:
            # Pad with zeros if shorter.  Important for consistent input.
            padding = target_samples - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')

        return audio, sr

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None, None



# Example usage:
audio_file = "path/to/your/audio.wav"  # Replace with your audio file path

audio_data, sr = prepare_audio_for_yamnet(audio_file)

if audio_data is not None:
    print(f"Audio data shape: {audio_data.shape}")
    print(f"Sample Rate: {sr}")

    # Now you can use audio_data with YAMNET:
    # Example (assuming you have the YAMNET model loaded as 'yamnet_model'):
    # scores, embeddings, spectrogram = yamnet_model.predict(np.expand_dims(audio_data, axis=0)) # Add batch dimension

    # ... further processing of scores, embeddings, or spectrogram ...

else:
    print("Audio processing failed.")



# Example of how to create a dummy audio file for testing:
import soundfile as sf
dummy_audio = np.random.uniform(-1, 1, 16000) # 1 second of random audio at 16kHz
sf.write('dummy_audio.wav', dummy_audio, 16000) # Save it.
