import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import sounddevice as sd
import glob

# --- UTILITY AND ANALYSIS FUNCTIONS ---

def generate_test_wav(filename="generated_test_audio.wav"):
    """Generates a simple test WAV file for demonstration."""
    print(f"\nGenerating a test audio file: '{filename}'")
    sample_rate = 44100
    duration = 3
    freq1, freq2 = 440, 880
    t = np.linspace(0., duration, int(sample_rate * duration))
    amplitude = np.iinfo(np.int16).max * 0.4
    data = amplitude * (np.sin(2. * np.pi * freq1 * t) + 0.5 * np.sin(2. * np.pi * freq2 * t))
    wavfile.write(filename, sample_rate, data.astype(np.int16))
    print("Test audio generated successfully.")
    return filename

def compress_reconstruct(audio_data, sample_rate, keep_percentage, verbose=True):
    """
    Performs FFT compression, returns reconstructed audio and performance metrics.
    """
    if verbose:
        print(f"\nProcessing... Keeping top {keep_percentage:.2f}% of frequencies.")
    
    original_length = len(audio_data)
    fft_coeffs = np.fft.rfft(audio_data.astype(np.float64))
    
    fft_magnitudes = np.abs(fft_coeffs)
    
    if keep_percentage >= 100:
        threshold = 0
    elif keep_percentage <= 0:
        threshold = np.inf
    else:
        sorted_magnitudes = np.sort(fft_magnitudes)[::-1]
        coeffs_to_keep = int(len(sorted_magnitudes) * keep_percentage / 100)
        threshold = sorted_magnitudes[coeffs_to_keep - 1]

    compressed_fft_coeffs = fft_coeffs * (fft_magnitudes >= threshold)
    
    original_non_zero = np.count_nonzero(fft_coeffs)
    compressed_non_zero = np.count_nonzero(compressed_fft_coeffs)
    if compressed_non_zero == 0: compressed_non_zero = 1
    compression_ratio = original_non_zero / compressed_non_zero
    
    if verbose:
        print(f"Coefficient compression ratio: {compression_ratio:.2f} : 1")
    
    reconstructed_signal = np.fft.irfft(compressed_fft_coeffs, n=original_length)
    reconstructed_audio = np.clip(reconstructed_signal, -32767, 32767).astype(audio_data.dtype)
    
    return reconstructed_audio, fft_coeffs, compressed_fft_coeffs, compression_ratio

def play_audio(audio_data, sample_rate, description=""):
    """Plays audio using sounddevice, robustly interruptible."""
    sd.play(audio_data, sample_rate)
    try:
        input(f"▶️  Playing {description} audio... (Press ENTER or Ctrl+C to stop)")
        sd.stop()
    except KeyboardInterrupt:
        sd.stop()
        print("\nPlayback stopped by user.")
    except Exception as e:
        sd.stop()
        print(f"An error occurred during playback: {e}")

def visualize_all(original_data, reconstructed_data, fft_coeffs, compressed_fft_coeffs, sample_rate):
    """Creates a 2x2 grid of plots for comprehensive analysis."""
    print("\n📊 Generating comprehensive plots...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('FFT Audio Compression Analysis', fontsize=16)

    ax = axs[0, 0]
    NFFT_SIZE = 1024
    if len(original_data) > NFFT_SIZE:
        Pxx, freqs, bins, im = ax.specgram(original_data, Fs=sample_rate, NFFT=NFFT_SIZE, cmap='viridis')
        fig.colorbar(im, ax=ax).set_label('Intensity [dB]')
    else:
        ax.text(0.5, 0.5, 'Audio too short for spectrogram', ha='center', va='center')
    ax.set_title('Spectrogram of Original Audio'); ax.set_xlabel('Time [s]'); ax.set_ylabel('Frequency [Hz]')

    freq_axis = np.fft.rfftfreq(len(original_data), d=1./sample_rate)
    def safe_plot(ax, x, y, **kwargs): min_len = min(len(x), len(y)); ax.semilogy(x[:min_len], np.abs(y[:min_len]), **kwargs)

    ax = axs[0, 1]; safe_plot(ax, freq_axis, fft_coeffs, color='blue'); ax.set_title('FFT of Original Signal'); ax.set_xlabel('Frequency [Hz]'); ax.set_ylabel('Magnitude (log scale)'); ax.grid(True)
    ax = axs[1, 0]; safe_plot(ax, freq_axis, compressed_fft_coeffs, color='orange'); ax.set_title('FFT after Compression'); ax.set_xlabel('Frequency [Hz]'); ax.set_ylabel('Magnitude (log scale)'); ax.grid(True)
    ax = axs[1, 1]; fft_reconstructed = np.fft.rfft(reconstructed_data.astype(np.float64)); safe_plot(ax, freq_axis, fft_reconstructed, color='green'); ax.set_title('FFT of Reconstructed Signal'); ax.set_xlabel('Frequency [Hz]'); ax.set_ylabel('Magnitude (log scale)'); ax.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

def calculate_mse(original, reconstructed):
    """Calculates the Mean Squared Error between two signals."""
    return np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)

def calculate_snr(original, reconstructed):
    """Calculates the Signal-to-Noise Ratio (in dB)."""
    noise = original.astype(np.float64) - reconstructed.astype(np.float64)
    power_signal = np.mean(original.astype(np.float64) ** 2)
    power_noise = np.mean(noise ** 2)
    if power_noise == 0: return np.inf
    return 10 * np.log10(power_signal / power_noise)

# --- MODIFIED FUNCTION TO INCLUDE A TABLE ---
def plot_quality_analysis(original_audio, sample_rate):
    """
    Runs compression at multiple levels, prints a results table,
    and then plots the trade-off.
    """
    print("\n🔬 Running Quality vs. Compression Analysis...")
    print("This may take a moment as it processes the audio multiple times.")
    
    compression_levels = [95, 80, 60, 40, 20, 15, 10, 5, 2, 1]
    snr_results, mse_results, ratio_results = [], [], []

    for level in compression_levels:
        print(f"  Testing with {level}% of frequencies kept...")
        reconstructed, _, _, ratio = compress_reconstruct(original_audio, sample_rate, level, verbose=False)
        snr_results.append(calculate_snr(original_audio, reconstructed))
        mse_results.append(calculate_mse(original_audio, reconstructed))
        ratio_results.append(ratio)
        
    print("\n--- Analysis Complete ---")

    # --- NEW: Print a formatted table of the results ---
    print("="*60)
    print(f"{'Keep (%)':<12} | {'Comp. Ratio':<15} | {'SNR (dB)':<12} | {'MSE':<15}")
    print("-"*60)
    for level, ratio, snr, mse in zip(compression_levels, ratio_results, snr_results, mse_results):
        print(f"{level:<12.1f} | {ratio:<15.2f} | {snr:<12.2f} | {mse:<15.2f}")
    print("="*60)
    print("\nGenerating plot...")

    # Plotting code remains the same
    fig, ax1 = plt.subplots(figsize=(12, 7))
    color1 = 'tab:blue'
    ax1.set_xlabel('Coefficient Compression Ratio (Higher is more compressed)')
    ax1.set_ylabel('Signal-to-Noise Ratio (SNR) in dB', color=color1)
    ax1.plot(ratio_results, snr_results, 'o-', color=color1, label='SNR (higher is better)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, which='both', linestyle='--')
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Mean Squared Error (MSE)', color=color2)
    ax2.plot(ratio_results, mse_results, 's--', color=color2, label='MSE (lower is better)')
    ax2.tick_params(axis='y', labelcolor=color2)
    if np.max(mse_results) > 0 and np.min(mse_results) > 0 and np.max(mse_results) / np.min(mse_results) > 100:
        ax2.set_yscale('log')
        ax2.set_ylabel('Mean Squared Error (MSE) - Log Scale', color=color2)

    plt.title('Quality vs. Compression Trade-off')
    fig.tight_layout()
    plt.show()

# --- MAIN INTERACTIVE LOOP ---
def main():
    original_audio, sample_rate, input_path = None, None, None
    reconstructed_audio, error_signal = None, None
    fft_coeffs, compressed_fft_coeffs = None, None
    keep_percentage = 10.0

    while True:
        if original_audio is None:
            print("\n" + "="*50 + "\n         FFT AUDIO COMPRESSION TOOL\n" + "="*50)
            wav_files = glob.glob('*.wav')
            if not wav_files:
                if input("No .wav files found. Generate a test file? (y/n): ").lower() != 'y': return
                input_path = generate_test_wav()
            else:
                print("\nFound .wav files:"); [print(f"  {i+1}. {f}") for i, f in enumerate(wav_files)]
                try:
                    choice = int(input(f"Choose a file (1-{len(wav_files)}): ")) - 1
                    if not 0 <= choice < len(wav_files): raise ValueError
                    input_path = wav_files[choice]
                except (ValueError, IndexError): print("Invalid choice."); continue
            try:
                sample_rate, original_audio = wavfile.read(input_path)
                if original_audio.ndim > 1: original_audio = original_audio.mean(axis=1).astype(original_audio.dtype)
                if original_audio.size == 0: raise ValueError("Audio file is empty.")
                print(f"Successfully loaded '{input_path}'")
            except Exception as e: print(f"Error: {e}. Please choose a valid .wav file."); original_audio = None; continue
        
        try:
            new_percentage_str = input(f"\nEnter percentage of frequencies to keep [{keep_percentage}%]: ").strip()
            if new_percentage_str: keep_percentage = float(new_percentage_str)
            if not 0 < keep_percentage <= 100: print("Invalid percentage. Must be between 0 and 100."); continue
        except ValueError: print("Invalid input. Please enter a number."); continue
        
        reconstructed_audio, fft_coeffs, compressed_fft_coeffs, ratio = compress_reconstruct(original_audio, sample_rate, keep_percentage)
        error_signal = (original_audio.astype(np.float64) - reconstructed_audio.astype(np.float64)).astype(original_audio.dtype)
        
        mse = calculate_mse(original_audio, reconstructed_audio)
        snr = calculate_snr(original_audio, reconstructed_audio)
        
        print(f"\n--- Quality Metrics ---")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB")
        
        reconstructed_filename = f"reconstructed_{keep_percentage:.1f}pct.wav"
        discarded_filename = f"discarded_noise_{keep_percentage:.1f}pct.wav"
        wavfile.write(reconstructed_filename, sample_rate, reconstructed_audio)
        wavfile.write(discarded_filename, sample_rate, error_signal)
        print(f"\n✅ Reconstructed audio saved to '{reconstructed_filename}'")
        print(f"✅ Discarded 'noise' saved to '{discarded_filename}'")

        while True:
            print("\n" + "-"*45); print("                 MENU"); print("-"*45)
            print(f"(File: {os.path.basename(input_path)} | Compression: {keep_percentage}%)")
            print("1. Play Original Audio")
            print("2. Play Reconstructed Audio")
            print("3. Play the Discarded Information ('Noise')")
            print("-------------------------------------------------")
            print("4. Show comparison plots")
            print("5. Plot Quality vs. Compression Analysis")
            print("6. Change compression level")
            print("7. Select a different audio file")
            print("8. Exit")
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == '1': play_audio(original_audio, sample_rate, "original")
            elif choice == '2': play_audio(reconstructed_audio, sample_rate, "reconstructed")
            elif choice == '3': play_audio(error_signal, sample_rate, "discarded information (noise)")
            elif choice == '4': visualize_all(original_audio, reconstructed_audio, fft_coeffs, compressed_fft_coeffs, sample_rate)
            elif choice == '5': plot_quality_analysis(original_audio, sample_rate)
            elif choice == '6': break
            elif choice == '7': original_audio = None; break
            elif choice == '8': print("Exiting. Goodbye!"); return
            else: print("Invalid choice.")

if __name__ == "__main__":
    main()