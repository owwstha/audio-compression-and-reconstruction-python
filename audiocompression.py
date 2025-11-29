#%%

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import sounddevice as sd
import glob

def generate_test_wav(filename="test_audio.wav"):
    print(f"\'{filename}'")
    sample_rate = 44100
    duration = 3
    freq1, freq2 = 440, 880
    t = np.linspace(0., duration, int(sample_rate * duration))
    amp = np.iinfo(np.int16).max * 0.4
    data = amp * (np.sin(2. * np.pi * freq1 * t) + 0.5 * np.sin(2. * np.pi * freq2 * t))
    wavfile.write(filename, sample_rate, data.astype(np.int16))
    return filename

def compress_reconstruct(audio_data, sample_rate, keep_percent, verbose=True):
    if verbose:
        print(f"\Keeping {keep_percent:.2f}%.")
    
    og_length = len(audio_data)
    fft_coeffs = np.fft.rfft(audio_data.astype(np.float64))
    
    fft_mag = np.abs(fft_coeffs)
    
    if keep_percent >= 100:
        threshold = 0
    elif keep_percent <= 0:
        threshold = np.inf
    else:
        sorted_mag = np.sort(fft_mag)[::-1]
        coeffs_keep = int(len(sorted_mag) * keep_percent / 100)
        threshold = sorted_mag[coeffs_keep - 1]

    comp_fft_coeffs = fft_coeffs * (fft_mag >= threshold)
    
    og_non_zero = np.count_nonzero(fft_coeffs)
    comp_non_zero = np.count_nonzero(comp_fft_coeffs)
    if comp_non_zero == 0: comp_non_zero = 1
    comprsn_ratio = og_non_zero / comp_non_zero
    
    if verbose:
        print(f"Coefficient compression ratio: {comprsn_ratio:.2f} : 1")
    
    recon_signal = np.fft.irfft(comp_fft_coeffs, n=og_length)
    recon_audio = np.clip(recon_signal, -32767, 32767).astype(audio_data.dtype)
    
    return recon_audio, fft_coeffs, comp_fft_coeffs, comprsn_ratio

def play_audio(audio_data, sample_rate, description=""):
    sd.play(audio_data, sample_rate)
    try:
        input(f"Playing {description} audio")
        sd.stop()
    except KeyboardInterrupt:
        sd.stop()
        print("\nPlayback stopped by user.")
    except Exception as e:
        sd.stop()
        print(f"An error occurred: {e}")

def visualize_all(og_data, recon_data, fft_coeffs, comp_fft_coeffs, sample_rate):
    print("\nGenerating plots")
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('FFT Audio compression Analysis', fontsize=16)

    ax = axs[0, 0]
    NFFT_SIZE = 1024
    if len(og_data) > NFFT_SIZE:
        Pxx, freqs, bins, im = ax.specgram(og_data, Fs=sample_rate, NFFT=NFFT_SIZE, cmap='viridis')
        fig.colorbar(im, ax=ax).set_label('Intensity [dB]')
    else:
        ax.text(0.5, 0.5, 'Audio too short', ha='center', va='center')
    ax.set_title('Spectrogram of og Audio'); ax.set_xlabel('Time [s]'); ax.set_ylabel('Frequency [Hz]')

    freq_axis = np.fft.rfftfreq(len(og_data), d=1./sample_rate)
    def safe_plot(ax, x, y, **kwargs): min_len = min(len(x), len(y)); ax.semilogy(x[:min_len], np.abs(y[:min_len]), **kwargs)

    ax = axs[0, 1]; safe_plot(ax, freq_axis, fft_coeffs, color='blue'); ax.set_title('FFT of og Signal'); ax.set_xlabel('Frequency [Hz]'); ax.set_ylabel('Magnitude (log scale)'); ax.grid(True)
    ax = axs[1, 0]; safe_plot(ax, freq_axis, comp_fft_coeffs, color='orange'); ax.set_title('FFT after compression'); ax.set_xlabel('Frequency [Hz]'); ax.set_ylabel('Magnitude (log scale)'); ax.grid(True)
    ax = axs[1, 1]; fft_recon = np.fft.rfft(recon_data.astype(np.float64)); safe_plot(ax, freq_axis, fft_recon, color='green'); ax.set_title('FFT of recon Signal'); ax.set_xlabel('Frequency [Hz]'); ax.set_ylabel('Magnitude (log scale)'); ax.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

def calculate_mse(og, recon):
    return np.mean((og.astype(np.float64) - recon.astype(np.float64)) ** 2)

def calculate_snr(og, recon):
    noise = og.astype(np.float64) - recon.astype(np.float64)
    power_signal = np.mean(og.astype(np.float64) ** 2)
    power_noise = np.mean(noise ** 2)
    if power_noise == 0: return np.inf
    return 10 * np.log10(power_signal / power_noise)

def plot_quality_analysis(og_audio, sample_rate):
    print("\nRunning Quality vs. comprsn Analysis...")
    
    comprsn_levels = [95, 80, 60, 40, 20, 15, 10, 5, 2, 1]
    snr_results, mse_results, ratio_results = [], [], []

    for level in comprsn_levels:
        print(f"  Testing with {level}% of frequencies kept...")
        recon, _, _, ratio = compress_reconstruct(og_audio, sample_rate, level, verbose=False)
        snr_results.append(calculate_snr(og_audio, recon))
        mse_results.append(calculate_mse(og_audio, recon))
        ratio_results.append(ratio)
        
    print("\nAnalysis Complete")

    print("="*60)
    print(f"{'Keep (%)':<12} | {'Comp. Ratio':<15} | {'SNR (dB)':<12} | {'MSE':<15}")
    print("-"*60)
    for level, ratio, snr, mse in zip(comprsn_levels, ratio_results, snr_results, mse_results):
        print(f"{level:<12.1f} | {ratio:<15.2f} | {snr:<12.2f} | {mse:<15.2f}")
    print("="*60)
    print("\nGenerating plot...")

    fig, ax1 = plt.subplots(figsize=(12, 7))
    color1 = 'tab:blue'
    ax1.set_xlabel('Coefficient comprsn Ratio ')
    ax1.set_ylabel('Signal-to-Noise Ratio (SNR) in dB', color=color1)
    ax1.plot(ratio_results, snr_results, 'o-', color=color1, label='SNR')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, which='both', linestyle='--')
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Mean Squared Error (MSE)', color=color2)
    ax2.plot(ratio_results, mse_results, 's--', color=color2, label='MSE')
    ax2.tick_params(axis='y', labelcolor=color2)
    if np.max(mse_results) > 0 and np.min(mse_results) > 0 and np.max(mse_results) / np.min(mse_results) > 100:
        ax2.set_yscale('log')
        ax2.set_ylabel('Mean Squared Error (MSE) - Log Scale', color=color2)

    plt.title('Quality vs. comprsn Trade-off')
    fig.tight_layout()
    plt.show()

def main():
    og_audio, sample_rate, input_path = None, None, None
    recon_audio, error_signal = None, None
    fft_coeffs, comp_fft_coeffs = None, None
    keep_percent = 10.0

    while True:
        if og_audio is None:
            print("\n" + "="*50 + "\n compression tool\n" + "="*50)
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
                sample_rate, og_audio = wavfile.read(input_path)
                if og_audio.ndim > 1: og_audio = og_audio.mean(axis=1).astype(og_audio.dtype)
                if og_audio.size == 0: raise ValueError("Audio file is empty.")
                print(f"Successfully loaded '{input_path}'")
            except Exception as e: print(f"Error: {e}. Please choose a valid .wav file."); og_audio = None; continue
        
        try:
            new_percent_str = input(f"\nEnter percent of frequencies to keep [{keep_percent}%]: ").strip()
            if new_percent_str: keep_percent = float(new_percent_str)
            if not 0 < keep_percent <= 100: print("Invalid percent. Must be between 0 and 100."); continue
        except ValueError: print("Invalid input. Please enter a number."); continue
        
        recon_audio, fft_coeffs, comp_fft_coeffs, ratio = compress_reconstruct(og_audio, sample_rate, keep_percent)
        error_signal = (og_audio.astype(np.float64) - recon_audio.astype(np.float64)).astype(og_audio.dtype)
        
        mse = calculate_mse(og_audio, recon_audio)
        snr = calculate_snr(og_audio, recon_audio)
        
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB")
        
        recon_filename = f"recon_{keep_percent:.1f}pct.wav"
        discarded_filename = f"discarded_noise_{keep_percent:.1f}pct.wav"
        wavfile.write(recon_filename, sample_rate, recon_audio)
        wavfile.write(discarded_filename, sample_rate, error_signal)
        print(f"\n recon audio saved to '{recon_filename}'")
        print(f" Discarded 'noise' saved to '{discarded_filename}'")

        while True:
            print("\n" + "-"*45); print("                 MENU"); print("-"*45)
            print(f"(File: {os.path.basename(input_path)} | comprsn: {keep_percent}%)")
            print("1. Play og Audio")
            print("2. Play recon Audio")
            print("3. Play the Discarded Information")
            print("4. Show comparison plots")
            print("5. Plot Quality vs. comprsn Analysis")
            print("6. Change comprsn level")
            print("7. Select a different audio file")
            print("8. Exit")
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == '1': play_audio(og_audio, sample_rate, "og")
            elif choice == '2': play_audio(recon_audio, sample_rate, "recon")
            elif choice == '3': play_audio(error_signal, sample_rate, "discarded information (noise)")
            elif choice == '4': visualize_all(og_audio, recon_audio, fft_coeffs, comp_fft_coeffs, sample_rate)
            elif choice == '5': plot_quality_analysis(og_audio, sample_rate)
            elif choice == '6': break
            elif choice == '7': og_audio = None; break
            elif choice == '8': print("Exiting. Goodbye!"); return
            else: print("Invalid choice.")

if __name__ == "__main__":
    main()
# %%
