import sys
import os
import subprocess
import parselmouth
import numpy as np
import csv

print("Running Python from:", sys.executable)

# === 1. Paths ===
AUDIO_FOLDER = "/Users/claireott/Desktop/Vocal_Pitch/Wicklow-Wexford"  # Change this per constituency
CONSTITUENCY = os.path.basename(AUDIO_FOLDER)  # e.g., 'Carlow-Kilkenny'
TEMP_WAV = os.path.join(os.getcwd(), "temp.wav")
OUTPUT_CSV = os.path.join(os.getcwd(), "pitch_results.csv")

# === 2. Convert .m4a to .wav ===
def convert_to_wav(input_path, output_path):
    command = ['ffmpeg', '-y', '-i', input_path, '-ac', '1', '-ar', '16000', output_path]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0 and os.path.exists(output_path)

# === 3. Extract pitch stats ===
def extract_pitch_stats(wav_path, min_pitch=75, max_pitch=500):
    try:
        snd = parselmouth.Sound(wav_path)
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=min_pitch, pitch_ceiling=max_pitch)
        pitch_values = pitch.selected_array['frequency']
        voiced = pitch_values[pitch_values > 0]

        if len(voiced) == 0:
            return None

        return {
            "mean_pitch": np.mean(voiced),
            "min_pitch": np.min(voiced),
            "max_pitch": np.max(voiced)
        }
    except Exception as e:
        print(f"  Error processing {wav_path}: {e}")
        return None

# === 4. Append to CSV ===
file_exists = os.path.exists(OUTPUT_CSV)

with open(OUTPUT_CSV, mode='a', newline='') as csvfile:
    fieldnames = ['speaker', 'constituency', 'mean_pitch_hz', 'min_pitch_hz', 'max_pitch_hz']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    if not file_exists:
        writer.writeheader()

    for filename in os.listdir(AUDIO_FOLDER):
        if filename.lower().endswith(".m4a"):
            full_audio_path = os.path.join(AUDIO_FOLDER, filename)

            # Capitalize each name part
            raw_name = os.path.splitext(filename)[0]
            speaker = ' '.join(part.capitalize() for part in raw_name.split('_'))

            print(f"Processing {speaker} in {CONSTITUENCY}...")

            if convert_to_wav(full_audio_path, TEMP_WAV):
                stats = extract_pitch_stats(TEMP_WAV)

                if stats:
                    writer.writerow({
                        'speaker': speaker,
                        'constituency': CONSTITUENCY,
                        'mean_pitch_hz': round(stats["mean_pitch"], 2),
                        'min_pitch_hz': round(stats["min_pitch"], 2),
                        'max_pitch_hz': round(stats["max_pitch"], 2)
                    })
                    print(f"  Done: mean pitch = {stats['mean_pitch']:.2f} Hz")
                else:
                    print("  No voiced speech detected.")
            else:
                print(f"  Conversion failed for {filename}")

# === 5. Cleanup ===
if os.path.exists(TEMP_WAV):
    os.remove(TEMP_WAV)

print(f"Finished processing folder: {CONSTITUENCY}")
print(f"Results saved to: {OUTPUT_CSV}")
