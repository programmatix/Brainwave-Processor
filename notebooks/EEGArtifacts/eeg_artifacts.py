import numpy as np
import csv
import pandas as pd
DEFAULT_THRESHOLD=60
DEFAULT_MIN_DURATION=50
DEFAULT_MERGE_GAP=400
DEFAULT_MIN_SECTION_LENGTH=400


def get_artifact_sections(data, threshold=DEFAULT_THRESHOLD, min_duration=DEFAULT_MIN_DURATION, merge_gap=DEFAULT_MERGE_GAP, min_section_length=DEFAULT_MIN_SECTION_LENGTH):
    # Find where data exceeds threshold
    artifacts = np.abs(data) > threshold
    
    # Find continuous segments
    artifact_sections = []
    start = None
    
    for i in range(len(artifacts)):
        if artifacts[i] and start is None:
            start = i
        elif not artifacts[i] and start is not None:
            if (i - start) >= min_duration:
                artifact_sections.append((start, i))
            start = None
    
    if start is not None and (len(artifacts) - start) >= min_duration:
        artifact_sections.append((start, len(artifacts)))
    
    # Merge sections that are close together
    merged_sections = []
    if artifact_sections:
        merged_sections = [artifact_sections[0]]
        for current in artifact_sections[1:]:
            prev = merged_sections[-1]
            if current[0] - prev[1] <= merge_gap:
                merged_sections[-1] = (prev[0], current[1])
            else:
                merged_sections.append(current)
        
        # Filter out sections that are too short
        merged_sections = [(start, end) for start, end in merged_sections 
                         if (end - start) >= min_section_length]

    return merged_sections, artifact_sections


def save_artifacts_to_csv(sections, output_file):
    print(f"Saving artifacts to {output_file}")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start', 'end'])
        for start, end in sections:
            writer.writerow([
                int(start),
                int(end)
            ])

import os
import io
import convert
import contextlib

def process_artifacts(root, dir_name, force=False):
    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
        input_fif_file = os.path.join(root, dir_name, "raw.fif")
        artifacts_csv_path = os.path.join(root, dir_name, "raw.artifacts.csv")

        try:
            print(f"Processing file: " + input_fif_file, flush=True)
            if not os.path.exists(input_fif_file):
                return None, False, output_buffer.getvalue(), "File not found " + input_fif_file
            
            if not os.path.exists(artifacts_csv_path) or force:

                raw, input_file_without_ext, mne_filtered = convert.load_mne_file(print, input_fif_file)
                data = mne_filtered.get_data(units=dict(eeg="uV"))
                channels = mne_filtered.ch_names
                
                all_artifacts = []
                
                for i in range(data.shape[0]):
                    channel = channels[i]
                    data_channel = data[i, :]
                    sections, artifact_sections = get_artifact_sections(data_channel)
                    
                    for start, end in sections:
                        all_artifacts.append({
                            'channel': channel,
                            'start': int(start),
                            'end': int(end)
                        })
                
                # Create DataFrame and save to single CSV
                artifacts_df = pd.DataFrame(all_artifacts)
                artifacts_df.to_csv(artifacts_csv_path, index=False)
                
                return artifacts_df, False, output_buffer.getvalue(), True
            return pd.read_csv(artifacts_csv_path), True, output_buffer.getvalue(), True
        except Exception as e:
            return None, False, output_buffer.getvalue(), "Error: " + str(e)

