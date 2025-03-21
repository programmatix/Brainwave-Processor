import numpy as np
import csv
import pandas as pd
from datetime import datetime

# N3 can spike up to around 150
DEFAULT_THRESHOLD=150
DEFAULT_MIN_DURATION=50
DEFAULT_MERGE_GAP=400
# 400 was too short, included some N3
DEFAULT_MIN_SECTION_LENGTH=500
DEFAULT_MAX_AMPLITUDE=1000




def get_artifact_sections(data, amplitude_threshold=DEFAULT_THRESHOLD, min_artifact_duration=DEFAULT_MIN_DURATION, 
                         max_merge_gap=DEFAULT_MERGE_GAP, min_section_length=DEFAULT_MIN_SECTION_LENGTH, max_amplitude=DEFAULT_MAX_AMPLITUDE, verbose=False):
    """
    Identify artifact sections in EEG data based on amplitude thresholds and temporal constraints.
    
    Parameters:
    -----------
    data : array-like
        The EEG data to analyze for artifacts
    amplitude_threshold : float
        The amplitude threshold above which a data point is considered an artifact
    min_artifact_duration : float
        The minimum duration (in samples) for an individual artifact to be considered valid
    max_merge_gap : float
        The maximum gap (in samples) between artifacts to merge them into a single section
    min_section_length : float
        The minimum length (in samples) for a section to be retained
    max_amplitude : float
        The maximum amplitude threshold. Any section containing an amplitude above this will always be kept
        regardless of other parameters like min_section_length
    verbose : bool
        If True, print detailed logging information during processing
        
    Returns:
    --------
    tuple
        (merged_sections, artifact_sections)
        - merged_sections: list of (start, end) tuples for merged artifact sections
        - artifact_sections: list of (start, end) tuples for individual artifact sections
    """
    if verbose:
        print(f"Starting artifact detection with parameters:")
        print(f"  - Data length: {len(data)} samples")
        print(f"  - Amplitude threshold: {amplitude_threshold}")
        print(f"  - Min artifact duration: {min_artifact_duration} samples")
        print(f"  - Max merge gap: {max_merge_gap} samples")
        print(f"  - Min section length: {min_section_length} samples")
        print(f"  - Max amplitude: {max_amplitude}")
    
    # Find artifact sections
    artifact_sections = []
    start = None
    
    for i in range(len(data)):
        if abs(data[i]) > amplitude_threshold:
            if start is None:
                start = i
        elif start is not None:
            if i - start >= min_artifact_duration:
                artifact_sections.append((start, i))
                if verbose:
                    print(f"Abnormal amplitude period detected: samples {start}-{i} (duration: {i-start})")
            start = None
    
    # Handle case where artifact extends to the end of the data
    if start is not None and len(data) - start >= min_artifact_duration:
        artifact_sections.append((start, len(data)))
        if verbose:
            print(f"Abnormal amplitude period at end of data: samples {start}-{len(data)} (duration: {len(data)-start})")
    
    if verbose:
        print(f"Found {len(artifact_sections)} individual abnormal amplitude periods")
    
    # Track sections with amplitudes exceeding max_amplitude
    high_amplitude_sections = []
    for start, end in artifact_sections:
        section_data = data[start:end]
        if np.max(np.abs(section_data)) > max_amplitude:
            high_amplitude_sections.append((start, end))
            if verbose:
                print(f"Section {start}-{end} contains amplitude exceeding max_amplitude ({max_amplitude})")
    
    # Merge sections that are close together
    merged_sections = []
    if artifact_sections:
        if verbose:
            print("Starting section merging process...")
        
        current_start, current_end = artifact_sections[0]
        
        for start, end in artifact_sections[1:]:
            if start - current_end <= max_merge_gap:
                # Merge with the current section
                if verbose:
                    print(f"Merging sections: ({current_start}-{current_end}) and ({start}-{end}), gap: {start-current_end}")
                current_end = end
            else:
                # Check if the current merged section meets the minimum length requirement
                # or contains amplitudes exceeding max_amplitude
                section_data = data[current_start:current_end]
                exceeds_max_amplitude = np.max(np.abs(section_data)) > max_amplitude
                
                if exceeds_max_amplitude or current_end - current_start >= min_section_length:
                    merged_sections.append((current_start, current_end))
                    if verbose:
                        if exceeds_max_amplitude:
                            print(f"Added merged section: {current_start}-{current_end} (exceeds max amplitude)")
                        else:
                            print(f"Added merged section: {current_start}-{current_end} (length: {current_end-current_start})")
                else:
                    if verbose:
                        print(f"Rejected short merged section: {current_start}-{current_end} (length: {current_end-current_start})")
                # Start a new section
                current_start, current_end = start, end
        
        # Add the last section if it meets the minimum length requirement
        # or contains amplitudes exceeding max_amplitude
        section_data = data[current_start:current_end]
        exceeds_max_amplitude = np.max(np.abs(section_data)) > max_amplitude
        
        if exceeds_max_amplitude or current_end - current_start >= min_section_length:
            merged_sections.append((current_start, current_end))
            if verbose:
                if exceeds_max_amplitude:
                    print(f"Added final merged section: {current_start}-{current_end} (exceeds max amplitude)")
                else:
                    print(f"Added final merged section: {current_start}-{current_end} (length: {current_end-current_start})")
        elif verbose:
            print(f"Rejected final short merged section: {current_start}-{current_end} (length: {current_end-current_start})")
    
    # Filter out short sections that don't exceed max_amplitude
    final_merged_sections = []
    for start, end in merged_sections:
        section_data = data[start:end]
        exceeds_max_amplitude = np.max(np.abs(section_data)) > max_amplitude
        
        if exceeds_max_amplitude or end - start >= min_section_length:
            final_merged_sections.append((start, end))
        elif verbose:
            print(f"Filtered out section: {start}-{end} (length: {end-start}, doesn't exceed max amplitude)")

    if verbose:
        print(f"Final results: {len(final_merged_sections)} merged sections, {len(artifact_sections)} individual abnormal amplitude periods")
    
    return final_merged_sections, artifact_sections


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


def remove_artifacts(data, artifacts_df, epoch_idx, samples_per_epoch = 7500):
    data_artifact_removed = data.copy()
    start_sample = epoch_idx * samples_per_epoch    
    end_sample = start_sample + samples_per_epoch
    for _, row in artifacts_df.iterrows():
        #print(f"row['start']={row['start']}, row['end']={row['end']}, start_sample={start_sample}, end_sample={end_sample}")
        if start_sample >= row['start'] and start_sample <= row['end'] or end_sample >= row['start'] and end_sample <= row['end']:
            start_removing_from = max(start_sample, row['start'])
            end_removing_to = min(end_sample, row['end'])
            print(f"Removing artifacts between {row['start']} and {row['end']} start_sample={start_sample}, end_sample={end_sample}")
            data_artifact_removed[start_removing_from - start_sample:end_removing_to - start_sample] = np.nan
    removed_samples = np.isnan(data_artifact_removed).sum()
    return data_artifact_removed, removed_samples


def epochs_containing_artifacts(artifacts_df, samples_per_epoch=7500):
    epochs_in_artifacts = set()
    for _, row in artifacts_df.iterrows():
        # Calculate the first epoch that contains the start of the artifact
        start_epoch = row['start'] // samples_per_epoch
        
        # Calculate the last epoch that contains the end of the artifact
        end_epoch = row['end'] // samples_per_epoch
        
        # Add all epochs that contain any part of the artifact
        for i in range(start_epoch, end_epoch + 1):
            epochs_in_artifacts.add(i)
            
    return epochs_in_artifacts

import os
import io
import convert
import contextlib

force_if_older_than = datetime(2025, 3, 11, 10, 0, 0)

def process_artifacts(root, dir_name, force=False, verbose=False):
    def regenerate():
        return artifacts_pipeline(print, input_fif_file, artifacts_csv_path, force, verbose)

    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
        input_fif_file = os.path.join(root, dir_name, "raw.fif")
        artifacts_csv_path = os.path.join(root, dir_name, "raw.artifacts.csv")

        try:
            print(f"Processing file: " + input_fif_file, flush=True)
            if not os.path.exists(input_fif_file):
                return None, False, output_buffer.getvalue(), "File not found " + input_fif_file

            if not os.path.exists(artifacts_csv_path) or force:
                print(f"Regenerating artifacts as file doesn't exist or force is true")
                artifacts_df = regenerate()                
                return artifacts_df, False, output_buffer.getvalue(), True

            modification_time = os.path.getmtime(artifacts_csv_path)
            modification_date = datetime.fromtimestamp(modification_time)

            if modification_date < force_if_older_than:
                print(f"Regenerating artifacts as older than {force_if_older_than}")
                artifacts_df = regenerate()                
                return artifacts_df, False, output_buffer.getvalue(), True

            return pd.read_csv(artifacts_csv_path), True, output_buffer.getvalue(), True
        except Exception as e:
            if verbose:
                print(f"Error processing {input_fif_file}: {str(e)}")
            return None, False, output_buffer.getvalue(), "Error: " + str(e)


def artifacts_pipeline(log, input_fif_file: str, artifacts_csv_path: str, force: bool = False, verbose: bool = False):
    raw, input_file_without_ext, mne_filtered = convert.load_mne_file(print, input_fif_file)
    data = mne_filtered.get_data(units=dict(eeg="uV"))
    channels = mne_filtered.ch_names
    
    if verbose:
        print(f"Loaded data with {len(channels)} channels and {data.shape[1]} samples")
    
    all_artifacts = []
    
    for i in range(data.shape[0]):
        channel = channels[i]
        data_channel = data[i, :]
        
        if verbose:
            print(f"\nProcessing channel {i+1}/{data.shape[0]}: {channel}")
        
        sections, artifact_sections = get_artifact_sections(data_channel, verbose=verbose)
        
        for start, end in sections:
            all_artifacts.append({
                'channel': channel,
                'start': int(start),
                'end': int(end)
            })
        
        if verbose:
            print(f"Found {len(sections)} artifact sections in channel {channel}")
    
    # Create DataFrame and save to single CSV
    artifacts_df = pd.DataFrame(all_artifacts)
    artifacts_df.to_csv(artifacts_csv_path, index=False)

    return artifacts_df

