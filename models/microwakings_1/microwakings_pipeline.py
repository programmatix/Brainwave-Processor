import os

import pandas as pd

import convert
from models.microwakings_1 import microwakings1
from models.microwakings_1.microwakings1 import PerFile
import traceback

from memory import garbage_collect

def cached_microwakings_pipeline(log, input_file: str, post_yasa_df: pd.DataFrame, force: bool = False):
    input_file_without_ext = os.path.splitext(input_file)[0]
    cached = input_file_without_ext + ".microwakings.csv"

    def regenerate():
        out = microwakings_pipeline(log, input_file, post_yasa_df)
        return out, False

    if os.path.exists(cached):
        log("Loading cached file " + cached)
        out = pd.read_csv(cached)

        if force:
            log("Forced rebuild")
            return regenerate()

        return out, True
    else:
        log(f"No cached file {cached}, rebuilding")
        return regenerate()


def microwakings_pipeline(log, input_file: str, post_yasa_df: pd.DataFrame):
    log("Loading MNE file " + input_file)
    raw, input_file_without_ext, mne_filtered = convert.load_mne_file(log, input_file)

    # Run current best microwakings model - needs to be done after adding Main channel
    garbage_collect(log)
    log("Running microwakings model")
    try:
        microwakings_model = microwakings1.load_model()
        pf = PerFile(None, mne_filtered, post_yasa_df, input_file_without_ext)
        pf.prepare_model_data(microwakings1.RESAMPLING_RATE, False)
        return microwakings1.predict_file(log, microwakings_model, pf)
    except Exception as e:
        log("Error running microwakings model: " + str(e))
        log(traceback.format_exc())
        raise e
