raw.with_features.csv:
The final stage, with all data glued together. 
It would ne nice to get rid of it, as it's big, but many things are using it.
It doesn't have all data. It only has Main, not the individual channels (if remove_non_main_eeg, which is default). 
- Can't recall if that was intentional.. It is tripping epoch viewer up. Probably just to save bytes.
- Making remove_non_main_eeg non-default, to help epoch viewer.