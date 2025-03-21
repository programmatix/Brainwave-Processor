from datetime import datetime, timedelta
import dateutil.parser
import os
# '2024-09-18 06:54:44.539987087+01:00' -> '2024-09-17'
def day_and_night_of(timestamp_str: str) -> str:
    # Parse the timestamp
    dt = dateutil.parser.parse(timestamp_str)
    
    # If time is before 11am, use the previous day
    if dt.hour < 11:
        dt = dt - timedelta(days=1)
    
    # Format as YYYY-MM-DD
    return dt.strftime('%Y-%m-%d')


def day_and_night_of_dir(input_dir: str, day_and_night: str) -> str:
    dirs = next(os.walk(input_dir))[1]
    
    options = []

    for idx, dir_name in enumerate(dirs):
        if day_and_night in dir_name:
            options.append(dir_name)

    if len(options) == 0:
        raise ValueError(f"No directory found for {day_and_night}")
    elif len(options) > 1:
        raise ValueError(f"Multiple directories found for {day_and_night}")
    return os.path.join(input_dir, options[0]), options[0]

