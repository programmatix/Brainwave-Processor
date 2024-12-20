import sys
from IPython.display import display, Markdown

def exit_early(reason: str = None):
    # Papermill treats as an early-return signal:
    # https://github.com/nteract/papermill/pull/449
    # https://stackoverflow.com/questions/53847734/is-it-possible-to-halt-execution-of-a-jupyter-notebook-using-papermill
    # Downside: it does make the TDQM progress bar red (error)
    sys.exit(0)
