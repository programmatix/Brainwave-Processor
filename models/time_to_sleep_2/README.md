Predict time to sleep, and investigate settling period.

See also models.eeg_states.

Diary:
* Mar '25: Picking back up & adding SettlingInvestigation. With more data the model looks to have got more accurate and is only +/- 5m out now (MAE). 
** Update: after clipping abs and doing test/train split based on days not epochs, perf is awful. +/- 15m.  Trying a new discrete (classification) model.  Nope - also pretty useless.


# Specific dates

## Where model does very well
