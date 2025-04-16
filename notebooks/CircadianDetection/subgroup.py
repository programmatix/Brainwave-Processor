import pysubgroup as ps

# Define target and search space
target = ps.NumericTarget('LEP_time - shower_time')
searchspace = ps.create_selectors(df_lep, ignore=['LEP_time', 'shower_time'])

# Find interesting subgroups
task = ps.SubgroupDiscoveryTask(df_lep, target, searchspace, result_set_size=10, depth=3)
result = ps.BeamSearch().execute(task)


df = df_lep.copy()
# replace ":" with "_"
df.columns = df.columns.str.replace(':', '_')
