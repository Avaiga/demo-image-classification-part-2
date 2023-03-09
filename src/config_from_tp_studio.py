from main_functions import *
from taipy import Config
import taipy as tp 

Config.load('built_with_tp_studio.toml')
scenario_cfg = Config.scenarios['testing_scenario']

tp.Core().run()
main_scenario = tp.create_scenario(scenario_cfg)
tp.submit(main_scenario)







