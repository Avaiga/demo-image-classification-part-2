from main_functions import *
from taipy import Config
import taipy as tp 

#######################################################################################################
##############################################PIPELINE 1###############################################
#######################################################################################################

###TASK 1.1: Building the base model
#input dn
loss_fn_cfg = Config.configure_data_node("loss_fn", default_data='categorical_crossentropy')
#output dn
base_model_cfg = Config.configure_generic_data_node("base_model", 
                                                    read_fct=tf_read, read_fct_params=('models/base_model',),
                                                    write_fct=tf_write, write_fct_params=('models/base_model',))
#task
BUILD_CNN_BASE_cfg = Config.configure_task("BUILD_CNN_BASE",
                                    initialize_model,
                                    loss_fn_cfg,
                                    base_model_cfg)

###TASK 1.2: Initial training with a fixed number of epochs
#input dn
initial_n_epochs_cfg = Config.configure_data_node("initial_n_epochs", default_data=30)
#output dn
initial_train_perf_cfg = Config.configure_data_node("initial_train_perf")
trained_initial_model_cfg = Config.configure_generic_data_node("trained_initial_model", 
                                                    read_fct=tf_read, read_fct_params=('models/trained_initial_model',),
                                                    write_fct=tf_write, write_fct_params=('models/trained_initial_model',))
#task
INITIAL_TRAIN_cfg = Config.configure_task("INITIAL_TRAIN",
                                    initial_model_training,
                                    [initial_n_epochs_cfg, base_model_cfg],
                                    [initial_train_perf_cfg, trained_initial_model_cfg])
#pipeline
pipeline_1_cfg = Config.configure_pipeline("pipeline_1",
                                               [BUILD_CNN_BASE_cfg,
                                                INITIAL_TRAIN_cfg])

#######################################################################################################
##############################################PIPELINE 2###############################################
#######################################################################################################

###TASK 2.1: Merge train with a chosen number of epochs (training + validation set as training)
#input dn
optimal_n_epochs_cfg = Config.configure_data_node("optimal_n_epochs", default_data=13)
#output dn
merged_train_perf_cfg = Config.configure_data_node("merged_train_perf")
merged_trained_model_cfg = Config.configure_generic_data_node("merged_trained_model", 
                                                    read_fct=tf_read, read_fct_params=('models/merged_trained_model',),
                                                    write_fct=tf_write, write_fct_params=('models/merged_trained_model',))
#task
MERGED_TRAIN_cfg = Config.configure_task("MERGED_TRAIN",
                                    merged_train,
                                    [optimal_n_epochs_cfg, base_model_cfg],
                                    [merged_train_perf_cfg, merged_trained_model_cfg])


###TASK 2.2: Make a prediction from an image path
#input dn: the trained model datanode, already set up
image_path_dn_cfg = Config.configure_data_node("image_path_dn", default_data="test_images/dog.jpg") 
#output dn
prediction_cfg = Config.configure_data_node("image_prediction")
#task
IMAGE_PREDICT_cfg = Config.configure_task("IMAGE_PREDICT", predict_image,
                                          [image_path_dn_cfg, merged_trained_model_cfg],
                                          [prediction_cfg])
#pipeline
pipeline_2_cfg = Config.configure_pipeline("pipeline_2",
                                               [MERGED_TRAIN_cfg, 
                                                IMAGE_PREDICT_cfg])


#######################################################################################################
##############################################Scenario#################################################
#######################################################################################################
scenario_cfg = Config.configure_scenario("testing_scenario",
                                         [pipeline_1_cfg, pipeline_2_cfg])

tp.Core().run()
main_scenario = tp.create_scenario(scenario_cfg)
tp.submit(main_scenario)
Config.export("tpcore.toml")





