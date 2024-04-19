# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
strings module
==============

Module used to control in one place all the strings of the program

"""


class s:

    # Helpers
    verbose_help = "Define the level of verbosity in the logs"
    silent_help = "Flag to deactivate the output to the STDOUT"
    csv_help = "Flag to activate the csv output style"

    # default conf
    default_conf = "default_ini"

    # IO keys
    io_name_key = "name"
    io_type_key = "type"
    io_subtype_key = "subtype"
    io_path_key = "path"
    io_exists_key = "exists"

    # Types of IO objects
    folder_obj_types = ["folder", "direcotry"]
    file_obj_types = ["file"]
    sub_obj_type = ["input_data"]

    # Data objects
    data_key = "data"

    # General data container
    data_path = "datasets_path"
    input_data = "input_data"
    positive_file = "positives"
    negative_file = "negatives"
    half_file = "halfs"

    # Results containers
    results_path = "result_path"
    pp_path = "pp_analysis"
    ms_path = "model_analysis"

    # Checkpoints containers
    checkpoints_path = "checkpoint_path"
    tensorboard_path = "tensorboard_path"
    csv_path = "csv_path"

    # Pkl containers
    pkl_path = "pkl_path"

    # Logs
    log_path = "log_path"
    log_file = "log_file"

    # PreProcessing keys
    pp_actions = "actions"
    pp_flow_name = "flow_name"
    goods_wdw = "goods_windows"
    goods_trg = "goods_targets"
    goods_cls = "goods_classes"
    bads_wdw = "bads_windows"
    bads_trg = "bads_targets"
    bads_cls = "bads_classes"
    grays_wdw = "grays_windows"
    grays_trg = "grays_targets"
    grays_cls = "grays_classes"
    pkl_training = "training_normalized"
    pkl_validation = "validation_normalized"
    pkl_test = "test_normalized"
    pkl_gray = "gray_normalized"
    goods_wdw_norm = "goods_windows_normalized"
    bads_wdw_norm = "bads_windows_normalized"
    grays_wdw_norm = "grays_windows_normalized"
    gray_post_train_portion = "gray_post_train_portion"
    gray_in_train_portion = "gray_in_train_portion"

    # Environment parameters
    param_value = "value"
    param_generic_type = "generic"
    param_PreProcessing_type = "preprocessing"
    param_Environment_type = "environment"
    param_numpyRng_type = "numpyRng"
    param_action_type = "action"
    param_action_args = "args"
    param_action_kwargs = "kwargs"
    param_experiment_type = "experiment"
    param_model_type = "model"
    param_reinforce_model_type = "reinforcementModel"
    param_embedding_type = "embedding"
    param_layer_type = "layer"
    param_metric_type = "metric"
    param_loss_type = "loss"
    param_lossParam_type = "lossParam"
    param_callback_type = "callback"
    param_flow_type = "flow"
    param_fitmethod_type = "fitMethod"
    param_study_type = "study"
    param_grayEvolution_type = "grayEvolution"
    param_reward_function_type = "rewardFunction"
    param_RLPerfEval_type = "RLPerfEval"
    param_RLObsPP_type = "RLObservationPP"
    param_RLActPolicy_type = "RLActionPolicy"
    param_RLEnvExitFunction_type = "RLEnvExitFunction"
    param_types = [param_generic_type, param_PreProcessing_type, param_Environment_type, param_numpyRng_type, param_action_type, param_experiment_type, param_model_type, param_embedding_type, param_layer_type, param_metric_type, param_loss_type, param_callback_type, param_flow_type, param_fitmethod_type, param_study_type, param_grayEvolution_type, param_lossParam_type, param_reinforce_model_type, param_reward_function_type, param_RLPerfEval_type, param_RLObsPP_type, param_RLActPolicy_type, param_RLEnvExitFunction_type]
    param_args_ast = [param_action_type, param_embedding_type, param_layer_type, param_metric_type, param_loss_type, param_callback_type, param_flow_type, param_fitmethod_type, param_grayEvolution_type, param_lossParam_type, param_reward_function_type, param_RLPerfEval_type, param_RLObsPP_type, param_RLActPolicy_type, param_RLEnvExitFunction_type]
    env_par = "environment_settings"
    numpy_rng = "numpy_rng"
    env_rng = "rng"
    env_tf_rng = "tf_rng"

    # Data processing keys parameters
    data_par = "data_pre_processing"
    data_object = "object"
    data_override = "override"
    data_limit = "limit"
    data_window = "window_size"
    data_vmaf_threshold = "vmaf_threshold"
    data_train_prt = "train_portion"
    data_val_prt = "val_portion"
    data_features = "features"
    data_drop_features = "drop_features"

    # Data proportioning
    prp_par = "data_proportioning"
    prp_special = "special_options"
    gray_inference = "gray_inference"
    gray_anchor_only = "gray_anchor"
    prp_cons = "consideration"
    prp_pres = "balancing"

    # Callbacks parameters
    callbacks_par = "callBacks_settings"
    cb_swap = "swap_cb"
    swap_function = "function"
    swap_ep_limit = "epoch_limit"
    cb_earlyStopping = "EarlyStopping"
    es_patience = "patience"
    es_delta = "min_delta"
    es_restore = "restore_best_weights"
    cb_ModelCheckpoint = "ModelCheckpoint"
    mc_weights_only = "save_weights_only"
    mc_save_freq = "save_freq"

    # metrics parameters
    metrics_par = "metrics_settings"
    metrics_functions = "functions"

    # Loss functions parameters
    loss_par = "loss_settings"
    loss_functions = "loss_functions"
    loss_variables = "loss_variables"
    kick_str = "kick_strength"
    sigmoid_range = "sigmoid_range"
    sigmoid_value = "sigmoid_value"
    cv_thr = "cv_threshold"

    # Model settings
    model_par = "model_settings"
    activation_key = "activation"
    n_nodes_key = "n_nodes"
    adam_strength_key = "adam"
    ml_eagerly = "run_eagerly"

    # Reinforcement learning settings
    rl_par = "reinforcement_learning_settings"
    rl_environment = "environment"
    rl_actuator = "actuator"
    rl_active = "active"
    rl_adam = "learning_rate"
    rl_episodes = "episodes"
    rl_steps = "steps"
    rl_actions = "actions"
    rl_inputs = "inputs"
    rl_nnodes = "num_nodes"
    rl_gamma = "gamma"
    rl_delta = "delta"
    rl_skip = "skip"
    rl_infer_episodes = "inference_episodes"
    rl_sigma = "sigma"

    # Experiments keys parameters
    fitMethod = "fitMethod"
    experiment_key = "experiment_settings"
    rng_key = "rng"
    tf_rng_key = "tf_rng"
    bf_key = "bf_size"
    train_dst_key = "train_portion"
    val_dst_key = "val_portion"
    shape_key = "shape"
    epochs_key = "epochs"
    epochs_switch_loss = "epochs_switch_loss"
    appendix_key = "appendix"
    margin_key = "margin"
    batch_size_key = "batch_size"
    prefetch_key = "prefetch"
    patience_key = "patience"
    min_delta_key = "delta"
    window_size = "window_size"
    exp_vmaf_flag = "vmaf_flag"
    features = "features"
    slimit = "slimit"
    vmaf_threshold = "vmaf_threshold"
    exp_input_dataset = "input_dataset"
    exp_verbose = "verbose"
    exp_sp_metric = "sp_metric"
    gray_inf_rep = "gray_inference_repetitions"
    grayEvMethod = "grayEvolutionMethod"

    # GrayFitMethods
    gray_default = "grays"
    gray_afterFit = "graysAfterFit"
    gray_afterFit_netReset = "graysAfterFit_netReset"
    gray_checkpoint = "graysAfterFit_modelCheckpoint"
    gray_generateEmb = "graysAfterFit_generateEmbeddings"
    rlTrnFrz_graysAfterFit = "RLTrnFrz_graysAfterFit"
    rlTrnFrz_external_graysAfterFit = "RLTrnFrz_External_graysAfterFit"
    rlTrnFrz_external_AfterTrivial = "RLTrnFrz_External_AfterTrivial"
    all_grays = [gray_default, gray_afterFit, gray_afterFit_netReset, gray_checkpoint, gray_generateEmb, rlTrnFrz_graysAfterFit, rlTrnFrz_external_graysAfterFit, rlTrnFrz_external_AfterTrivial]
    emb_grays = [gray_generateEmb]
    rl_training = [rlTrnFrz_graysAfterFit, rlTrnFrz_external_graysAfterFit, rlTrnFrz_external_AfterTrivial]

    # Exp parameters handler groups
    expph_grouping_key = "group_generator_keys"
    expph_triplet_gen = "triplet_generator_keys"

    # Plot conf
    plot_key = "plot_conf"
    plot_format = "plot_format"
    plot_features = "plot_features"
    network_plot_features = "network_plot_features"
    history_xlim = "history_xlim"
    history_ylim = "history_ylim"

    # Dataset
    main_group_column = "exp_id"
    seconds_column = "second"
    expectation = "expectation"
    group_column = "group"
    origin_column_name = "From"
    nn_column = "Network"
    index_col = [main_group_column, seconds_column]

    vmaf_column = "vmaf"
    vmafXsecond_column = "vmafXsecond"
    no_feature_col = [vmaf_column, vmafXsecond_column]

    # Log utils
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Pkl save
    group_df_pkl = "groups_df"
    last_layer_test = "tsne-resulting-df-{}"

    # Dataframes
    p_origin = "positives"
    n_origin = "negatives"
    h_origin = "halfs"

    a_net = "Anchor"
    p_net = "Positive"
    n_net = "Negative"
