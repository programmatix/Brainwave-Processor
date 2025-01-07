def get_model_feature_names(model):
    if 'feature_names' in dir(model):
        return model.feature_names
    if 'feature_names_' in dir(model):
        return model.feature_names_
    if 'feature_names_in_' in dir(model):
        return model.feature_names_in_
    raise Exception("Could not find feature names in model")