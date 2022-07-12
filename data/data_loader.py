
def CreateDataLoader(opt,is_val=False,keyframes=None):

    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt,is_val=is_val,keyframes=keyframes)
    
    return data_loader
