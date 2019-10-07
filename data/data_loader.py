
def CreateDataLoader(opt, pic_path=None):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader(opt, pic_path)
    print(data_loader.name())
    return data_loader
