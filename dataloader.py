from helpers import *

def clean_data(data, columns):
    """
    This function removes certain columns from the given data, cleaning it up for use
    """
    return np.delete(data, columns, axis=1)

def prepare_input_data(x_train, x_test, columns):
    """
    This function cleans and prepares the input data
    example use:
    
    data_path = "dataset"
    to_clean = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    x_train_raw, x_test_raw, y_train_raw, train_ids, test_ids = load_csv_data(data_path, sub_sample=False)
    x_train, x_test = prepare_input_data(x_train_raw, x_test_raw, to_clean)
    """
    return np.transpose(clean_data(x_train, columns)), np.transpose(clean_data(x_test, columns))