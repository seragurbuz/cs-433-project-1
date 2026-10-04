from helpers import *
import csv
import os

REPLACEMENTS = {
    "PHYSHLTH": {88: 0, 77: 0, 99: 0,np.nan: "median"},
    "MENTHLTH": {88: 0, 77: 0, 99: 0,np.nan: "median"},
}

LIST_KEEP = [
    # General health
    "GENHLTH", "PHYSHLTH", "MENTHLTH", "POORHLTH",
    # Health care access
    "HLTHPLN1", "PERSDOC2", "MEDCOST", "CHECKUP1",
    # Classic cardiac risk factors
    "_RFHYPE5", "BPMEDS", "_CHOLCHK", "_RFCHOL", "DIABETE3",
    # Other chronic conditions
    "CVDSTRK3", "_ASTHMS1", "CHCSCNCR", "CHCOCNCR", "CHCCOPD1",
    "_DRDXAR1", "ADDEPEV2", "CHCKIDNY",
    # Demographics
    "SEX", "_AGE80", "MARITAL", "_EDUCAG", "_INCOMG", "RENTHOM1",
    "VETERAN3", "EMPLOY1", "_CHLDCNT", "INTERNET",
    # Body mass
    "_BMI5",
    # Functional limitations
    "QLACTLM2", "USEEQUIP", "BLIND", "DECIDE",
    "DIFFWALK", "DIFFDRES", "DIFFALON",
    # Lifestyle
    "_SMOKER3", "USENOW3", "_DRNKWEK", "_RFBING5",
    "_FRUTSUM", "_VEGESUM",
    "_TOTINDA", "PA1MIN_", "_PACAT1", "_PASTRNG",
    # Prevention
    "FLUSHOT6", "PNEUVAC3", "HIVTST6", "_RFSEAT2",
]

def get_delete_indices(csv_path, keep_list):
    with open(csv_path) as f:
        header = next(csv.reader(f))[1:]
    missing = [n for n in keep_list if n not in header]
    if missing:
        raise ValueError(f"Not in header: {missing}")
    return [i for i, name in enumerate(header) if name not in keep_list]

def clean_data(data, columns):
    """
    This function removes certain columns from the given data, 
    cleaning it up for use.
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
    return clean_data(x_train, columns), clean_data(x_test, columns)

def replace_values(data, header, replacements):
    """
    This function replaces values in the data according to the 
    specified replacement rules.

    Example :
    Dictionnary = {
        "CELLFON3": {88: 0, 77: 1}
        }
    So if the column CELLFON3 has a value of 88 or 77, 
    it will be replaced by 0 and 1 respectively.
    """
    data = data.copy()
    for name, rules in replacements.items():
        if name not in header:
            continue
        col = data[:, header.index(name)]
        for old, new in rules.items():
            if np.isnan(old):
                mask = np.isnan(col)
            else:
                mask = col == old
            if new == "median":
                new = np.nanmedian(col)
            col[mask] = new
    return data


def count_values(data, keeplist, dictionnary):
    """
    This fonction counts the number of occurrences of each value
    in the specified columns of the data, according to the given 
    dictionary of replacements.
    """
    for i, name in enumerate(keeplist):
        if name in dictionnary:
            for value, replacement in dictionnary[name].items():
                count = np.sum(data[:, i] == value)
                print(f"{name}: {value}, count: {count}")

def save_csv_data(savedatapath, data, filename, keepList):
    """
    This function saves the given data to a CSV file with the specified filename.
    It adds a header line with the column names from the keeplist.
    """
    with open(os.path.join(savedatapath, filename), 'w') as f:
        f.write(','.join(keepList) + '\n')
        np.savetxt(f, data, delimiter=',', fmt='%s')

if __name__ == "__main__":
    # Example usage of the functions in this file

    # Define the replacements and the list of columns to keep
    REPLACEMENTS = {
    "CELLFON3": {88: 0, 77: 0, 99: 0,np.nan: "median"},
    "MENTHLTH": {88: 0, 77: 0, 99: 0,np.nan: "median"},
    }
    LIST_KEEP = ["CELLFON3", "MENTHLTH"]

    # Load the data
    data_path = "dataset"
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)

    # Get the indices of the columns to delete based on the LIST_KEEP
    indices_to_delete = get_delete_indices(os.path.join(data_path, "x_train.csv"), LIST_KEEP)

    # Get the data of interest
    x_train_clean, x_test_clean = prepare_input_data(x_train, x_test, indices_to_delete)
    print("x_train_clean shape:", x_train_clean.shape)

    # Check for NaN values in the cleaned training data
    mask = np.isnan(x_train_clean)
    print(mask.sum(axis=0))

    # Replace the values in the dictionnary with the specified replacements
    replaced_data = replace_values(x_train_clean, LIST_KEEP, REPLACEMENTS)

    count_values(x_train_clean, LIST_KEEP, REPLACEMENTS)
    count_values(replaced_data, LIST_KEEP, REPLACEMENTS)
    print(x_train_clean[:10,:])
    print(replaced_data[:10,:])

    # Save the cleaned and replaced data to CSV files
    data_path = "dataset/cleaned_data"
    save_csv_data(data_path, x_train_clean, "x_train_clean.csv", LIST_KEEP)
    save_csv_data(data_path, replaced_data, "x_train_replaced.csv", LIST_KEEP)