# Helper functions
from pandas import DataFrame


def load_df(input_path: str, sep: str = ";", decimal: str = ",") -> DataFrame:
    """
    Load a DataFrame from a file.

    Args:
        input_path (str): The path to the input DataFrame file.
        sep (str, optional): The delimiter used in the input file (for CSV files). Defaults to ";".
        decimal (str, optional): The character recognized as a decimal point (for CSV files). Defaults to ",".

    Returns:
        DataFrame: The loaded DataFrame.

    Raises:
        ValueError: If the input file format is not 'csv' or 'pkl'.
    """
    import pandas as pd
    import os

    try:
        # Extract filename and extension
        _, in_extension = os.path.splitext(os.path.basename(input_path))

        # Read the dataframe
        if in_extension == ".csv":
            df = pd.read_csv(input_path, sep=sep, decimal=decimal)
        elif in_extension == ".pkl":
            df = pd.read_pickle(input_path)
        else:
            raise ValueError(
                f"Format {in_extension} not supported. Only 'csv' or 'pkl' are supported.")

        return df

    except Exception as e:
        raise e


def drop_columns(df: DataFrame, cols_to_drop: list = []):
    """
    Drop specified columns from the DataFrame.

    This function drops the columns specified in the 'cols_to_drop' list from the input DataFrame 'df'.

    Args:
        df (DataFrame): The DataFrame from which columns will be dropped.
        cols_to_drop (list, optional): A list of column names to be dropped from the DataFrame. Defaults to [].

    Returns:
        DataFrame: The DataFrame with specified columns dropped.

    Raises:
        TypeError: If the input DataFrame is not a pandas DataFrame.
        ValueError: If 'cols_to_drop' is not a list or if any of the specified columns do not exist in the DataFrame.
    """
    import pandas as pd

    try:
        # Check if input is a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")

        # Check if cols_to_drop is a list
        if not isinstance(cols_to_drop, list):
            raise ValueError("'cols_to_drop' must be a list.")

        # Check if the specified columns exist in the DataFrame
        non_existing_cols = [
            col for col in cols_to_drop if col not in df.columns]
        if non_existing_cols:
            raise ValueError(
                f"Columns {non_existing_cols} do not exist in the DataFrame.")

        # Drop specified columns
        df.drop(columns=cols_to_drop, axis=1, inplace=True)

        return df

    except Exception as e:
        # Raise an exception if an error occurs
        raise e


def detect_missing_values(df: DataFrame, threshold: float = 0.1) -> tuple:
    """
    Detect columns with missing values exceeding the specified threshold.

    This function detects columns in the input DataFrame 'df' 
    that have missing values exceeding the specified 'threshold'. It returns a tuple containing two elements:
    1. A list of columns to be dropped due to the ratio of missing values exceeding the threshold.
    2. A pandas Series containing the ratio of missing values for each column with missing values.

    Args:
        df (DataFrame): The DataFrame to check for missing values.
        threshold (float, optional): The threshold value for the ratio of missing values. 
            Columns with missing value ratios exceeding this threshold will be marked for dropping. Defaults to '0.1' (10 %).

    Returns:
        tuple: A tuple containing two elements:
               1. A list of columns to be dropped.
               2. A pandas Series containing the ratio of missing values for each column.

    Raises:
        TypeError: If the input 'df' is not a pandas DataFrame.
    """
    import pandas as pd

    try:
        # Check if input is a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")

        # Calculate the number of rows
        nrows = len(df)

        # Retrieve columns with missing values
        col_w_nulls = df.columns[df.isna().any()]

        # Calculate the ratio of missing values
        null_ratio = df[col_w_nulls].isna().sum(
        ).sort_values(ascending=False) / nrows

        # Identify columns to be dropped based on the threshold
        cols_to_drop = list(null_ratio.index[null_ratio > threshold])

        return cols_to_drop, null_ratio

    except Exception as e:
        raise e


def impute_missing(df: DataFrame, set_type: str = "train", imputer: callable = None):
    """
    Impute missing values in a DataFrame using the specified imputation strategy.

    This function replaces missing values in the input DataFrame 'df' with 
    values computed based on the specified imputation strategy. 
    The imputation can be performed on either the training set or the test set, as indicated 
    by the 'set_type' parameter.

    Args:
        df (DataFrame): The DataFrame containing the data with missing values to be imputed.
        set_type (str, optional): Specifies whether the imputation should be performed on the training set 
            ('train') or the test set ('test'). Defaults to 'train'.
        imputer (callable, optional): An object implementing the 'fit' and 'transform' methods 
            for imputing missing values. If not provided, a 'SimpleImputer' with strategy 'most_frequent' 
            will be used. Defaults to None.

    Returns:
        callable: The imputer object used for imputation if 'set_type' is 'train'. Otherwise, None.

    Raises:
        ValueError: If an invalid value is provided for the 'set_type' parameter.
                    If an imputer object is not provided for imputing missing values in the test set.
        Exception: If an error occurs during the imputation process.
    """
    import pandas as pd
    from sklearn.impute import SimpleImputer

    try:
        if set_type.lower() == "train":
            if imputer is None:
                imputer = SimpleImputer(strategy="most_frequent")
            df[df.columns] = imputer.fit_transform(df)
            df[df.columns] = df[df.columns].infer_objects()
            return imputer
        elif set_type.lower() == "test":
            if imputer is None:
                raise ValueError(
                    "An imputer object is required for imputing missing values in the test set.")
            df[df.columns] = imputer.transform(df)
            df[df.columns] = df[df.columns].infer_objects()
        else:
            raise ValueError(
                "Invalid set_type. Allowed options are 'train' or 'test'.")

        return

    except Exception as e:
        raise e


def replace_outliers(df: DataFrame, whiskers: float = 1.5) -> DataFrame:
    """
    Replace outliers with median value using the IQR criteria.

    Args:
        df (DataFrame): Input data frame with numeric values only
        whiskers (float, optional): Threshold to compute the upper and lower bounds. Defaults to 1.5.

    Returns:
        Data Frame: With outliers replaced.

    Raises:
        Exception: If an error occurs during the imputation process.
    """
    import numpy as np

    try:
        # Make sure the selected values are the numeric ones
        df_numeric = df.select_dtypes('number')
        # Compute the quantiles for each column
        quant = df_numeric.quantile(q=[0.75, 0.25])
        # Compute the IQR and the upper and lower bounds used to consider outliers
        iqr = quant.iloc[0] - quant.iloc[1]

        up_bound = quant.iloc[0] + (whiskers*iqr)
        low_bound = quant.iloc[1] - (whiskers*iqr)

        # Replace values above upper bound with median
        df_numeric = df_numeric.apply(lambda x: x.mask(
            x > up_bound.loc[x.name], np.nan), axis=0)

        # Replace values below lower bound with median
        df_numeric = df_numeric.apply(lambda x: x.mask(
            x < low_bound.loc[x.name], np.nan), axis=0)

        # Replace the numeric columns in the data set
        df[df_numeric.columns] = df_numeric

        return df

    except Exception as e:
        raise e


def scale_df(
        df: DataFrame, set_type: str = "train",
        scaler_chosen: callable = None, scaler_type: str = "standard") -> tuple:
    """
    Scale a dataset with a passed/selected scaler type.

    Args:
        df (Data Frame): Dataset to scale.
        set_type (str, optional): Dataset type 'train' or 'test'. Defaults to "train".
        scaler_chosen (callable, optional): Scaler object to use. Needed if set_type = test. Defaults to None.
        scaler_type (str, optional): Scaler type to be used. Needed for training datasets. Defaults to "standard".
            Allowed methods: 'robust', 'standard', 'power'.

    Returns:
        Data Frame: Dataframe scaled.
        callable: Scaler used.

    Raises:
        Exception: If an error occurs during the imputation process.
    """
    import pandas as pd
    from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer

    scaler_dict = {"robust": RobustScaler(),
                   "standard": StandardScaler(),
                   "power": PowerTransformer()}
    try:
        if set_type.lower() == "train":
            if scaler_type is None:
                raise ValueError(
                    "A valid scaler type has to be chosen. Valid scalers are: 'robust', 'standard' and 'power'")
            scaler_chosen = scaler_dict[scaler_type]
            array_scaled = scaler_chosen.fit_transform(df)
            df_scaled = pd.DataFrame(
                data=array_scaled, columns=scaler_chosen.get_feature_names_out())

        elif set_type.lower() == "test":
            if scaler_chosen is None:
                raise ValueError("An scaler object has to be passed.")
            array_scaled = scaler_chosen.fit_transform(df)
            df_scaled = pd.DataFrame(
                data=array_scaled, columns=scaler_chosen.get_feature_names_out())

        df_scaled.index = df.index

        return df_scaled, scaler_chosen

    except Exception as e:

        raise e
