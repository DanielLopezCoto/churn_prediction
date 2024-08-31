import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from helper_functions import impute_missing, replace_outliers, scale_df
from helper_functions import load_df, drop_columns, detect_missing_values



def feature_selection(
        features_selected: list, input_path: str, output_path: str,
        sep: str = ";", decimal: str = ",") -> str:
    """
    Selects specified features from a DataFrame and saves the result to a pickle file.

    Args:
        features_selected (list): A list of feature column names to select.
        input_path (str): The path to the input DataFrame file.
        output_path (str): The directory path where the output file will be saved.
        sep (str, optional): The delimiter used in the input file (for CSV files). Defaults to ";".
        decimal (str, optional): The character recognized as a decimal point (for CSV files). Defaults to ",".

    Returns:
        str: The path to the saved pickle file.

    Raises:
        ValueError: If the input file format is not 'csv' or 'pkl'.
    """
    try:

        # Extract filename and extension
        filename, _ = os.path.splitext(os.path.basename(input_path))

        # Load dataframe
        df = load_df(input_path=input_path, sep=sep, decimal=decimal)

        # Use the input filename as the base name for the output
        outfilename = f"{filename}_features_selected.pkl"

        # Select the features
        df_out = df[features_selected]

        # Save dataframe to pickle format
        full_output_path = os.path.join(output_path, outfilename)
        df_out.to_pickle(full_output_path)

        return

    except Exception as e:
        raise e


def split_data(input_path: str, output_path: str, target_feature: str, test_size: float = 0.2):
    """
    Load the data from 'input_path', identify the target variable specified by 
    'target_feature', and split the dataset into training and testing sets.
    Save the resulting data splits as pickle files in the 'output_path' directory.

    Args:
        input_path (str): The path to the input file containing the dataset.
        output_path (str): The directory path where the output files will be saved.
        target_feature (str): The name of the target variable to be predicted.
        test_size (float, optional): The proportion of the dataset to include in the test split. 
            Should be between 0.0 and 1.0. Defaults to 0.2.

    Returns:
        str: A message indicating the success of the data loading and splitting process.

    Raises:
        Exception: If an error occurs during the process.
    """
    from sklearn.model_selection import train_test_split

    try:
        # Load dataset
        data = load_df(input_path, sep=";", decimal=".")

        print("Successful data load.")

        # Identify target and features
        df_target = data[target_feature]
        df_features = data.drop(columns=target_feature)

        # Split train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            df_features, df_target, stratify=df_target, test_size=test_size, random_state=42)

        # Save train and test sets
        X_train.to_pickle(os.path.join(output_path, 'train_features.pkl'))
        X_test.to_pickle(os.path.join(output_path, 'test_features.pkl'))
        y_train.to_pickle(os.path.join(output_path, 'train_target.pkl'))
        y_test.to_pickle(os.path.join(output_path, 'test_target.pkl'))

        print("Train and test sets saved.")

        return

    except Exception as e:
        raise e


def clean_data(
        train_input_path: str, test_input_path: str,
        output_path: str, drop_columns_bool: bool = False,
        missing_val_threshold: float = 0.1,
        scale_data: bool = True, scaler_type='standard'
) -> None:
    """
    Clean and preprocess the training and testing datasets.

    This function loads the training and testing datasets from the provided input paths, 
    applies data cleaning and preprocessing steps, and saves the cleaned datasets to the 
    specified output directory.

    Args:
        train_input_path (str): The file path to the training dataset.
        test_input_path (str): The file path to the testing dataset.
        output_path (str): The directory path where the cleaned datasets will be saved.
        drop_columns_bool (bool, optional): Whether to drop columns with missing values exceeding the threshold. Defaults to False.
        missing_val_threshold (float, optional): Threshold of allowed missing values. Default to 0.1.
        scale_data (bool, optional): Wheter to scale the data or not. Default to True.
        scaler_type (str, optional): Scaler type to be used. Needed for trainin datasets. Defaults to "standard".
            Allowed methods: 'robust', 'standard', 'power'.

    Returns:
        None

    Raises:
        ValueError: If 'train_input_path', 'test_input_path', or 'output_path' is empty or None.
                    If the specified input files do not exist.
                    If an error occurs during data cleaning and preprocessing.

        Raises: Exception: If an error occurs during the imputation process.
    """
    try:
        # Check if input and output paths are provided
        if not train_input_path or not test_input_path or not output_path:
            raise ValueError("Input and output paths must be provided.")

        # Check if input files exist
        for file_path in [train_input_path, test_input_path]:
            if not os.path.isfile(file_path):
                raise ValueError(f"File '{file_path}' does not exist.")

        # Load the training and testing datasets
        X_train = load_df(train_input_path)
        X_test = load_df(test_input_path)

        # Replace outliers
        replace_outliers(X_train)
        replace_outliers(X_test)

        if drop_columns_bool:
            # Detect and drop columns with missing values exceeding the threshold
            cols_to_drop, _ = detect_missing_values(
                df=X_train, threshold=missing_val_threshold)
            drop_columns(df=X_train, cols_to_drop=cols_to_drop)
            drop_columns(df=X_test, cols_to_drop=cols_to_drop)

        # Impute missing values
        imputer = impute_missing(df=X_train, set_type="train")
        impute_missing(df=X_test, set_type="test", imputer=imputer)

        if scale_data:
            X_train_num_scaled, scaler_obj = scale_df(X_train.select_dtypes(
                'number'), set_type="train", scaler_type=scaler_type)
            X_test_num_scaled, _ = scale_df(X_test.select_dtypes(
                'number'), set_type="test", scaler_chosen=scaler_obj)

            # Replace the columns scaled
            X_train[X_train_num_scaled.columns] = X_train_num_scaled
            X_test[X_test_num_scaled.columns] = X_test_num_scaled

        # Save cleaned datasets
        X_train.to_pickle(os.path.join(
            output_path, "train_features_cleaned.pkl"))
        X_test.to_pickle(os.path.join(
            output_path, "test_features_cleaned.pkl"))

        return

    except Exception as e:
        raise e


def encode_data(
        train_input_path: str, test_input_path: str,
        output_path: str, encoder_type: str = "onehot") -> None:
    """
    Encodes categorical variables in the training and testing datasets and saves them as pickle files.

    Parameters:
        train_input_path (str): Path to the training dataset.
        test_input_path (str): Path to the testing dataset.
        output_path (str): Path to save the encoded datasets.
        encoder_type (str, optional): Type of encoder to use. Default is "onehot". 
            Valid options are "onehot" and "label".

    Raises:
        ValueError: If input and output paths are not provided or if the encoder type is invalid.
        FileNotFoundError: If any of the input files do not exist.
        Exception: If an error occurs during the imputation process.

    Returns:
        None
    """
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder

    try:
        if not train_input_path or not test_input_path or not output_path:
            raise ValueError("Input and output paths must be provided.")

        # Check if input files exist
        for file_path in [train_input_path, test_input_path]:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File '{file_path}' does not exist.")

        # Load the training and testing datasets
        X_train = load_df(train_input_path)
        X_test = load_df(test_input_path)

        # Extract categorical variables
        cat_var = X_train.select_dtypes('object').columns

        encoder_dict = {
            "onehot": OneHotEncoder(handle_unknown='ignore'),
            "label": LabelEncoder()
        }
        # Initialize encoder based on the encoder type
        if encoder_type not in encoder_dict.keys():
            raise ValueError(
                "A valid encoder type must be provided. Valid options are: 'onehot' and 'label'.")
        cat_encoder = encoder_dict[encoder_type]

        # Encode the training set
        feature_train_encoded = cat_encoder.fit_transform(
            X_train[cat_var]).toarray()
        feature_labels = cat_encoder.get_feature_names_out()
        X_train_cat = pd.DataFrame(
            data=feature_train_encoded, columns=feature_labels)
        X_train_cat.index = X_train.index
        X_train = pd.concat([X_train, X_train_cat], axis=1)
        X_train.drop(columns=cat_var, axis=1, inplace=True)

        # Encode the testing set
        feature_test_encoded = cat_encoder.transform(
            X_test[cat_var]).toarray()
        X_test_cat = pd.DataFrame(
            data=feature_test_encoded, columns=cat_encoder.get_feature_names_out())
        X_test_cat.index = X_test.index
        X_test = pd.concat([X_test, X_test_cat], axis=1)
        X_test.drop(columns=cat_var, axis=1, inplace=True)

        # Save encoded data
        X_train.to_pickle(os.path.join(
            output_path, "train_features_encoded.pkl"))
        X_test.to_pickle(os.path.join(
            output_path, "test_features_encoded.pkl"))

        return

    except Exception as e:
        raise e


def train_model_classification(
        features_train_input_path: str,
        target_train_input_path: str,
        output_path: str) -> None:
    """
    Trains a RandomForestClassifier model for classification using provided features and targets.

    Parameters:
        features_train_input_path (str): Path to the file containing training features.
        target_train_input_path (str): Path to the file containing training targets.
        output_path (str): Directory path to save the trained model.

    Raises:
        FileNotFoundError: If any of the input files do not exist.
        Exception: If an error occurs during the imputation process.

    Returns:
        None
    """
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    try:
        # Check if input files exist
        for file_path in [features_train_input_path, target_train_input_path]:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File '{file_path}' does not exist.")

        # Load the training features and targets
        X_train = load_df(features_train_input_path)
        y_train = load_df(target_train_input_path)

        # Initialize and train RandomForestClassifier
        rfc = RandomForestClassifier(random_state=42)
        rfc.fit(X_train, y_train)

        # Save trained model
        joblib.dump(rfc, os.path.join(output_path, "trained_rfc_model.pkl"))

        # Capture the feature names and their importance
        feature_names = X_train.columns.tolist()
        feature_scores = rfc.feature_importances_

        # Print feature importance
        sorted_features = sorted(
            zip(feature_scores, feature_names), reverse=True)
        for importance, feature in sorted_features:
            print(f"{feature}: {importance}")

        return

    except Exception as e:
        raise e


def evaluate_model(
        features_test_input_path: str,
        target_test_input_path: str,
        model_path: str) -> None:
    """
    Evaluates the model passed.

    Args:
        features_test_input_path (str): Test features file path.
        target_test_input_path (str): Test target file path.
        model_path (str): Path where the model is stored.

    Return:
        None

    Raises:
        FileNotFoundError: If any of the input files do not exist.
        Exception: If an error occurs during the imputation process.
    """
    import joblib
    from sklearn.metrics import accuracy_score
    try:
        # Check if input files exist
        for file_path in [features_test_input_path, target_test_input_path, model_path]:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File '{file_path}' does not exist.")

        # Load the training features and targets
        X_test = load_df(features_test_input_path)
        y_test = load_df(target_test_input_path)

        # Load model
        model = joblib.load(model_path)

        # Make prediction
        y_pred = model.predict(X_test)

        # Evaluate the model
        score = accuracy_score(y_true=y_test, y_pred=y_pred)

        print(f"Accuracy of the model: {score:.2f}")

        return

    except Exception as e:
        raise e
