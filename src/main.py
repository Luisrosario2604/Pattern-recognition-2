#!/usr/bin/python3
# Importing python3 from local, just use "python3 <binary>" if is not the same location

# RECONOCIMIENTO DE PATRONES - RETO 2
# Máster en Vision Artificial
# Luis Rosario Tremoulet y Vicente Gilabert Maño.

# Features used :   x1 -> sum of rows with pixels in the first 5 rows
#                   x2 -> width ** (1 / 3)

# [IMPORTS]
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# [GLOBAL VARIABLES]
is_printing_shapes = False      # Do you want to print the datasets sizes ?
is_showing_image = False        # Do you want to see the representation of the three's and seven's ?
is_showing_extractions = True   # Do you want to see the extractions x1 and x2 ?
is_saving_model = False         # Do you want to save the model into a file (with pickle) ?
is_testing_model = True         # Do you want to test the model with the test set ?
is_resulting_model = False      # Do you want to save the results from the 10.000 MNIST files ?


# To force that every run produces the same outcome (comment, or remove, to get randomness)
np.random.seed(seed=123)

# [FUNCTION DECLARATIONS]

# Adding noise for a better visualization
def jitter(x, sigma=0.06):

    random_sign = (-1) ** np.random.randint(1, 3, *x.shape)
    return x + np.random.normal(0, sigma, *x.shape) * random_sign


# Scales the data from [O,255] to [0,1]
def scale_to_unit(data):

    data = (data / 255.0)
    return data


# Split general datasets to train and test dataset
def split_train_test(data, test_ratio):

    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train_set = data.iloc[train_indices]
    test_set = data.iloc[test_indices]
    return train_set.reset_index(drop=True), test_set.reset_index(drop=True)


# Getting extraction's of the digit's = x1 to x55
def feat_extraction(data, label, theta=0.55):

    features = np.zeros([data.shape[0], 3])  # <- allocate memory with zeros
    data = data.values.reshape([data.shape[0], 28, 28])
    # -> axis 0: id of instance, axis 1: width(cols) , axis 2: height(rows)
    for k in range(data.shape[0]):
        # current image
        x = data[k, :, :]

        # --width feature
        sum_cols = x.sum(axis=0)  # <- axis0 of x, not of data!!
        indc = np.argwhere(sum_cols > theta * sum_cols.max())
        width = indc[-1] - indc[0]
        # --height feature
        sum_rows = x.sum(axis=1)  # <- axis1 of x, not of data!!

        sum_rows_tmp1 = sum(sum_rows[tmp] > 0 for tmp in range(6))
        x1 = sum_rows_tmp1
        x2 = width ** (1 / 3)
        features[k, 0] = label
        features[k, 1] = x1
        features[k, 2] = x2

    col_names = ['label', 'x1', 'x2']
    return pd.DataFrame(features, columns=col_names)


# Function to predict results with competition dataset
def result_model(model):

    reto1_dataset = pd.read_csv("./datasets/reto2_X.csv", header=None)
    reto1_dataset = scale_to_unit(reto1_dataset)
    reto1_dataset = feat_extraction(reto1_dataset, -1)
    reto1_dataset = reto1_dataset.drop(columns=['label'])
    predictions_reto1 = model.predict(reto1_dataset)

    predictions_reto1 = np.array(predictions_reto1)
    # --- Save prediction in .csv file.
    np.savetxt('./predictions/Reto2_Ypred.csv', predictions_reto1, fmt='%i', delimiter=',')


# Function to test model with our testing dataset
def test_model(model, carac_test, label_test, show_cm=True):

    predictions = model.predict(carac_test)

    score = model.score(carac_test, label_test)
    print("Score of the model: ", score * 100, "%")
    print("Mean square error: ", mean_squared_error(label_test, predictions) * 100)
    cm = confusion_matrix(label_test, predictions, labels=model.classes_)

    if show_cm:
        disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
        disp.plot()
        plt.show()
    else:
        print("Confusion Matrix of the model: ")
        print(cm)


# Show plot with our features for both numbers.
def show_extraction(extraction0, extraction3, extraction6, extraction9):

    f1, ax = plt.subplots(2)  # 3 size of the subplot

    ax[0].plot(jitter(extraction0['x1']), 'o', color="blue", ms=2)
    ax[0].plot(jitter(extraction3['x1']), 'x', color="red", ms=2)
    ax[0].plot(jitter(extraction6['x1']), 'v', color="black", ms=2)
    ax[0].plot(jitter(extraction9['x1']), '>', color="yellow", ms=2)
    ax[0].title.set_text("X1")

    ax[1].plot(jitter(extraction0['x2']), 'o', color="blue", ms=2)
    ax[1].plot(jitter(extraction3['x2']), 'x', color="red", ms=2)
    ax[1].plot(jitter(extraction6['x2']), 'v', color="black", ms=2)
    ax[1].plot(jitter(extraction9['x2']), '>', color="yellow", ms=2)
    ax[1].title.set_text("X2")

    plt.show()

# Function to show all the digit's as pixels
def show_image(set1, set2, set3, set4, index):

    instance_id_to_show = index  # <- index of the instance of the digit that will be shown in a figure

    # --- Plot the whole Data Sets
    f1, ax = plt.subplots(2, 4)  # 2 rows, 4 columns size of the subplot
    ax[0, 0].imshow(set1, cmap='Blues')
    ax[0, 1].imshow(set2, cmap='Blues')
    ax[0, 2].imshow(set3, cmap='Blues')
    ax[0, 3].imshow(set4, cmap='Blues')
    # --> 2 figures in which each row is a row of the dataset

    # --- Plot an instance of set1
    ax[1, 0].imshow(set1.iloc[instance_id_to_show].values.reshape(28, 28), 'Blues')
    # --- Plot an instance of set2
    ax[1, 1].imshow(set2.iloc[instance_id_to_show].values.reshape(28, 28), 'Blues')
    # --- Plot an instance of set3
    ax[1, 2].imshow(set3.iloc[instance_id_to_show].values.reshape(28, 28), 'Blues')
    # --- Plot an instance of set4
    ax[1, 3].imshow(set4.iloc[instance_id_to_show].values.reshape(28, 28), 'Blues')
    plt.show()


# Read .csv, split and normalize dataset.
def loading_datasets(location_zero, location_three, location_six, location_nine):

    fraction_test = 0.2  # <- Percentage of the dataset held for test, in [0,1]

    full_set_0 = pd.read_csv(location_zero, header=None)
    full_set_3 = pd.read_csv(location_three, header=None)
    full_set_6 = pd.read_csv(location_six, header=None)
    full_set_9 = pd.read_csv(location_nine, header=None)

    # --- Separate Test set
    train_set_0, test_set_0 = split_train_test(full_set_0, fraction_test)
    train_set_3, test_set_3 = split_train_test(full_set_3, fraction_test)
    train_set_6, test_set_6 = split_train_test(full_set_6, fraction_test)
    train_set_9, test_set_9 = split_train_test(full_set_9, fraction_test)

    if is_printing_shapes:
        print("Shape of full set 0 = ", full_set_0.shape)
        print("\nShape of full set 3 = ", full_set_3.shape)
        print("\nShape of full set 6 = ", full_set_6.shape)
        print("\nShape of full set 9 = ", full_set_9.shape)

        print("\nShape of train set 0 = ", train_set_0.shape)
        print("\nShape of train set 3 = ", train_set_3.shape)
        print("\nShape of train set 6 = ", train_set_6.shape)
        print("\nShape of train set 9 = ", train_set_9.shape)

        print("\nShape of test set 0 = ", test_set_0.shape)
        print("\nShape of test set 3 = ", test_set_3.shape)
        print("\nShape of test set 6 = ", test_set_6.shape)
        print("Shape of test set 9 = ", test_set_9.shape)

    full_set_0 = scale_to_unit(full_set_0)
    full_set_6 = scale_to_unit(full_set_6)
    full_set_3 = scale_to_unit(full_set_3)
    full_set_9 = scale_to_unit(full_set_9)

    train_set_0 = scale_to_unit(train_set_0)
    train_set_3 = scale_to_unit(train_set_3)
    train_set_6 = scale_to_unit(train_set_6)
    train_set_9 = scale_to_unit(train_set_9)

    test_set_0 = scale_to_unit(test_set_0)
    test_set_3 = scale_to_unit(test_set_3)
    test_set_6 = scale_to_unit(test_set_6)
    test_set_9 = scale_to_unit(test_set_9)

    return full_set_0, full_set_3, full_set_6, full_set_9,\
            train_set_0, train_set_3, train_set_6, train_set_9, \
            test_set_0, test_set_3, test_set_6, test_set_9

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
def main():

    # --- Load dataset and get splited and normalize data.
    full_set_0, full_set_3, full_set_6, full_set_9, \
    train_set_0, train_set_3, train_set_6, train_set_9, \
    test_set_0, test_set_3, test_set_6, test_set_9 = loading_datasets('./datasets/1000_cero.csv', './datasets/1000_tres.csv', './datasets/1000_seis.csv', './datasets/1000_nueve.csv')

    # --- Show image with some index (0, 3, 6 and 9).
    if is_showing_image:
        index = 5
        show_image(train_set_0, train_set_3, train_set_6, train_set_9, index)

    # --- Features extraction of training and testing datasets (0, 3, 6 and 9)
    extraction_train_set_0 = feat_extraction(train_set_0, 0)
    extraction_train_set_3 = feat_extraction(train_set_3, 3)
    extraction_train_set_6 = feat_extraction(train_set_6, 6)
    extraction_train_set_9 = feat_extraction(train_set_9, 9)

    extraction_test_set_0 = feat_extraction(test_set_0, 0)
    extraction_test_set_3 = feat_extraction(test_set_3, 3)
    extraction_test_set_6 = feat_extraction(test_set_6, 6)
    extraction_test_set_9 = feat_extraction(test_set_9, 9)

    extraction_test_set_all = pd.concat([extraction_test_set_0, extraction_test_set_3, extraction_test_set_6, extraction_test_set_9], axis=0)
    extraction_train_set_all = pd.concat([extraction_train_set_0, extraction_train_set_3, extraction_train_set_6, extraction_train_set_9], axis=0)

    # --- If show_extraction = True -> Show Plots of new features.
    if is_showing_extractions:
        show_extraction(extraction_train_set_0, extraction_train_set_3, extraction_train_set_6, extraction_train_set_9)

    label_train = extraction_train_set_all.iloc[:, 0]
    label_test = extraction_test_set_all.iloc[:, 0]

    carac_train = extraction_train_set_all.iloc[:, 1: 3]
    carac_test = extraction_test_set_all.iloc[:, 1: 3]


# --- Create logistic regresion model and train (fit)
    model = make_pipeline(PolynomialFeatures(),
                          StandardScaler(),
                          LogisticRegression(solver='lbfgs', max_iter=50000))

    params = {
        "polynomialfeatures__degree": [2, 3, 4],
    }

    grid = GridSearchCV(model, param_grid=params, cv=4)

    grid.fit(carac_train, label_train)

# --- Save model
    if is_saving_model:
        filename = './model/trained_model.sav'
        pickle.dump(grid, open(filename, 'wb'))

    # --- Test model with test data
    if is_testing_model:
        test_model(grid, carac_test, label_test, show_cm=True)

    # --- Get prediction using our model and competition dataset. Generate .csv file with results.
    if is_resulting_model:
        result_model(grid)


# [MAIN BODY]
if __name__ == '__main__':
    main()
