#!/usr/bin/python3
# Importing python3 from local, just use "python3 <binary>" if is not the same location

# RECONOCIMIENTO DE PATRONES - RETO 1
# Máster en Vision Artificial
# Luis Rosario Tremoulet y Vicente Gilabert Maño.

# [IMPORTS]
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error

# [GLOBAL VARIABLES]
is_printing_shapes = True       # Do you want to print the datasets sizes ?
is_showing_image = True         # Do you want to see the representation of the three's and seven's ?
is_showing_extractions = True   # Do you want to see the extractions x1 and x2 ?
is_saving_model = False         # Do you want to save the model into a file (with pickle) ?
is_testing_model = True        # Do you want to test the model with the test set ?
is_resulting_model = True      # Do you want to save the results from the 10.000 MNIST files ?

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


# Getting extraction's of the three's and seven's = x1, x2
def feat_extraction(data, label, theta=0.5):

    features = np.zeros([data.shape[0], 3])  # <- allocate memory with zeros
    data = data.values.reshape([data.shape[0], 28, 28])
    # -> axis 0: id of instance, axis 1: width(cols) , axis 2: height(rows)
    for k in range(data.shape[0]):
        x = data[k, :, :]
        sum_rows = x.sum(axis=1)  # <- axis1 of x, not of data!!
        indr = np.argwhere(sum_rows > theta * sum_rows.max())
        height = indr[-1] - indr[0]

        sum_rows_tmp1 = 0
        for tmp in range(6):
            if sum_rows[tmp] > theta * sum_rows[tmp].max():
                sum_rows_tmp1 += 1

        x1 = height ** (1 / 3) / 2.7
        x2 = sum_rows_tmp1 / 5.0

        features[k, 0] = x1
        features[k, 1] = x2
        features[k, 2] = label

    col_names = ['x1', 'x2', 'label']
    return pd.DataFrame(features, columns=col_names)


# Function to predict results with competition dataset
def result_model(model):

    reto1_dataset = pd.read_csv("./Datasets/reto2_X.csv", header=None)
    reto1_dataset = scale_to_unit(reto1_dataset)
    reto1_dataset = feat_extraction(reto1_dataset, 2)
    reto1_dataset = reto1_dataset.drop(columns=['label'])
    predictions_reto1 = model.predict(reto1_dataset)

    # --- Replace 0 -> 3 and 1 -> 7.
    predictions_reto1 = np.array(predictions_reto1)
    replace = np.where(predictions_reto1 == 3, 9, predictions_reto1)
    replace = np.where(replace == 1, 3, replace)
    replace = np.where(replace == 2, 6, replace)
    # --- Save prediction in .csv file.
    np.savetxt('Reto2_Ypred.csv', replace, fmt='%i', delimiter=',')


# Function to test model with our testing dataset
def test_model(model, test_set, show_cm=True):

    predictions = model.predict(test_set.iloc[:, [0, 1]])
    score = model.score(test_set.iloc[:, [0, 1]], test_set.iloc[:, 2])
    print("Score of the model: ", score)
    print("Mean square error: ", mean_squared_error(test_set.iloc[:, 2], predictions))
    cm = confusion_matrix(test_set.iloc[:, 2], predictions, labels=model.classes_)

    if show_cm:
        disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
        disp.plot()
        plt.show()
    else:
        print("Confusion Matrix of the model: ")
        print(cm)


# Show plot with our features for both numbers.
def show_extraction(extraction0, extraction3, extraction6, extraction9):

    f1, ax = plt.subplots(2)  # 4 size of the subplot

    ax[0].plot(jitter(extraction0['x1']), 'o', color="blue", ms=0.7)
    ax[0].plot(jitter(extraction3['x1']), 'x', color="red", ms=0.7)
    ax[0].plot(jitter(extraction6['x1']), 'v', color="black", ms=0.7)
    ax[0].plot(jitter(extraction9['x1']), ',', color="yellow", ms=0.7)
    ax[0].title.set_text("NORM - Height ** (1/3)")

    ax[1].plot(jitter(extraction0['x2']), 'o', color="blue", ms=0.7)
    ax[1].plot(jitter(extraction3['x2']), 'x', color="red", ms=0.7)
    ax[1].plot(jitter(extraction6['x2']), 'v', color="black", ms=0.7)
    ax[1].plot(jitter(extraction9['x2']), ',', color="yellow", ms=0.7)
    ax[1].title.set_text("NORM - Count 6 firts pixels")

    plt.show()


# Function to show all the three's and seven's as pixels
def show_image(set1, set2, set3, set4, index):

    instance_id_to_show = index  # <- index of the instance of 3 and 7 that will be shown in a figure

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
        full_set_3 = scale_to_unit(full_set_3)
        full_set_6 = scale_to_unit(full_set_6)
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


def main():

    # --- Load dataset and get splited and normalize data.
    full_set_0, full_set_3, full_set_6, full_set_9, \
    train_set_0, train_set_3, train_set_6, train_set_9, \
    test_set_0, test_set_3, test_set_6, test_set_9 = loading_datasets('./Datasets/1000_cero.csv', './Datasets/1000_tres.csv', './Datasets/1000_seis.csv', './Datasets/1000_nueve.csv')

    # --- Show image with some index (0 and 3 and 6 and 9).
    if is_showing_image:
        index = 5
        show_image(train_set_0, train_set_3, train_set_6, train_set_9, index)

    # --- Features extraction of training and testing datasets (3 and 7)
    extraction_train_set_0 = feat_extraction(train_set_0, 0)
    extraction_train_set_3 = feat_extraction(train_set_3, 1)
    extraction_train_set_6 = feat_extraction(train_set_6, 2)
    extraction_train_set_9 = feat_extraction(train_set_9, 3)

    extraction_test_set_0 = feat_extraction(test_set_0, 0)
    extraction_test_set_3 = feat_extraction(test_set_3, 1)
    extraction_test_set_6 = feat_extraction(test_set_6, 2)
    extraction_test_set_9 = feat_extraction(test_set_9, 3)

    extraction_test_set_all = pd.concat([extraction_test_set_0, extraction_test_set_3, extraction_test_set_6, extraction_test_set_9], axis=0)
    extraction_train_set_all = pd.concat([extraction_train_set_0, extraction_train_set_3, extraction_train_set_6, extraction_train_set_9], axis=0)

    # --- If show_extraction = True -> Show Plots of new features.
    if is_showing_extractions:
        show_extraction(extraction_train_set_0, extraction_train_set_3, extraction_train_set_6, extraction_train_set_9)

    # --- Create logistic regresion model and train (fit)
    model = LogisticRegression()
    model.fit(extraction_train_set_all.iloc[:, [0, 1]], extraction_train_set_all.iloc[:, 2])

    # --- Save model
    if is_saving_model:
        filename = 'trained_model.sav'
        pickle.dump(model, open(filename, 'wb'))

    # --- Test model with test data
    if is_testing_model:
        test_model(model, extraction_test_set_all, show_cm=True)

    # --- Get prediction using our model and competition dataset. Generate .csv file with results.
    if is_resulting_model:
        result_model(model)


# [MAIN BODY]
if __name__ == '__main__':
    main()
