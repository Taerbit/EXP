#       CONTROLLER
import os

import numpy as np
import pandas as pd

import EXP.src.Algorithm as Algorithm
import EXP.src.Input as Input
import EXP.src.Metrics as Metrics


"""

    Master pipeline to process data taken from input through a list of algorithms and compile metrics
    based off of these algorithm results.
    A list of panda.DataFrames for each algorithm is returned

"""


def pipeline(input, algos, metrics, model=False):
    #       Setup pandas dataframe : Results
    if not model:
        temp = ['ID']
    else:
        temp = ['ID', 'Correct', 'True Class', 'True Class Score', 'Predicted Class', 'Predicted Class Score']

    for m in metrics:
        temp.append(m.name)
    x = pd.DataFrame(columns=temp)

    # Create n copies of the x pandas Dataframe where n is the number of algos
    results = []
    for a in algos:
        df = x
        df.name = a.name
        results.append(df)

    data_counter = 0
    print("Starting Pipeline Loop")

    # Load all the data through the interpretability algorithm and then metrics
    while input.has_next():

        # load data
        data, id = input.get_next()
        print(str(data_counter) + ":\t" + id)

        for x, a in enumerate(algos):

            def add_common_attributes(id, model, img, label):
                pred = model.predict(np.array([img]))
                pred_label = np.argmax(pred)
                predicted_score = pred[0][pred_label]
                label_score = pred[0][label]
                if label == pred_label:
                    correct = True
                else:
                    correct = False
                return [id, correct, label, label_score, pred_label, predicted_score]

            if not model:
                current_row = [id]
            else:
                # Process the default output (Correct, Score, Predicted, Labelled)
                current_row = add_common_attributes(id, model, data["Input_Image"], data["Label"])

            param = a.get_input()
            input_data = []
            for p in param:
                input_data.append(data[p])

            matrix = a.pass_through(input_data)  # Send image and label away

            # Process algorithm result through all metrics
            for y, m in enumerate(metrics):
                current_row.append(m.process(matrix, data["Segmentation"]))  # Send matrix and segmentation away

            (results[x]).loc[data_counter] = current_row
            data_counter = data_counter + 1
    return results


'''

    Plotting of data

'''

import seaborn as sns
import matplotlib.pyplot as plt


def histogram(results, save_path, metric_index):
    correct = results.loc[results['Correct'] == True].iloc[:, metric_index]
    incorrect = results.loc[results['Correct'] == False].iloc[:, metric_index]

    sns.distplot(correct, color='g', kde=False)
    sns.distplot(incorrect, color='r', kde=False)
    print("Saving...\tHistogram")
    plt.savefig(save_path + "_hist.png")

    plt.clf()


def scatterplot(results, save_path, metric_index, y_name="True Class Score", x="", y=""):
    ax = sns.scatterplot(x=results.iloc[:, metric_index], y=y_name, hue="Correct", style="Correct", data=results)
    if x != "":
        ax.set(xlabel=x)
    if y != "":
        ax.set(ylabel=y)
    print("Saving...\t" + y_name + " Scatter")
    plt.savefig(save_path + "_" + y_name + "_scatter.png")

    plt.clf()


"""
    Hons Pipeline  
        The Hons pipeline is the given pipeline for comparing grad_cam and shap against metrics Average and
        N. It takes a specified tag 2D array, model and relevant model parameters.

        The tags lists should contain lists with the front path and the suffix for the data wanted loaded.
        Data MUST be given in the order: INPUT IMAGE, LABEL, SEGMENTATION

"""

# e.g. filepath = [[list of image paths], [path to csv, target column], [list of segmentation paths]]
def Hons(model, filepaths, tags, input_size=(225, 300), output_size=(1022, 767),
         output=os.getcwd(), background=[], inside_colour=255,
         save_matrices=False, save_imgs=False, save_csv=False):
    #       INPUT
    # IMPROVE: Validate filepath[n] = expected_input_type[n]
    containers = [Input.Input_Image(filepaths[0], tags[0], input_size),
                  Input.Label(filepaths[1][0], tags[1], filepaths[1][1]),
                  Input.Segmentation(filepaths[2], tags[2], output_size)]
    input = Input.Sorter(containers)

    #       ALGO
    if background == []:
        l = Input.Linear_Loader(filepaths[0])
        while l.has_next():
            background.append(np.array([l.get_next()]))

    algos = [Algorithm.gradient_shap(background, model, output_size)]

    #       METRICS
    metrics = [Metrics.Average(inside_colour),
               Metrics.N(inside_colour, 1),
               Metrics.N(inside_colour, 2),
               Metrics.N(inside_colour, 3)]

    data = pipeline(input, algos, metrics, model)

    for i, result in enumerate(data):
        for m in range(len(metrics)):
            histogram(result, output + "_" + str(i), m + 6)
            scatterplot(result, output + "_" + str(i), m + 6)
            scatterplot(result, output + "_" + str(i), m + 6, y_name="Average")

        if save_csv:
            print("Saving...\tresults.csv for " + str(i))
            result.to_csv(output + "" + str(i) + ".csv")


"""

    pre_loaded_shap
    
"""


# e.g. filepath = [[list of image paths], [path to csv, target column], [list of segmentation paths]]
def pre_loaded_shap(model, tags, input_size=(225, 300), output_size=(1022, 767),
                    output=os.getcwd(), inside_colour=255,
                    save_matrices=False, save_imgs=False, save_csv=False):
    #       INPUT

    container = [Input.Input_Image(tags[0], input_size),
                 Input.Matrix(tags[1]),
                 Input.Segmentation(tags[2], output_size),
                 Input.Label(tags[3], "benign", "image")]
    input = Input.Sorter(container, 3)

    #       ALGO
    algos = [Algorithm.empty("shap_preloaded")]

    #       METRICS
    metrics = [Metrics.Average(inside_colour),
               Metrics.N(inside_colour, 1),
               Metrics.N(inside_colour, 2),
               Metrics.N(inside_colour, 3),
               Metrics.N(inside_colour, 4)]

    data = pipeline(input, algos, metrics, model)

    for result in data:
        for m in range(len(metrics)):
            o = output + result.name + "_" + metrics[m].name
            print("\t\n" + result.columns[m+6])
            histogram(result, o, m + 6)
            scatterplot(result, o, m + 6)
            scatterplot(result, o, m + 6, y_name="Average")
            scatterplot(result, o, m + 6, y_name="Predicted Class Score")

        if save_csv:
            print("\nSaving...\tresults.csv for " + result.name)
            result.to_csv(output + "" + result.name + ".csv")


"""

    Finn Torbet - 15/13/2020
    BSc Applied Computing Honours Project

"""
