#       CONTROLLER
import os
import copy

import numpy as np
import pandas as pd

import EXP.src.Algorithm as Algorithm
import EXP.src.Input as Input
import EXP.src.Metrics as Metrics

import logging
logging.basicConfig(level=logging.INFO)


"""

    Master pipeline to process data taken from input through a list of algorithms and compile metrics
    based off of these algorithm results.
    A list of panda.DataFrames for each algorithm is returned

"""


def pipeline(input, algos, metrics, model):
    #       Setup pandas dataframe columns
    temp = ['ID', 'Correct', 'True Class', 'True Class Score', 'Predicted Class', 'Predicted Class Score']

    # Add all the metrics as columns
    for m in metrics:
        temp.append(m.name)
    x = pd.DataFrame(columns=temp)

    # Create a copy of the pandas Dataframe for each algorithm inside the results list
    results = []
    for a in algos:
        df = x
        df.name = a.name
        results.append(df)

    # Remove the temp variables to set up the dataframes
    del x, m, a, df

    data_counter = 0
    print("Starting Pipeline Loop")

    # Load all the data through the interpretability algorithm and then metrics for each data point
    # whilst there is data from the Input Loader still to be retreived
    while input.has_next():

        # Retreive the loaded data
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


            # Process the meta data for the datapoint (Correct, Score, Predicted, Labelled)
            current_row = add_common_attributes(id, model, data["Input_Image"], data["Label"])

            # Get the input names for this algorithm
            param = a.get_input()
            input_data = []
            # Sort through the dictionary of data loaded in and retreive the requested input
            for p in param:
                input_data.append(data[p])

            matrix = a.pass_through(input_data, id)  # Send image and label away

            # Process algorithm result through all metrics
            for y, m in enumerate(metrics):
                current_row.append(m.process(matrix, data["Segmentation"]))  # Send matrix and segmentation away

            (results[x]).loc[data_counter] = current_row
            data_counter = data_counter + 1
    return results


'''

    Plotting of data
    
    These methods can be called to plot data in particular ways

'''

import seaborn as sns
import matplotlib.pyplot as plt


def histogram(results, save_path, metric_index, model):
    """Histograms a specific metric in a dataframe"""

    correct = results.loc[results['Correct'] == True]
    correct = correct[metric_index]
    incorrect = results.loc[results['Correct'] == False]
    incorrect = incorrect[metric_index]

    ax = sns.distplot(correct, color="g", kde=False)
    ax = sns.distplot(incorrect, color='r', kde=False)
    ax.set(ylabel="Frequency")
    ax.set(title=model + " - Distribution of " + metric_index)

    plt.savefig(save_path + metric_index + "_hist.png")

    plt.clf()


def scatterplot(results, save_path, metric_name, model, y_name="True Class Score", desc=""):
    """Creates a scatterplot for a metric against a specified data series, defaults to correct class score"""

    sns.set_style("whitegrid")
    ax = sns.scatterplot(x=results[metric_name], y=results[y_name], hue="Correct", style="Correct", data=results)

    if desc == "":
        desc = y_name
    ax.set(title=model + " - " + metric_name + " vs " + desc)

    ax.set(xlabel=metric_name)
    ax.set(ylabel=y_name)

    print("Saving...\t" + y_name + " Scatter")

    plt.savefig(save_path + "_" + metric_name + "_" + y_name + "_scatter.png")

    plt.clf()

def boxplot(results, save_path, metric_index, model, x_range=None, x_min=None):
    """Creates a scatterplot for a metric against a specified data series, defaults to correct class score"""

    ax = sns.boxplot(x=results.iloc[:, metric_index], hue="Correct", data=results)

    y_name = results.columns[metric_index]

    ax.set(title=model + " - " + results.columns[metric_index] )
    ax.set(xlim=(x_min, x_range))
    print("Saving...\t" + y_name + " Boxplot")
    plt.savefig(save_path + y_name + "_box.png")
    plt.clf()

def scatterplot_difference_algorithms(dataframe1, dataframe2, metric_name, save_path):
    """This scatterplot shows the difference in the one metric between two dataframes"""
    m1 = dataframe1[metric_name]
    m2 = dataframe2[metric_name]

    sns.scatterplot(x=m1, y=m2, hue="Correct", style="Correct", data=dataframe1)
    plt.xlabel(dataframe1.name)
    plt.ylabel(dataframe2.name)
    plt.title("Comparison of Algorithms for " + metric_name)

    plt.savefig(save_path + metric_name + "_algo_scatter.png")
    plt.clf()

def scatterplot_difference_confidence(dataframe, save_path, metric_name, model):
    """Creates a scatterplot for the difference in incorrect between correct and incorrect classification"""

    # Drop all Correct predictions
    df = dataframe[dataframe["Correct"]== False]

    # Retreive the class scores for predicted and labelled
    true_class_scores = df["True Class Score"]
    pred_class_scores = df["Predicted Class Score"]

    # Find the difference between the two
    differences = pred_class_scores.subtract(true_class_scores)


    # Plot the data against the specified metrix
    sns.scatterplot(x=dataframe[metric_name], y=differences)
    plt.title(model + " - Difference in confidence for incorrect classifications vs " + metric_name)
    plt.xlabel(metric_name)
    plt.ylabel("Predicted Class Scores - Labelled Class Scores")

    plt.savefig(save_path + metric_name + "_confidence_diff_scatter.png")
    plt.clf()

"""

    Sanity Checks


"""
import cv2

def visualization_of_sorted_metric(dataframe, img_paths, metric_name, ascending=True, output=os.getcwd()):
    """Sanity check to compile a video of an algorithms visualizations by a metric sort

    REQUIREMENTS:
        Visualization algorithm images must be the same name as corresponding ID

    PARAMETERS:
        dataframe: A single pandas dataframe including the ID and metric values
        img_paths: A list of absolute paths to the algorithm visualization images
        metric_name: The name of the Series in dataframe that lists the metric values

    """

    # Strip the dataframe to only the ID and the metric
    # Then sort the dataframe by the metric and retreive the names as a list
    df = dataframe[['ID', metric_name]]
    df.sort_values(by=[metric_name], inplace=True, ascending=ascending)
    names = df['ID'].values.tolist()

    images = []
    for n in names:

        # Find the corresponding image for the name, load it and append it to the images list
        for i, p in enumerate(img_paths):

            if n in p:
                logging.info(p + "  ===   " + n)
                img = cv2.imread(p)
                height, width, layers = img.shape
                size = (width, height)
                images.append(img)
                img_paths.pop(i)
                continue

        logging.info("ERROR: " + n + " no image")


    out = cv2.VideoWriter(output + dataframe.name + '-visualization_of_' + metric_name + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 1, size)

    for i in range(len(images)):
        out.write(images[i])
    out.release()

"""
    hons Pipeline  
        The Hons pipeline is the given pipeline for comparing grad_cam and shap against metrics Average and
        N. It takes a specified tag 2D array, model and relevant model parameters.

        The tags lists should contain lists with the front path and the suffix for the data wanted loaded.
        Data MUST be given in the order: INPUT IMAGE, LABEL, SEGMENTATION

"""

# e.g. filepath = [[list of image paths], [path to csv, target column], [list of segmentation paths]]
def hons(model, tags, layer_name, input_size=(300, 225), output_size=(1022, 767),
         output=os.getcwd(), background=[], inside_colour=255,
         save_matrices=False, save_imgs=False, save_csv=False):

    #       INPUT
    # IMPROVE: Validate filepath[n] = expected_input_type[n]
    containers = [Input.Input_Image(tags[0], input_size),
                  Input.Segmentation(tags[1], output_size),
                  Input.Label(tags[2], "benign", "image"),
                  Input.Original_Image(tags[0], input_size, ordered=False),
                  Input.Numbered(tags[3])
                  ]
    input = Input.Sorter(containers)

    #       ALGO

    if background == []:
        l = Input.Linear_Loader(copy.deepcopy(containers[0]))
        while l.has_next():
            background.append(np.array([l.get_next()]))


    if save_imgs:
        img_path = output
    else:
        img_path = ""
    if save_matrices:
        matrix_path = output
    else:
        matrix_path = ""

    algos = [
        #Algorithm.grad_cam(model, output_size, layer_name, img_path=img_path, matrix_path=matrix_path)
        Algorithm.gradient_shap(background, model, output_size, img_path=img_path, matrix_path=matrix_path)
    ]

    #       METRICS
    metrics = [Metrics.Average(inside_colour),
               Metrics.N(inside_colour, 1),
               Metrics.N(inside_colour, 2),
               Metrics.N(inside_colour, 3),
               Metrics.N(inside_colour, 4)]

    data = pipeline(input, algos, metrics, model)

    # Visualize Results from pipeline
    for result in data:

        """
        # Process the plots per metrics
        for m in range(len(metrics)):
            
            o = output + result.name + "_" + metrics[m].name
            print("\t\n" + result.columns[m+7])
            histogram(result, o, m + 7)
            scatterplot(result, o, m + 7)
            scatterplot(result, o, m + 7, y_name="Average")
            scatterplot(result, o, m + 7, y_name="Predicted Class Score")
            scatterplot_difference_confidence(result, o, metrics[m].name)
        """
        if save_csv:
            print("\nSaving...\tresults.csv for " + result.name)
            result.to_csv(output + "" + result.name + ".csv")

    #for m in metrics:
        #scatterplot_difference_algorithms(data[0], data[1], m.name, output)

"""

    pre_loaded_shap
    
"""


# e.g. filepath = [[list of image paths], [path to csv, target column], [list of segmentation paths]]
def pre_loaded_shap(model, tags, input_size=(300, 225), output_size=(1022, 767),
                    output=os.getcwd(), inside_colour=255,
                    save_matrices=False, save_imgs=False, save_csv=False):
    #       INPUT

    container = [Input.Input_Image(tags[0], input_size),
                 Input.Matrix(tags[1]),
                 Input.Segmentation(tags[2], output_size),
                 Input.Label(tags[3], "benign", "image")]
    input = Input.Sorter(container, 0)

    #       ALGO
    algos = [Algorithm.empty("NEW")]

    #       METRICS
    metrics = [Metrics.Average(inside_colour),
               Metrics.N(inside_colour, 1),
               Metrics.N(inside_colour, 2),
               Metrics.N(inside_colour, 3),
               Metrics.N(inside_colour, 4)]

    data = pipeline(input, algos, metrics, model)

    for result in data:
        for m in range(len(metrics)):
            """
            o = output + result.name + "_" + metrics[m].name
            print("\t\n" + result.columns[m+6])
            histogram(result, o, m + 6)
            scatterplot(result, o, m + 6)
            scatterplot(result, o, m + 6, y_name="Average")
            scatterplot(result, o, m + 6, y_name="Predicted Class Score")
            """
        if save_csv:
            print("\nSaving...\tresults.csv for " + result.name)
            result.to_csv(output + "" + result.name + ".csv")


"""

    Finn Torbet - 15/13/2020
    BSc Applied Computing Honours Project

"""
