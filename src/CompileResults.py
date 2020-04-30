import pandas as pd
import seaborn as sns
import Controller

threshold = 0.5

output = "C:\\Users\\finnt\\Documents\Honours Results\\plots\\"

models = [

    # DenseNet201
    [
        #GradCAM
    "C:\\Users\\finnt\\Documents\Honours Results\\200416_DenseNet201_001\\grad_cam\\NEW.csv",
        #GradientSHAP
    "C:\\Users\\finnt\\Documents\Honours Results\\200416_DenseNet201_001\\shap\\NEW.csv"
    ],

    # EfficientNetB0
    [
        # GradCAM
        "C:\\Users\\finnt\\Documents\Honours Results\\200324_EfficientNetB0NoisyStudent_001\\grad_cam\\NEW.csv",
        # GradientSHAP
        "C:\\Users\\finnt\\Documents\Honours Results\\200324_EfficientNetB0NoisyStudent_001\\shap\\NEW.csv"
    ],

    # EfficientNetB7
    [
        # GradCAM
        "C:\\Users\\finnt\\Documents\Honours Results\\200411_EfficientNetB7NoisyStudent_001\\grad_cam\\NEW.csv",
        # GradientSHAP
        #"C:\\Users\\finnt\\Documents\Honours Results\\200411_EfficientNetB7NoisyStudent_001\\shap\\200411_EfficientNetB7NoisyStudent_001_shap.csv"
    ]
]

model_names = ["DenseNet201", "EfficientNetB0", "EfficientNetB7"]

algo_names = ["GradCAM", "GradientSHAP"]

metrics = [
    "Average",
    "N(1)",
    "N(2)",
    "N(3)",
    "N(4)"
]

#Construct a confusion matrix based off a pandas series of correct values and metric
def calculate_confusion_matrix(df, metric_name):

    C = df['Correct'] == True
    I = df['Correct'] == False

    GT = df[metric_name] >= threshold
    LT = df[metric_name] < threshold

    C_GT = len(df[C & GT].index)
    C_LT = len(df[C & LT].index)
    I_GT = len(df[I & GT].index)
    I_LT = len(df[I & LT].index)

    return C_GT, C_LT, I_GT, I_LT


results = []
c = ['Model', 'Algorithm', 'Metric', 'Correct and N>1', 'Correct and N<1', 'Incorrect and N>1', 'Incorrect and N<1', 'Localization Rate', 'Localization and Accuracy Rate', 'Lesion-Focussed Accuracy', 'Inaccuracy due to Noise', 'Lesion-Focussed Inaccuracy']
df_threshold_metrics = pd.DataFrame(columns=c)
data_counter = 0
# For each model
for i, m in enumerate(models):
    print("\t" + model_names[i])

    #Process GradCAM and SHAP
    algos = []
    for aN, a in enumerate(m):

        df = pd.read_csv(a)

        print(algo_names[aN])

        # Process each metric
        for metN, met in enumerate(metrics):

            # Skip Average for threshold measurements
            if metN != 0:
                C_GT, C_LT, I_GT, I_LT = calculate_confusion_matrix(df, met)

                total = C_GT + C_LT + I_LT + I_GT

                LI = ((C_GT + I_GT) / total) * 100
                LCA = (C_GT / total) * 100
                ILA = (C_GT / (C_GT + I_GT)) * 100
                ILP = (I_LT / (I_LT + I_GT)) * 100
                LFIA = (I_LT / total) * 100

                currentrow = [model_names[i], algo_names[aN], met, C_GT, C_LT, I_GT, I_LT, LI, LCA, ILA, ILP, LFIA]
                df_threshold_metrics.loc[data_counter] = currentrow
                data_counter += 1
            """
            Controller.scatterplot(df, output + model_names[i] + "\\scatter\\" + algo_names[aN], met,
                                   model_names[i], desc="Labelled Class softmax output")
            Controller.scatterplot(df, output + model_names[i] + "\\scatter\\" + algo_names[aN], met,
                                   model_names[i], y_name="Predicted Class Score", desc="Prediction softmax output")
            Controller.boxplot(df, output + model_names[i] + "\\boxplot\\" + algo_names[aN] + "_", 7 + metN, model_names[i])
            Controller.histogram(df, output + model_names[i] + "\\histograms\\" + algo_names[aN] + "_", met, model_names[i])
            Controller.scatterplot_difference_confidence(df, output + model_names[i] + "\\scatter\\" + algo_names[aN],
                                                         met, model_names[i])
            """
import matplotlib.pyplot as plt

xlabels = [ "Frequency (%) of High Activations within Lesion\n(High Activations within Lesion = N > " + str(threshold) + ")",
            "Frequency (%) of all Predicitons that Activate Highly inside the Lesion and are Correct\n(Activate Highly within Lesion = N > " + str(threshold) + ")",
            "Frequency (%) of Predictions that are Correct and Activating Highly within the Lesion\n(Activate Highly within Lesion = N > " + str(threshold) + ")",
            "Frequency (%) of Incorrect decisions Activating Higher outside the Lesion\n(High Activations outside the Lesion = N < " + str(threshold) + ")",
            "Frequency (%) of all Predictions that Activate Lowly inside the Lesion and are are Incorrect\n(Activate Lowly inside the Lesion = N < " + str(threshold) + ")"
            ]

df_threshold_metrics.to_csv(output + "threshold\\" + str(threshold) + "\\threshold metrics.csv")
for i in range(5):
    ax = sns.catplot(x=c[i + 7], y="Model", hue="Algorithm", kind="bar", data=df_threshold_metrics)
    ax.set(xlabel=xlabels[i])
    plt.savefig(output + "threshold\\" + str(threshold) + "\\" + c[i + 7] + ".png")
    plt.clf()
