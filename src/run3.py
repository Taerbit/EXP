import Controller
import pandas as pd
output_dir = "C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\pipeline\\output\\metric\\shap\\2\\"

results = pd.read_csv("C:\\Users\\finnt\\Documents\\Honours Results\\200416_DenseNet201_001\\grad_cam\\200416_DenseNet201_001_grad_cam.csv")
metric_names = [
    "Average",
    "N(1)",
    "N(2)",
    "N(3)",
    "N(4)"
]
explainer = [
    "Average Inside / Average Outside",
    "Ratio of highest values inside the segmentation  (Inside/Outside)"
]

m = 1
Controller.boxplot(results, output_dir, m+7)
Controller.scatterplot(results, output_dir, metric_names[m], "Predicted Class Score", x_range=5)
#results = results[results[metric_names[m]] > 0]


#Controller.histogram(results, output_dir, 7+m)
#Controller.scatterplot(results, output_dir, metric_names[m], "Predicted Class Score")

"""
for m in range(5):
    o = output_dir +  metric_names[m]
    #Controller.histogram(results, o, m + 7)
    Controller.scatterplot(results, o, m + 7, x=explainer[m])
    Controller.scatterplot(results, o, m + 7, y_name="Average", y="Average Inside / Average Outside", x=explainer[m])
    Controller.scatterplot(results, o, m + 7, y_name="Predicted Class Score", x=explainer[m])
"""