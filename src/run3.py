import Controller
import pandas as pd
output_dir = "C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\pipeline\\output\\metric\\shap\\2\\"

results = pd.read_csv(output_dir + "0.csv")
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
for m in range(5):
    o = output_dir +  metric_names[m]
    Controller.histogram(results, o, m + 7)
    Controller.scatterplot(results, o, m + 7, x=explainer[m])
    Controller.scatterplot(results, o, m + 7, y_name="Average", y="Average Inside / Average Outside", x=explainer[m])
    Controller.scatterplot(results, o, m + 7, y_name="Predicted Class Score", x=explainer[m])