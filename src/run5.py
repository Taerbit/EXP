from EXP import Controller
import pandas as pd

shap = pd.read_csv("C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\pipeline\\output\\metric\\shap\\4\\shap.csv")
grad = pd.read_csv("C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\pipeline\\output\\metric\\Grad-CAM\\Dual Classifier\\2\\GradCAM.csv")

shap.name = "SHAP"
grad.name = "Grad-CAM"


metric_names = ['Average', 'N(1)', 'N(2)', 'N(3)']

for m in metric_names:
    Controller.scatterplot_difference_algorithms(grad, shap, m, "C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\pipeline\\output\\metric\\")