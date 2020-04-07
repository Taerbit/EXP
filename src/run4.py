from EXP import Controller, Input
import pandas as pd

df = pd.read_csv("C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\pipeline\\output\\metric\\Grad-CAM\\5\\.c")  #"C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\pipeline\\output\\metric\\shap\\4\\shap.csv")
paths = Input.get_all_paths("C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\pipeline\\output\\overlapped\\Grad-CAM\\")  # "C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\pipeline\\output\\overlapped\\SHAP\\Dual Classifier\\", "*.png")

df.name = "Grad-CAM" #"SHAP"

Controller.visualization_of_sorted_metric(
    df,
    paths,
    "N(2)",
    output="C:\\Users\\finnt\\OneDrive\\Documents\\Uni\\Year 4\\Honours\\Project\\pipeline\\output\\metric\\shap\\4\\"
)