import os
import sys
import six
import csv

from framework.keras.autoencoder import VariationalAutoenocder as VAE

experiment_name = "train_usokin-100g-standardized-1layer-vae"

root_dir = "/Users/kevin/Documents/Research/XinghuaLuLab/single-cell-deep-learning"
experiment_dir = root_dir + "/results/" + experiment_name

def save_data_table(data, filepath, root=None, delimiter="\t"):
    if root is not None:
        filepath = root + "/" + filepath

    delimiter = str(delimiter) if six.PY2 else delimiter

    with open(filepath, "w") as f:
        writer = csv.writer(
            f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        for r in data:
            writer.writerow(r)

model_losses = {}
with open(experiment_dir + "/experiment.log", "r") as f:
    for line in f.readlines():
        if "Avg Validation Loss" in line:
            loss = line.rstrip("\n").split("Avg Validation Loss = ")[-1]
            model_name = line.split(" - ")[1].split("|")[0]
            model_losses[model_name] = float(loss)

experiment_results = [[
    "model_name",
    "encoder_layers",
    "latent_size",
    "optimizer",
    "batch_size",
    "10foldcv_loss"
]]

for model_name in os.listdir(experiment_dir):
    model_dir = os.path.join(experiment_dir, model_name)
    if os.path.isfile(model_dir) or "_FINAL" in model_name: continue
    model_config = VAE.load_config(model_dir)

    experiment_results.append([
        model_name,
        str("|".join(model_config["encoder_layers"])),
        model_config["latent_size"],
        model_config["optimizer"],
        model_config["batch_size"],
        model_losses[model_name]
    ])

save_data_table(
    experiment_results,
    experiment_dir + "/experiment_results.txt")
