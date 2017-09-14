import os
import six
import csv

from framework.keras.autoencoder import VariationalAutoencoder as VAE

experiment_name = "usokin/experiment_1c/train_usokin-500g-1layer-vae-wu"

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


total_losses = {}
recon_losses = {}
kl_losses = {}
with open(experiment_dir + "/experiment.log", "r") as f:
    for line in f.readlines():
        if "Avg loss" in line:
            loss = line.rstrip("\n").split("Avg loss = ")[-1]
            model_name = line.split(" - ")[1].split("|")[0]
            total_losses[model_name] = float(loss)

        if "Avg reconstruction_loss" in line:
            loss = line.rstrip("\n").split("Avg reconstruction_loss = ")[-1]
            model_name = line.split(" - ")[1].split("|")[0]
            recon_losses[model_name] = float(loss)

        if "Avg kl_divergence_loss" in line:
            loss = line.rstrip("\n").split("Avg kl_divergence_loss = ")[-1]
            model_name = line.split(" - ")[1].split("|")[0]
            kl_losses[model_name] = float(loss)

experiment_results = [[
    "model_name",
    "encoder_layers",
    "latent_size",
    "optimizer",
    "n_warmup_epochs",
    "batch_size",
    "cv_reconstruction_loss",
    "cv_kl_divergence_loss",
    "cv_total_loss"
]]

for model_name in os.listdir(experiment_dir):
    model_dir = os.path.join(experiment_dir, model_name)
    if os.path.isfile(model_dir) or "_FINAL" in model_name: continue
    model_config = VAE.load_config(model_dir)

    recon_loss = recon_losses[model_name] if model_name in recon_losses \
        else "NaN"
    kl_loss = kl_losses[model_name] if model_name in kl_losses \
        else "NaN"
    total_loss = total_losses[model_name] if model_name in total_losses \
        else "NaN"

    experiment_results.append([
        model_name,
        str("|".join(model_config["encoder_layers"])),
        model_config["latent_size"],
        model_config["optimizer"],
        model_config["n_warmup_epochs"],
        model_config["batch_size"],
        recon_loss,
        kl_loss,
        total_loss
    ])

save_data_table(
    experiment_results,
    experiment_dir + "/experiment_results.txt")
