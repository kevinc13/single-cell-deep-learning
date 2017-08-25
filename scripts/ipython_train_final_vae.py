import six
import csv
import os

from framework.common.dataset import Dataset
from framework.common.sampling import stratified_kfold
from framework.keras.autoencoder import VariationalAutoencoder as VAE
import numpy as np

exp_name = "train_usokin-1000g-2layer-vae"
ref_model = "11_UsokinVAE"
model_config = VAE.load_config("results/{}/{}".format(exp_name, ref_model))
model_config["name"] = "UsokinVAE_BestTotalLoss"
model_config["model_dir"] = \
    "/pylon5/mc4s8ap/kchen8/single-cell-deep-learning/results/{}/{}".format(
        exp_name, model_config["name"])
model_config["tensorboard"] = True
model_config["bernoulli"] = False
model_config["checkpoint"] = True
model_config["early_stopping_metric"] = "loss"
model_config["checkpoint_metric"] = "loss"

if not os.path.exists(model_config["model_dir"]):
    os.makedirs(model_config["model_dir"])


def read_data_table(filepath, delimiter="\t"):
    with open(filepath, "r") as f:
        data = []
        for line in f.readlines():
            data.append(line.replace("\n", "").split(delimiter))

        return data


def load_data():
    df = np.array(read_data_table(
        "data/Usokin/processed/usokin.1000g.standardized.txt"))
    features = df[1:, 1:-2]

    cell_ids = df[1:, 0]
    cell_types = df[1:, -2]
    cell_subtypes = df[1:, -1]

    return cell_ids, features, cell_types, cell_subtypes


def save_data_table(data, filepath, root=None, delimiter="\t"):
    if root is not None:
        filepath = root + "/" + filepath

    delimiter = str(delimiter) if six.PY2 else delimiter

    with open(filepath, "w") as f:
        writer = csv.writer(
            f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        for r in data:
            writer.writerow(r)


cell_ids, features, cell_types, cell_subtypes = load_data()

datasets = stratified_kfold(
    features, cell_subtypes,
    [cell_ids, cell_types, cell_subtypes],
    n_folds=5, convert_labels_to_int=True)
full_dataset = Dataset.concatenate(*datasets)
n_epochs = 200

final_vae = VAE(model_config)
final_vae.train(full_dataset,
                epochs=n_epochs, batch_size=model_config["batch_size"])
loss = final_vae.evaluate(full_dataset)
print(loss)

latent_reps = final_vae.encode(full_dataset.features)
results = np.hstack((
    np.expand_dims(full_dataset.sample_data[0], axis=1),
    latent_reps,
    np.expand_dims(full_dataset.sample_data[1], axis=1),
    np.expand_dims(full_dataset.sample_data[2], axis=1)
))

header = ["cell_ids"]
for l in range(1, model_config["latent_size"] + 1):
    header.append("dim{}".format(l))
header.append("cell_type")
header.append("cell_subtype")
header = np.array(header)

results = np.vstack((header, results))

print("Saving results")
save_data_table(
    results,
    model_config["model_dir"] + "/latent_representations.txt")
