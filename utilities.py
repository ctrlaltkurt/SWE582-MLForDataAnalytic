import numpy as np
import pandas as pd
from scipy.io import arff


class Utilities:

    def read_dataset(data_filepath, labels_filepath):

        data = np.load(data_filepath)
        labels = np.load(labels_filepath)

        return data, labels

    def read_arff(data_filepath):

        data = arff.loadarff(data_filepath)
        data = pd.DataFrame(data[0])
        assert np.sum(np.array(data.isna())) == 0

        data["Class"] = data["Class"].map(lambda x: str(x).split("b'")[-1].rstrip("'"))

        return data

    def get_normalization_values(data, target_column):

        #Area's type is numeric
        #Perimeter's type is numeric
        #Major_Axis_Length's type is numeric
        #Minor_Axis_Length's type is numeric
        #Eccentricity's type is numeric
        #Convex_Area's type is numeric
        #Extent's type is numeric
        #Class's type is nominal, range is ('Cammeo', 'Osmancik')

        normalization_values = dict()
        for col in data.columns:
            if col == target_column:
                continue

            max_v = data[col].max()
            min_v = data[col].min()

            normalization_values[col] = {"max": max_v, "min": min_v}

        return normalization_values

    def normalize_data(data, normalization_values):

        data_norm = data.copy()

        for col, vals in normalization_values.items():
            data_norm[col] = (data_norm[col] - vals["min"]) / (vals["max"] - vals["min"])

        return data_norm

    def prepare_labels(data, target_column, classes):

        labels = np.array(data[target_column])

        for l in np.unique(labels):
            assert l in classes

        labels = np.array(list(map(lambda x: np.where(classes == x)[0][0], labels)), dtype=float)

        return labels

    def split_data(data, labels, validation_ratio):

        n_samples = len(data)
        val_n_samples = int(n_samples * validation_ratio)

        val_indices = np.random.choice(list(range(n_samples)), replace=False, size=val_n_samples)
        train_indices = list(set(list(range(n_samples))).difference(val_indices))

        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

        train_data = np.array([data[i] for i in train_indices])
        train_labels = np.array([labels[i] for i in train_indices])

        val_data = np.array([data[i] for i in val_indices])
        val_labels = np.array([labels[i] for i in val_indices])

        return train_data, train_labels, val_data, val_labels

    def get_rice_cammeo_osmancik_data(data_filepath, validation_ratio):

        target_column = "Class"
        classes = np.array(["Cammeo", "Osmancik"])

        data = Utilities.read_arff(data_filepath)
        normalization_values = Utilities.get_normalization_values(data, target_column)
        data_norm = Utilities.normalize_data(data, normalization_values)

        labels = Utilities.prepare_labels(data_norm, target_column, classes).astype(np.float32)
        data_norm = np.array(data_norm.drop(target_column, axis=1), dtype=np.float32)

        train_data, train_labels, val_data, val_labels = Utilities.split_data(data_norm, labels, validation_ratio=validation_ratio)

        return train_data, train_labels, val_data, val_labels, classes

