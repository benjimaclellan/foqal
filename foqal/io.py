import pathlib
import warnings
import os
import datetime
import numpy as np
import json
import string
import pandas as pd
import random


def current_time():
    """
    Common format for current date & time
    :return: date and time as formatted string
    """
    return datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")


class IO:
    """
    The IO class encapsulates all saving/loading features of data, figures, etc.
    This provides consistent filetypes, naming conventions, etc.
    """

    # default save path always points to ../data no matter where this repository is located
    default_path = pathlib.Path(__file__).parent.parent.joinpath("data")

    def __init__(self, path=None, verbose=True):
        self.verbose = verbose

        # set default path to a 'data' folder
        if path is None:
            path = self.default_path
        if type(path) is str:
            path = pathlib.Path(path)

        self.path = pathlib.Path(path)
        return

    @classmethod
    def directory(
        cls,
        path=None,
        folder="",
        include_date=False,
        include_time=False,
        include_id=False,
        verbose=True,
    ):
        """

        :param path: The parent folder.
        :param folder: The main, descriptive folder name.
        :param include_date: If True, add the date to the front of the path. Otherwise, do not add the date
        :param include_time: If True, add the time to the front of the path. Otherwise, do not add the time
        :param include_id: If True, add a random string of characters to the end of the path. Otherwise, do not
        :param verbose: If True, will print out the path of each saved/loaded file.
        :return: A new IO class instance
        """
        if path is None:
            path = cls.default_path

        if type(path) is str:
            path = pathlib.Path(path)

        date = datetime.date.today().isoformat()
        time = datetime.datetime.now().strftime("%H-%M-%S")
        if not folder:  # if empty string
            warnings.warn(
                "No folder entered. Saving to a folder with a unique identifier"
            )
            include_data, include_id, verbose = True, True, True

        # build the full folder name with date, time, and uuid, if selected
        _str = ""
        if include_date:
            _str = _str + date
        if include_time:
            _str = _str + "_" + time

        _str = _str + "_" + folder

        if include_id:
            _str = (
                _str + "_" + "".join(random.choice(string.hexdigits) for _ in range(4))
            )

        path = path.joinpath(_str)
        return cls(path=path, verbose=verbose)

    def save_json(self, variable, filename):
        full_path = self.path.joinpath(filename)
        os.makedirs(full_path.parent, exist_ok=True)
        self._save_json(variable, full_path)
        if self.verbose:
            print(f"{current_time()} | Saved to {full_path} successfully.")

    def load_json(self, filename):
        full_path = self.path.joinpath(filename)
        file = self._load_json(full_path)
        if self.verbose:
            print(f"{current_time()} | Loaded from {full_path} successfully.")
        return file

    def save_dataframe(self, df, filename):
        full_path = self.path.joinpath(filename)
        os.makedirs(full_path.parent, exist_ok=True)
        df.to_csv(str(full_path), sep=",", index=False, header=True)
        if self.verbose:
            print(f"{current_time()} | Saved to {full_path} successfully.")

    def load_dataframe(self, filename):
        full_path = self.path.joinpath(filename)
        df = pd.read_csv(str(full_path), sep=",", header=0)
        if self.verbose:
            print(f"{current_time()} | Loaded from {full_path} successfully.")
        return df

    def save_figure(self, fig, filename):
        full_path = self.path.joinpath(filename)
        os.makedirs(full_path.parent, exist_ok=True)
        fig.savefig(full_path, dpi=300, bbox_inches="tight")
        if self.verbose:
            print(f"{current_time()} | Saved figure to {full_path} successfully.")

    def save_np_array(self, variable, filename):
        full_path = self.path.joinpath(filename)
        os.makedirs(full_path.parent, exist_ok=True)
        np.save(str(full_path), variable)
        # np.savetxt(str(full_path), variable)
        if self.verbose:
            print(f"{current_time()} | Saved to {full_path} successfully.")

    def load_np_array(self, filename, complex=False):
        full_path = self.path.joinpath(filename)
        file = np.load(str(full_path))
        # file = np.loadtxt(str(full_path), dtype=np.complex if complex else np.float)
        if self.verbose:
            print(f"{current_time()} | Loaded from {full_path} successfully.")
        return file

    @staticmethod
    def _save_json(variable, path):
        with open(path, "w+") as json_file:
            json.dump(variable, json_file, indent=4)

    @staticmethod
    def _load_json(path):
        with open(path) as json_file:
            data = json.load(json_file)
        return data
