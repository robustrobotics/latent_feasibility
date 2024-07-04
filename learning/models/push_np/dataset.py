import argparse
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import pickle
import os

from pb_robot.planners.antipodalGraspPlanner import GraspableBody


class PushNPDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        # print(data)
        print(len(data))
        data = self.process_data(data)
        print(len(data))
        for key, attribute_list in data.items():
            pass

    def process_data(self, data) -> dict:
        assert data is not None
        dictionary = {}
        for instance1 in data:
            for instance in instance1:
                # print(type(instance))
                # a = instance[0], b = instance[1]
                # angle, contact_point, body, push_velocity = a
                # success, logs = b
                ((angle, contact_point, body, push_velocity), (success, logs)) = (
                    instance
                )
                assert isinstance(body, GraspableBody)
                # Turning namedtuple into regular tuple so it can be a key to a dictionary
                key = (body[0], tuple(body[1]), body[2], body[3])
                # print(body)
                # key = body
                # TODO: Change this from being just last element
                # print(key)
                if key not in dictionary:
                    dictionary[key] = []
                dictionary[key].append(
                    (angle, contact_point, push_velocity, success, logs[-1])
                )
        return dictionary

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


def main(args):
    # file_name = os.path.exists(args.file_name)

    file_name = os.path.join(
        "learning", "data", "pushing", "test_run", "train_dataset.pkl"
    )
    # print(file_name)
    dataset = PushNPDataset(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize final positions and orientations of sliders from pushing data."
    )
    parser.add_argument(
        "file_name",
        type=str,
        help="Name of the file that you want to test PushNP Dataset on.",
    )
    args = parser.parse_args()
    main(args)
