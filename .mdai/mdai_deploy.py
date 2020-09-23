import os
import sys
from io import BytesIO
import pydicom
import torch
import cv2
from PIL import Image
import numpy as np
import yaml
from mmcv.parallel import collate

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, MODEL_PATH)
from factory import set_reproducibility

import factory.evaluate as evaluate
import factory.builder as builder
import factory.models as models


class MDAIModel:
    def __init__(self):
        root_path = os.path.dirname(os.path.dirname(__file__))

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            self.device = torch.device("cpu")
            gpu_ids = []

        with open(os.path.join(root_path, "src", "configs", "experiment001.yaml")) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        torch.hub.set_dir("/tmp")
        self.model = builder.build_model(self.cfg, 0)
        self.model.load_state_dict(
            torch.load(
                # self.cfg["predict"]["checkpoint"],
                os.path.join(
                    root_path, "checkpoints", "experiment001", "RET50_019_VM-0.2294.PTH"
                ),
                map_location=lambda storage, loc: storage,
            )
        )
        self.model = self.model.eval()

    def prepare_input(self, image):
        # Need to save temporary image file
        rand_img = os.path.join("/tmp", "img-{}.png".format(np.random.randint(1e10)))

        # Assuming image is (H, W)
        if image.ndim == 2:
            X = np.expand_dims(image, axis=-1)
            X = np.repeat(X, 3, axis=-1)
        else:
            X = image
        cv2.imwrite(rand_img, X)
        ann = [{"filename": rand_img, "height": X.shape[0], "width": X.shape[1]}]
        self.cfg["dataset"]["data_dir"] = "/tmp"
        loader = builder.build_dataloader(self.cfg, ann, mode="predict")
        data = loader.dataset[0]
        img = data.pop("img")
        img_meta = {0: data}
        os.system("rm {}".format(rand_img))
        return img, img_meta

    def _predict(self, x):
        img, img_meta = self.prepare_input(x)

        with torch.no_grad():
            output = self.model(
                [img.unsqueeze(0)], img_meta=[img_meta], return_loss=False, rescale=True
            )
        output = output[0]

        threshold = 0.2  # adjust as necessary
        output = output[output[:, -1] >= threshold]
        return output

    def predict(self, data):
        """
        The input data has the following schema:

        {
            "instances": [
                {
                    "file": "bytes"
                    "tags": {
                        "StudyInstanceUID": "str",
                        "SeriesInstanceUID": "str",
                        "SOPInstanceUID": "str",
                        ...
                    }
                },
                ...
            ],
            "args": {
                "arg1": "str",
                "arg2": "str",
                ...
            }
        }

        Model scope specifies whether an entire study, series, or instance is given to the model.
        If the model scope is 'INSTANCE', then `instances` will be a single instance (list length of 1).
        If the model scope is 'SERIES', then `instances` will be a list of all instances in a series.
        If the model scope is 'STUDY', then `instances` will be a list of all instances in a study.

        The additional `args` dict supply values that may be used in a given run.

        For a single instance dict, `files` is the raw binary data representing a DICOM file, and
        can be loaded using: `ds = pydicom.dcmread(BytesIO(instance["file"]))`.

        The results returned by this function should have the following schema:

        [
            {
                "type": "str", // 'NONE', 'ANNOTATION', 'IMAGE', 'DICOM', 'TEXT'
                "study_uid": "str",
                "series_uid": "str",
                "instance_uid": "str",
                "frame_number": "int",
                "class_index": "int",
                "data": {},
                "probability": "float",
                "explanations": [
                    {
                        "name": "str",
                        "description": "str",
                        "content": "bytes",
                        "content_type": "str",
                    },
                    ...
                ],
            },
            ...
        ]

        The DICOM UIDs must be supplied based on the scope of the label attached to `class_index`.
        """
        input_instances = data["instances"]
        input_args = data["args"]

        results = []

        for instance in input_instances:
            tags = instance["tags"]
            try:
                ds = pydicom.dcmread(BytesIO(instance["file"]))
                arr = ds.pixel_array
                output = self._predict(arr)
                if len(output) == 0:
                    result = {
                        "type": "NONE",
                        "study_uid": tags["StudyInstanceUID"],
                        "series_uid": tags["SeriesInstanceUID"],
                        "instance_uid": tags["SOPInstanceUID"],
                        "frame_number": None,
                    }
                    results.append(result)
                else:
                    for i in range(len(output)):
                        elem = output[i]
                        data = {
                            "x": int(elem[0]),
                            "y": int(elem[2]),
                            "width": int(elem[1]) - int(elem[0]),
                            "height": int(elem[3]) - int(elem[4]),
                        }
                        result = {
                            "type": "ANNOTATION",
                            "study_uid": tags["StudyInstanceUID"],
                            "series_uid": tags["SeriesInstanceUID"],
                            "instance_uid": tags["SOPInstanceUID"],
                            "frame_number": None,
                            "class_index": 0,
                            "probability": float(elem[4]),
                            "data": data,
                        }
                        results.append(result)
            except:
                e = sys.exc_info()[0]
                print(e)
                continue
        return results