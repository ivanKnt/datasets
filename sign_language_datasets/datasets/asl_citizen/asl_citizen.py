"""ASL Citizen - A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition"""
import csv
import tarfile
from os import path

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from pose_format import Pose
from pose_format import Pose, PoseHeader
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeaderDimensions
from pose_format.utils.holistic import holistic_components

from sign_language_datasets.utils.features import PoseFeature

from ..warning import dataset_warning
from ...datasets.config import SignDatasetConfig, cloud_bucket_file

_DESCRIPTION = """
The dataset contains about 84k video recordings of 2.7k isolated signs from American Sign Language (ASL).
"""

_CITATION = """
@article{desai2023asl,
  title={ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition},
  author={Desai, Aashaka and Berger, Lauren and Minakov, Fyodor O and Milan, Vanessa and Singh, Chinmay and Pumphrey, Kriston and Ladner, Richard E and Daum{\'e} III, Hal and Lu, Alex X and Caselli, Naomi and Bragg, Danielle},
  journal={arXiv preprint arXiv:2304.05934},
  year={2023}
}
"""

_DOWNLOAD_URL = 'https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip'


_POSE_URLS = {
    "holistic": cloud_bucket_file("poses/holistic/ASLCitizen.zip"),
}
_POSE_HEADERS = {"holistic": path.join(path.dirname(path.realpath(__file__)), "holistic.poseheader")}


class ASLCitizen(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for ASL Citizen dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    BUILDER_CONFIGS = [
        SignDatasetConfig(name="default", include_pose='holistic',include_video=False),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = {
            "id": tfds.features.Text(),
            "video_code": tfds.features.Text(),
            "text": tfds.features.Text(),
            "signer_id": tfds.features.Text(),
            "asl_lex_code": tfds.features.Text(),
        }

        # TODO: add videos

        if self._builder_config.include_pose == "holistic":
            pose_header_path = _POSE_HEADERS[self._builder_config.include_pose]
            stride = 1 if self._builder_config.fps is None else 30 / self._builder_config.fps
            features["pose"] = PoseFeature(shape=(None, 1, 576, 3),
                                           header_path=pose_header_path,
                                           stride=stride)

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage="https://www.microsoft.com/en-us/research/project/asl-citizen/",
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dataset_warning(self)


        data = []
        with GFile(path.join(archive_path, 'ASL_Citizen', 'splits', f"{split}.csv"), "r") as csv_file:
            csv_data = csv.reader(csv_file, delimiter=",")
            next(csv_data)  # Ignore the header

            for i, row in enumerate(csv_data):
                datum = {
                    "id": str(i),
                    "video_code": 5,  #Logic to gather video code here
                    "text": row[2],
                    "signer_id": row[0],
                    "asl_lex_code": row[3],
                }
                data.append(datum)


        #download video if requested
        if self._builder_config.include_video:
            archive = dl_manager.download_and_extract(_DOWNLOAD_URL)
            print("no videos available for now ..")




        if self._builder_config.include_pose:
            poses_dir = str(dl_manager.download_and_extract(_POSE_URLS['holistic']))
            for datum in data:
                pose_file = path.join(poses_dir, 'poses', datum["video_code"]+'.pose')
                datum["pose"] = pose_file if pose_file.exists() else None




        return {"train": self._generate_examples(data)}

    def _generate_examples(self, data):
        """ Yields examples. """
        for datum in data:
            yield datum["id"],datum

