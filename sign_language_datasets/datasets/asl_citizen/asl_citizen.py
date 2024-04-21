"""ASL Citizen - A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition"""
import csv

from os import path
import requests

from io import StringIO
import tensorflow_datasets as tfds


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

    def _download_csv_and_remove_header(self):
        # Send a GET request to the URL
        url = "https://www.mediafire.com/file/9md24gieos6w3cv/train.csv/file"
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful

        # Decode the content to text and use StringIO to treat it as a file-like object
        content = StringIO(response.content.decode('utf-8'))

        # Read the CSV file, removing the header
        csv_data = csv.reader(content, delimiter=",")
        next(csv_data)  # Skip the header

        # Collect all rows into a list
        return list(csv_data)

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dataset_warning(self)


        data = []
        csv_data = self._download_csv_and_remove_header()

        for i, row_val in enumerate(csv_data):
            video_code = row_val[1].replace('.mp4', '')
            datum = {
                "id": str(i),
                "video_code": video_code,
                "text": row_val[2],
                "signer_id": row_val[0],
                "asl_lex_code": row_val[3],
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

