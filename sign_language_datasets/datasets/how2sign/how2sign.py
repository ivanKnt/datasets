"""How2Sign: A multimodal and multiview continuous American Sign Language (ASL) dataset"""
import os
from itertools import chain
from os import path
import requests

import tensorflow as tf
import tensorflow_datasets as tfds
from pose_format.utils.openpose import load_openpose_directory

from ..warning import dataset_warning
from ...datasets.config import SignDatasetConfig
from ...utils.features import PoseFeature

_DESCRIPTION = """
A multimodal and multiview continuous American Sign Language (ASL) dataset, 
consisting of a parallel corpus of more than 80 hours of sign language videos and a set of corresponding modalities 
including speech, English transcripts, and depth.
"""

_CITATION = """
@inproceedings{Duarte_CVPR2021,
    title={{How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language}},
    author={Duarte, Amanda and Palaskar, Shruti and Ventura, Lucas and Ghadiyaram, Deepti and DeHaan, Kenneth and
                   Metze, Florian and Torres, Jordi and Giro-i-Nieto, Xavier},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021}
}
"""

_SPLITS = {
    tfds.Split.TRAIN: {
        "rgb_clips_front": "https://drive.google.com/uc?id=1VX7n0jjW0pW3GEdgOks3z8nqE6iI6EnW&export=download",
        "rgb_clips_side": "https://drive.google.com/uc?id=1oiw861NGp4CKKFO3iuHGSCgTyQ-DXHW7&export=download",
        "bfh_2d_front": "https://drive.google.com/uc?id=1TBX7hLraMiiLucknM1mhblNVomO9-Y0r&export=download",
    },
    tfds.Split.VALIDATION: {
        "rgb_clips_front": "https://drive.google.com/uc?id=1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO&export=download",
        "rgb_clips_side": "https://drive.google.com/uc?id=1mxL7kJPNUzJ6zoaqJyxF1Krnjo4F-eQG&export=download",
        "bfh_2d_front": "https://drive.google.com/uc?id=1JmEsU0GYUD5iVdefMOZpeWa_iYnmK_7w&export=download",
    },
    tfds.Split.TEST: {
        "rgb_clips_front": "https://drive.google.com/uc?id=1qTIXFsu8M55HrCiaGv7vZ7GkdB3ubjaG&export=download",
        "rgb_clips_side": "https://drive.google.com/uc?id=1j9v9P7UdMJ0_FVWg8H95cqx4DMSsrdbH&export=download",
        "bfh_2d_front": "https://drive.google.com/uc?id=1g8tzzW5BNPzHXlamuMQOvdwlHRa-29Vp&export=download",
    },
}

_POSE_HEADERS = {"openpose": path.join(path.dirname(path.realpath(__file__)), "openpose.header")}
def download_file_from_google_drive(url, session=None):
    if session is None:
        session = requests.Session()
    response = session.get(url, stream=True)
    for key, value in response.cookies.items():
        if 'download_warning' in key:
            token = value
            url = url + "&confirm=" + token
            response = session.get(url, stream=True)
            break
    return response

class How2Sign(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for how2sign dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    BUILDER_CONFIGS = [SignDatasetConfig(name="default", include_video=True, include_pose="openpose")]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = {"id": tfds.features.Text(), "fps": tf.int32}

        if self._builder_config.include_video:
            features["video"] = {
                "front": self._builder_config.video_feature((1280, 720)),
                "side": self._builder_config.video_feature((1280, 720)),
            }

        if self._builder_config.include_pose == "openpose":
            pose_header_path = _POSE_HEADERS[self._builder_config.include_pose]
            stride = 1 if self._builder_config.fps is None else 24 / self._builder_config.fps
            features["pose"] = {
                "front": PoseFeature(shape=(None, 1, 137, 2), header_path=pose_header_path, stride=stride),
                # "side": PoseFeature(shape=(None, 1, 137, 2), header_path=pose_header_path, stride=stride),
            }

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage="https://how2sign.github.io/",
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators, integrating custom download logic."""
        dataset_warning(self)
        session = requests.Session()

        downloads = {}
        for split_name, split_dict in _SPLITS.items():
            for key, url in split_dict.items():
                if url is not None:
                    if "drive.google.com" in url:
                        response = download_file_from_google_drive(url, session)
                        # Assuming the directory to save files is predefined or configurable
                        file_path = path.join(dl_manager.download_dir, f"{split_name}_{key}.mp4")
                        with open(file_path, "wb") as f:
                            f.write(response.content)
                        downloads[key] = file_path
                    else:
                        downloads[key] = dl_manager.download(url)

        return [
            tfds.core.SplitGenerator(
                name=name,
                gen_kwargs={k: downloads[k] for k, v in split.items() if k in downloads},
            ) for name, split in _SPLITS.items()
        ]

    def _generate_examples(self, rgb_clips_front: str, rgb_clips_side: str, bfh_2d_front: str):
        """ Yields examples. """

        # TODO get ids from translation file
        ids = []
        ids = [p[: -len("-rgb_front.mp4")] for p in os.listdir(path.join(rgb_clips_front, "raw_videos"))]
        ids = ids[:10]

        for _id in ids:
            datum = {
                "id": _id,
                "fps": 24,
            }

            if self.builder_config.include_video:
                datum["video"] = {
                    "front": path.join(rgb_clips_front, "raw_videos", _id + "-rgb_front.mp4"),
                    "side": path.join(rgb_clips_side, "raw_videos", _id + "-rgb_side.mp4"),
                }

            if self._builder_config.include_pose == "openpose":
                front_path = path.join(bfh_2d_front, "openpose_output", "json", _id + "-rgb_front")
                front_pose = load_openpose_directory(front_path, fps=24, width=1280, height=720)

                # TODO add side pose when available
                # side_path = path.join(bfh_2d_side, 'openpose_output', 'json', _id + '-rgb_side')
                # side_pose = load_openpose_directory(side_path, fps=24, width=1280, height=720)

                datum["pose"] = {"front": front_pose}

            yield _id, datum
