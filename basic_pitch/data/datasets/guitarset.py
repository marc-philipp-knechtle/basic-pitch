#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2024 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import random
import time

from typing import Any, List, Dict, Tuple, Optional

import apache_beam as beam
import mirdata

from basic_pitch.data import commandline, pipeline


class GuitarSetInvalidTracks(beam.DoFn):
    """
    This Implementation does not filter out anything, it is just a conversion in the apache beam workflow.
    """

    def process(self, element: Tuple[str, str], *args: Tuple[Any, Any], **kwargs: Dict[str, Any]) -> Any:
        """
        The *args and **kwargs are required for the beam.DoFn interface, but are not used in this function.
        """
        track_id, split = element
        yield beam.pvalue.TaggedOutput(split, track_id)


class GuitarSetToTfExample(beam.DoFn):
    DOWNLOAD_ATTRIBUTES = ["audio_mic_path", "jams_path"]

    def __init__(self, source: str, download: bool) -> None:
        self.source = source
        self.download = download

    def setup(self) -> None:
        import apache_beam as beam
        import mirdata

        self.guitarset_remote = mirdata.initialize("guitarset", data_home=self.source)
        self.filesystem = beam.io.filesystems.FileSystems()  # TODO: replace with fsspec
        if self.download:
            self.guitarset_remote.download()

    def process(self, element: List[str], *args: Tuple[Any, Any], **kwargs: Dict[str, Any]) -> List[Any]:
        """

        Args:
            element: List of track_ids to process in this batch
            *args: Empty for this case
            **kwargs: Empty for this case

        Returns: batch with all the data
        """
        import tempfile

        import mirdata
        import numpy as np
        import sox

        from basic_pitch.constants import (
            AUDIO_N_CHANNELS,
            AUDIO_SAMPLE_RATE,
            FREQ_BINS_CONTOURS,
            FREQ_BINS_NOTES,
            ANNOTATION_HOP,
            N_FREQ_BINS_NOTES,
            N_FREQ_BINS_CONTOURS,
        )
        from basic_pitch.data import tf_example_serialization

        logging.info(f"Processing {element}")
        batch = []

        for track_id in element:
            track_remote = self.guitarset_remote.track(track_id)
            with tempfile.TemporaryDirectory() as local_tmp_dir:
                guitarset_local = mirdata.initialize("guitarset", local_tmp_dir)
                track_local = guitarset_local.track(track_id)

                for attribute in self.DOWNLOAD_ATTRIBUTES:
                    source = getattr(track_remote, attribute)
                    destination = getattr(track_local, attribute)
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    with self.filesystem.open(source) as s, open(destination, "wb") as d:
                        d.write(s.read())

                local_wav_path = f"{track_local.audio_mic_path}_tmp.wav"

                # SOX is used to convert the audio to the correct format
                tfm = sox.Transformer()
                tfm.rate(AUDIO_SAMPLE_RATE)
                tfm.channels(AUDIO_N_CHANNELS)
                tfm.build(track_local.audio_mic_path, local_wav_path)

                duration = sox.file_info.duration(local_wav_path)
                """duration in seconds"""
                time_scale = np.arange(0, duration + ANNOTATION_HOP, ANNOTATION_HOP)
                """
                length = n_time_frames
                shape = (n_time_frames)
                defines the real time (start) of each frame index
                """
                n_time_frames = len(time_scale)

                note_indices: np.ndarray
                """
                shape = (num_time_note_combis, 2)
                defines the time index and frequency index for notes 
                max_value: n_time_frames -> error if it's longer than that... 
                """
                note_velocities: np.ndarray
                """
                shape = (num_time_note_combis)
                amplitude (np.ndarray): Array of amplitude values for each index
                This represents the velocity for each note. 
                However since some dataset don't over this information, it is empty. 
                """
                note_indices, note_velocities = track_local.notes_all.to_sparse_index(
                    time_scale, "s", FREQ_BINS_NOTES, "hz"
                )

                onset_indices: np.ndarray
                """
                shape = (num_notes, 2)
                describes each onset with time and frequency index
                you can also see the onsets by comparing note_indices with onset_indices
                """
                onset_velocities: np.ndarray
                """
                shape = (num_notes)
                same as note_velocities, but for onsets
                """
                onset_indices, onset_velocities = track_local.notes_all.to_sparse_index(
                    time_scale, "s", FREQ_BINS_NOTES, "hz", onsets_only=True
                )

                contour_indices: np.ndarray
                """
                shape = (num_time_contour_combis, 2)
                actually almost the same as the other indices
                describes the time and and frequency for the contours
                """
                contour_velocities: np.ndarray
                """
                shape = (num_time_contour_combis)
                """
                contour_indices, contour_velocities = track_local.multif0.to_sparse_index(
                    time_scale, "s", FREQ_BINS_CONTOURS, "hz"
                )

                batch.append(
                    tf_example_serialization.to_transcription_tfexample(
                        track_local.track_id,
                        "guitarset",
                        local_wav_path,
                        note_indices,
                        note_velocities,
                        onset_indices,
                        onset_velocities,
                        contour_indices,
                        contour_velocities,
                        (n_time_frames, N_FREQ_BINS_NOTES),
                        (n_time_frames, N_FREQ_BINS_CONTOURS),
                    )
                )
        return [batch]


def create_input_data(
        train_percent: float, validation_percent: float, seed: Optional[int] = None
) -> List[Tuple[str, str]]:
    """
    test is everything what is not allocated by train_percent and validation_percent
    :return: Tuple with Track Id + split
    """
    assert train_percent + validation_percent < 1.0, "Don't over allocate the data!"

    # Test percent is 1 - train - validation
    validation_bound = train_percent
    test_bound = validation_bound + validation_percent

    if seed:
        random.seed(seed)

    def determine_split(index: int) -> str:
        if index < len(track_ids) * validation_bound:
            return "train"
        elif index < len(track_ids) * test_bound:
            return "validation"
        else:
            return "test"

    guitarset = mirdata.initialize("guitarset")
    guitarset.download(['index'])
    track_ids = guitarset.track_ids
    random.shuffle(track_ids)

    return [(track_id, determine_split(i)) for i, track_id in enumerate(track_ids)]


def main(known_args: argparse.Namespace, pipeline_args: List[str]) -> None:
    time_created = int(time.time())
    destination = commandline.resolve_destination(known_args, time_created)
    input_data = create_input_data(known_args.train_percent, known_args.validation_percent, known_args.split_seed)

    pipeline_options = {
        "runner": known_args.runner,
        "job_name": f"guitarset-tfrecords-{time_created}",
        "machine_type": "e2-standard-4",
        "num_workers": 25,
        "disk_size_gb": 128,
        "experiments": ["use_runner_v2"],
        "save_main_session": True,
        "sdk_container_image": known_args.sdk_container_image,
        "job_endpoint": known_args.job_endpoint,
        "environment_type": "DOCKER",
        "environment_config": known_args.sdk_container_image,
    }
    pipeline.run(
        pipeline_options,
        pipeline_args,
        input_data,
        GuitarSetToTfExample(known_args.source, download=True),
        GuitarSetInvalidTracks(),
        destination,
        known_args.batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commandline.add_default(parser, os.path.basename(os.path.splitext(__file__)[0]))
    commandline.add_split(parser)
    known_args, pipeline_args = parser.parse_known_args()

    main(known_args, pipeline_args)
