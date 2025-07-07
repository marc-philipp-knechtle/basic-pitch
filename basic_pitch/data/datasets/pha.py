import argparse
import logging
import os.path
import shutil
import tempfile
import time
import re
from glob import glob

from basic_pitch.data import commandline, pipeline
from typing import Any, List, Dict, Tuple, Optional

import apache_beam as beam
import tensorflow as tf

import mirdata.io as mirdata_io
from mirdata import annotations as mirdata_annotations

from tests.data import utils


class PhAInvalidTracks(beam.DoFn):
    """
    This Implementation does not filter out anything, it is just a conversion in the apache beam workflow.
    """

    def process(self, element: Tuple[str, str], *args: Tuple[Any, Any], **kwargs: Dict[str, Any]) -> Any:
        """
        The *args and **kwargs are required for the beam.DoFn interface, but are not used in this function.
        """
        track_id, split = element
        yield beam.pvalue.TaggedOutput(split, track_id)


class PhAToTfExample(beam.DoFn):
    # noinspection PyMissingConstructor
    # -> beam.DoFn does not use __init__
    def __init__(self, source: str) -> None:
        self.source = source
        self.phenicx_anechoic_mixaudio_wav = os.path.join(source, 'mixaudio_wav_22050_mono')
        self.phenicx_anechoic_annotations = os.path.join(source, 'annotations')

    @classmethod
    def available_groups(cls):
        return ['beethoven', 'bruckner', 'mahler', 'mozart']

    def process(self, element: List[str], *args: Tuple[Any, Any], **kwargs: Dict[str, Any]) -> List[Any]:
        import sox
        import numpy as np

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

        logging.info(f'Processing {element}')
        batch: List[tf.train.Example] = []

        for track_id in element:
            basename = os.path.splitext(track_id)[0]
            local_wav_path = os.path.join(self.phenicx_anechoic_mixaudio_wav, basename + '.wav')

            midi_filepaths: List[str] = glob(os.path.join(self.phenicx_anechoic_annotations, basename, '*.mid'))
            # remove the all.mid file, where all the _o files are included
            midi_filepaths = [f for f in midi_filepaths if not re.compile(fr".*all.mid").search(f)]
            # remove all original files (not warped to the actual recording)
            midi_filepaths = [f for f in midi_filepaths if not re.compile(fr".*_o.mid").search(f)]

            local_midi_path: str = utils.combine_midi_files_ticks_per_beat(midi_filepaths, os.path.join(
                self.phenicx_anechoic_annotations, basename, 'warped_all.mid'))

            notes: mirdata_annotations.NoteData = mirdata_io.load_notes_from_midi(local_midi_path)
            multif0s: mirdata_annotations.MultiF0Data = mirdata_io.load_multif0_from_midi(local_midi_path)

            with tempfile.TemporaryDirectory() as local_tmp_dir:
                tmpaudio = os.path.join(local_tmp_dir, basename + '.wav')
                tmpaudio_resampled = os.path.join(local_tmp_dir, basename + '_resampled.wav')
                shutil.copy(local_wav_path, tmpaudio)

                # SOX is used to convert the audio to the correct format
                tfm = sox.Transformer()
                tfm.rate(AUDIO_SAMPLE_RATE)
                tfm.channels(AUDIO_N_CHANNELS)
                tfm.build(tmpaudio, tmpaudio_resampled)

                duration = sox.file_info.duration(tmpaudio_resampled)
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
                note_indices, note_velocities = notes.to_sparse_index(
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
                onset_indices, onset_velocities = notes.to_sparse_index(
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
                contour_indices, contour_velocities = multif0s.to_sparse_index(
                    time_scale, "s", FREQ_BINS_CONTOURS, "hz"
                )

                batch.append(
                    tf_example_serialization.to_transcription_tfexample(
                        track_id,
                        "pha",
                        tmpaudio_resampled,
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


def create_input_data(source: str, groups: List[str], split: str) -> List[Tuple[str, str]]:
    assert os.path.exists(
        source), f"Source path {source} does not exist. Please download the dataset manually and try again."
    pha_audio_dir = os.path.join(source, 'mixaudio_wav_22050_mono')

    audio_filepaths: List[str] = []

    for group in groups:
        audio_filepaths.append(os.path.join(pha_audio_dir, group + '.wav'))

    return [(os.path.basename(audio_filepath), split) for audio_filepath in audio_filepaths]


def main(known_args: argparse.Namespace, pipeline_args: List[str]) -> None:
    time_created = int(time.time())
    source: str = known_args.source
    destination = commandline.resolve_destination(known_args, time_created)

    input_data_train = create_input_data(source, ['beethoven'], 'train')
    input_data_validation = create_input_data(source, ['mahler'], 'validation')

    pipeline_options = {
        "runner": known_args.runner,
        "job_name": f"pha-tfrecords-{time_created}",
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

    pipeline.run(pipeline_options,
                 pipeline_args,
                 input_data_train + input_data_validation,
                 PhAToTfExample(known_args.source),
                 PhAInvalidTracks(),
                 destination,
                 known_args.batch_size)
