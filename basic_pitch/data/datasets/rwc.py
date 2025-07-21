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


non_piano_ids: List[str] = ['001', '002', '003', '004', '005', '007', '008', '009', '010', '011', '012', '013',
                            '014', '015', '016', '017', '024', '025', '036', '038', '041']
non_piano_train_ids: List[str] = ['001', '002', '003', '004', '005', '007', '008', '009', '010', '011', '012',
                                  '013', '014', '024', '025']
non_piano_validation_ids: List[str] = ['041', '038', '036']
non_piano_test_ids: List[str] = ['015', '016', '017']

class RwcInvalidTracks(beam.DoFn):
    """
    This Implementation does not filter out anything, it is just a conversion in the apache beam workflow.
    """

    def process(self, element: Tuple[str, str], *args: Tuple[Any, Any], **kwargs: Dict[str, Any]) -> Any:
        """
        The *args and **kwargs are required for the beam.DoFn interface, but are not used in this function.
        """
        track_id, split = element
        yield beam.pvalue.TaggedOutput(split, track_id)


class RwcToTfExample(beam.DoFn):

    # noinspection PyMissingConstructor
    # -> beam.DoFn does not use __init__
    def __init__(self, source: str) -> None:
        self.source = source
        self.rwc_wav = os.path.join(source, 'wav_22050_mono')
        self.rwc_midi_warped = os.path.join(source, 'MIDI_warped')

    @classmethod
    def available_groups(cls):
        return ['rwc', 'non-piano', 'non-piano-train', 'non-piano-validation', 'non-piano-test']

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

            local_midi_path = os.path.join(self.rwc_midi_warped, basename + '_warped.mid')
            wav_path = os.path.join(self.rwc_wav, basename + '.wav')

            notes: mirdata_annotations.NoteData = mirdata_io.load_notes_from_midi(local_midi_path)
            multif0s: mirdata_annotations.MultiF0Data = mirdata_io.load_multif0_from_midi(local_midi_path)

            with tempfile.TemporaryDirectory() as local_tmp_dir:
                tmpaudio = os.path.join(local_tmp_dir, basename + '.wav')
                tmpaudio_resampled = os.path.join(local_tmp_dir, basename + '_resampled.wav')
                shutil.copy(wav_path, tmpaudio)

                # SOX is used to convert the audio to the correct format
                tfm = sox.Transformer()
                tfm.rate(AUDIO_SAMPLE_RATE)
                tfm.set_output_format(bits=16)
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
                        "rwc",
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
    rwc_audio_dir = os.path.join(source, 'wav_22050_mono')

    audio_filepaths: List[str] = glob(os.path.join(rwc_audio_dir, f'*.wav'))
    audio_filepaths_filtered: List[str] = []
    for group in groups:
        list_to_check = None
        if group == 'non-piano-train':
            list_to_check = non_piano_train_ids
        elif group == 'non-piano-validation':
            list_to_check = non_piano_validation_ids
        else:
            raise RuntimeError(f"Not implemented for group {group}")
        for audio_filepath in audio_filepaths:
            basename = os.path.splitext(os.path.basename(audio_filepath))[0]
            ident = basename[4:7] # Extract digits from the filename
            if ident in list_to_check:
                audio_filepaths_filtered.append(audio_filepath)

    return [(os.path.basename(audio_filepath), split) for audio_filepath in audio_filepaths_filtered]


def main(known_args: argparse.Namespace, pipeline_args: List[str]) -> None:
    time_created = int(time.time())
    source: str = known_args.source
    destination = commandline.resolve_destination(known_args, time_created)

    input_data_train = create_input_data(source, ['non-piano-train'], 'train')
    # input_data_train = input_data_train[:2]
    input_data_validation = create_input_data(source, ['non-piano-validation'], 'validation')

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
                 RwcToTfExample(known_args.source),
                 RwcInvalidTracks(),
                 destination,
                 known_args.batch_size)
