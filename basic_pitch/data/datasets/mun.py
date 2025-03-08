import argparse
import logging
import os.path
import shutil
import tempfile
import time
import pretty_midi
from glob import glob

import pandas as pd

from basic_pitch.data import commandline, pipeline
from typing import Any, List, Dict, Tuple, Optional

import apache_beam as beam
import tensorflow as tf

import mirdata.io as mirdata_io
from mirdata import annotations as mirdata_annotations

from tests.data import utils

validation_set_files = ['1729', '1733', '1755', '1756', '1765', '1766', '1805', '1807', '1811', '1828', '1829',
                        '1932', '1933', '2081', '2082', '2083', '2157', '2158', '2167', '2186', '2194', '2221',
                        '2222', '2289', '2315', '2318', '2341', '2342', '2480', '2481', '2629', '2632', '2633']
"""
MuN validation files, copied from repo: multipitch_architectures
"""

test_set_files: Dict = {
    'MuN-3-test': ['2303', '1819', '2382'],
    'MuN-10-test': ['2303', '1819', '2382', '2298', '2191', '2556', '2416', '2628', '1759', '2106'],
    'MuN-10-var-test': ['2303', '1819', '2382', '2298', '2191', '2556', '2416', '2629', '1759', '2106'],
    'MuN-10-slow-test': ['2302', '1818', '2383', '2293', '2186', '2557', '2415', '2627', '1758', '2105'],
    'MuN-10-fast-test': ['2310', '1817', '2381', '2296', '2186', '2555', '2417', '2626', '1757', '2104'],
    'MuN-36-cyc-test': ['2302', '2303', '2304', '2305',
                        '1817', '1818', '1819',
                        '2381', '2382', '2383', '2384',
                        '2293', '2294', '2295', '2296', '2297', '2298',
                        '2186', '2191',
                        '2555', '2556', '2557',
                        '2415', '2416', '2417',
                        '2626', '2627', '2628', '2629',
                        '1757', '1758', '1759', '1760',
                        '2104', '2105', '2106']
}


class MuNInvalidTracks(beam.DoFn):
    """
    This Implementation does not filter out anything, it is just a conversion in the apache beam workflow.
    """

    def process(self, element: Tuple[str, str], *args: Tuple[Any, Any], **kwargs: Dict[str, Any]) -> Any:
        """
        The *args and **kwargs are required for the beam.DoFn interface, but are not used in this function.
        """
        track_id, split = element
        yield beam.pvalue.TaggedOutput(split, track_id)


class MuNToTfExample(beam.DoFn):

    def __init__(self, source: str) -> None:
        self.source = source
        self.mun_generated_midi_annotations = os.path.join(source, '_musicnet_generated_midi')
        self.MUN_ANNOTATION_SAMPLERATE = 44100

    def _save_mun_csv_as_midi(self, csv_file, midi_path) -> str:
        if not os.path.exists(midi_path):
            os.mkdir(midi_path)

        csv_annotations: pd.DataFrame = pd.read_csv(csv_file, sep=',')
        midi_filename = os.path.basename(csv_file.replace('.csv', '.mid'))
        midi_filepath = os.path.join(midi_path, midi_filename)
        if os.path.exists(midi_filepath):
            return str(midi_filepath)

        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)

        for idx, row in csv_annotations.iterrows():
            onset: float = row['start_time'] / self.MUN_ANNOTATION_SAMPLERATE
            offset: float = row['end_time'] / self.MUN_ANNOTATION_SAMPLERATE
            pitch: int = int(row['note'])
            note = pretty_midi.Note(start=onset, end=offset, pitch=pitch, velocity=64)
            piano.notes.append(note)
        file: pretty_midi.PrettyMIDI = pretty_midi.PrettyMIDI()
        file.instruments.append(piano)
        file.write(midi_filepath)

        return str(midi_filepath)

    @classmethod
    def available_groups(cls):
        return ['MuN-3-train', 'MuN-3-test', 'MuN-10-train', 'MuN-10-test', 'MuN-10-var-train', 'MuN-10-var-test',
                'MuN-10-slow-train', 'MuN-10-slow-test', 'MuN-10-fast-train', 'MuN-10-fast-test',
                'MuN-36-cyc-train', 'MuN-36-cyc-test', 'MuN-validation']

    def setup(self):
        logging.info(f'Preprocessing files for MuN dataset (generating midi files from csv annotations)')
        ann_audio_note_filepaths_csv: List[str] = glob(
            os.path.join(os.path.join(self.source, 'musicnet'), '**', '*.csv'), recursive=True)
        assert len(ann_audio_note_filepaths_csv) == 330
        for filepath in ann_audio_note_filepaths_csv:
            self._save_mun_csv_as_midi(filepath, self.mun_generated_midi_annotations)

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
            basename: str = os.path.splitext(track_id)[0]
            local_midi_path: str = os.path.join(self.mun_generated_midi_annotations, basename + '.mid')
            wav_paths: List[str] = glob(os.path.join(os.path.join(self.source, 'musicnet'), '**', track_id),
                                        recursive=True)
            assert len(wav_paths) == 1, f'Expected exactly one wav file for track {track_id}, found {len(wav_paths)}'
            local_wav_path: str = wav_paths[0]

            notes: mirdata_annotations.NoteData = mirdata_io.load_notes_from_midi(local_midi_path)
            multif0s: mirdata_annotations.MultiF0Data = mirdata_io.load_multif0_from_midi(local_midi_path)

            with tempfile.TemporaryDirectory() as local_tmp_dir:
                tmpaudio = os.path.join(local_tmp_dir, basename + '.wav')
                tmpaudio_resampled = os.path.join(local_tmp_dir, basename + '_resampled.wav')
                shutil.copy(local_wav_path, tmpaudio)

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
                        "csd",
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


def create_input_data(source: str, group: str, split: str) -> List[Tuple[str, str]]:
    if not os.path.exists(source):
        raise ValueError(f"Source path {source} does not exist. Please download the dataset manually and try again.")

    all_audio_filepaths: List[str] = glob(os.path.join(os.path.join(source, 'musicnet'), '**', '*.wav'), recursive=True)
    audio_filepaths_filtered: List[str] = []
    # Filter audio files based on MuN groups defined above
    if 'test' in group:
        test_labels: List[str] = test_set_files[group]
        for filepath in all_audio_filepaths:
            if any(test_label in filepath for test_label in test_labels):
                audio_filepaths_filtered.append(filepath)
    elif 'train' in group:
        group_test = group[:-5] + 'test'
        test_labels: List[str] = test_set_files[group_test] + validation_set_files
        for filepath in all_audio_filepaths:
            if not any(test_label in filepath for test_label in test_labels):
                audio_filepaths_filtered.append(filepath)
    elif 'validation' in group:
        for filepath in all_audio_filepaths:
            if any(validation_label in filepath for validation_label in validation_set_files):
                audio_filepaths_filtered.append(filepath)
    else:
        raise ValueError(f'Specified unknown group for this dataset. Specified: {group}')

    if len(audio_filepaths_filtered) < 2:
        raise RuntimeError(
            f'Received unexpected number of files for group {group}, found {len(audio_filepaths_filtered)}')

    return [(os.path.basename(audio_filepath), split) for audio_filepath in audio_filepaths_filtered]


def main(known_args: argparse.Namespace, pipeline_args: List[str]):
    time_created = int(time.time())
    source: str = known_args.source
    destination = commandline.resolve_destination(known_args, time_created)

    input_data_train = create_input_data(source, 'MuN-10-var-train', 'train')
    input_data_validation = create_input_data(source, 'MuN-validation', 'validation')

    pipeline_options = {
        "runner": known_args.runner,
        "job_name": f"csd-tfrecords-{time_created}",
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
                 MuNToTfExample(known_args.source),
                 MuNInvalidTracks(),
                 destination,
                 known_args.batch_size)
