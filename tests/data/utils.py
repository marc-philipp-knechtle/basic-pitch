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

import logging
import os
from typing import List

import mido
import numpy as np
import pathlib

import pandas as pd
import soundfile as sf
import wave
import pretty_midi

from mido import MidiFile, MidiTrack, Message, merge_tracks


def save_nt_csv_realtime_as_midi(csv_filenames: List[str], path: str) -> str:
    """
    Saving csv/tsv representation as midi
    This method does not consider the velocity
    Args:
        csv_filenames: csv files to be converted to midi: format: [onset_time, offset_time, pitch]
        path: path to save the midis to
    Returns:
        path where the midis are saved to
    """
    if not os.path.exists(path):
        os.mkdir(path)
    csv_filename: str
    for csv_filename in csv_filenames:
        ann_audio_note: pd.DataFrame = pd.read_csv(csv_filename, sep=';')
        ann_audio_filepath = os.path.join(path, os.path.basename(csv_filename.replace('.csv', '.mid')))
        if os.path.exists(ann_audio_filepath):
            continue

        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)

        for idx, row in ann_audio_note.iterrows():
            onset: float = row[0]
            offset: float = row[1]
            pitch: int = int(row[2])
            note = pretty_midi.Note(start=onset, end=offset, pitch=pitch, velocity=64)
            piano.notes.append(note)
        file: pretty_midi.PrettyMIDI = pretty_midi.PrettyMIDI()
        file.instruments.append(piano)
        file.write(ann_audio_filepath)
    return path


def combine_midi_files(midi_filepaths: List[str], combined_midi_savepath: str) -> str:
    if not os.path.exists(os.path.dirname(combined_midi_savepath)):
        os.mkdir(os.path.dirname(combined_midi_savepath))

    combined_midi = MidiFile(type=0)
    all_tracks = []
    """
    pretty_midi (used internally by mirdata) is restricted to type 0 or type 1 MIDI files. 
    """
    for midi_path in midi_filepaths:
        midi_file = MidiFile(midi_path)
        for track in midi_file.tracks:
            all_tracks.append(track)
    merged_track = merge_tracks(
        all_tracks)  # we need to merge the tracks to get a type 0 midi file (with a single track)
    combined_midi.tracks.append(merged_track)
    combined_midi.save(combined_midi_savepath)
    return combined_midi_savepath


def combine_midi_files_ticks_per_beat(midi_filepaths: List[str], combined_midi_savepath: str,
                                      default_ticks_per_beat: int = None) -> str:
    """
    Combined multiple midi files to a type 0 single midi file
    Args:
        midi_filepaths: list of midi's to combine to a single path
        combined_midi_savepath:
        default_ticks_per_beat: If set, the internal ticks_per_beat of the midi files are overridden and ignored
                                otherwise, we use the value from the first midi and throw an error if some other midi
                                files has a different ticks_per_beat
    Returns: path of the unified midi file
    """
    if not os.path.exists(os.path.dirname(combined_midi_savepath)):
        os.mkdir(os.path.dirname(combined_midi_savepath))

    index_0_midi = mido.MidiFile(midi_filepaths[0])
    check_ticks_per_beat: bool = False
    if default_ticks_per_beat is None: default_ticks_per_beat, check_ticks_per_beat = index_0_midi.ticks_per_beat, True

    all_tracks = []
    for midi_path in midi_filepaths:
        if check_ticks_per_beat:
            midi_file = mido.MidiFile(midi_path)
            assert midi_file.ticks_per_beat == default_ticks_per_beat
        else:
            midi_file = mido.MidiFile(midi_path)
            # midi_file.ticks_per_beat = default_ticks_per_beat
        for track in midi_file.tracks:
            if midi_file.ticks_per_beat == default_ticks_per_beat:
                all_tracks.append(track)
            else:
                # resample track with different ticks_per_beat
                for message in track:
                    if message.is_meta:
                        continue
                    else:
                        message.time = int(message.time * (default_ticks_per_beat / midi_file.ticks_per_beat))
                all_tracks.append(track)

    # merge to type 0 midi because we handle just note tracking for this here.
    merged_track = mido.merge_tracks(all_tracks)
    combined_midi = mido.MidiFile(type=0, ticks_per_beat=default_ticks_per_beat)
    combined_midi.tracks.append(merged_track)

    combined_midi.save(combined_midi_savepath)
    return combined_midi_savepath


def create_mock_wav(output_fpath: pathlib.Path, duration_min: int) -> None:
    assert output_fpath.suffix == ".wav"

    duration_seconds = duration_min * 60
    sample_rate = 44100
    n_channels = 2  # Stereo
    sampwidth = 2  # 2 bytes per sample (16-bit audio)

    # Generate a silent audio data array
    num_samples = duration_seconds * sample_rate
    audio_data = np.zeros((num_samples, n_channels), dtype=np.int16)

    # Create the WAV file
    with wave.open(str(output_fpath), "w") as wav_file:
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    logging.info(f"Mock {duration_min}-minute WAV file '{output_fpath}' created successfully.")


def create_mock_flac(output_fpath: pathlib.Path) -> None:
    assert output_fpath.suffix == ".flac"

    frequency = 440  # A4
    duration = 2  # seconds
    sample_rate = 44100  # standard
    amplitude = 0.5

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sin_wave = amplitude * np.sin(duration * np.pi * frequency * t)

    # Save as a FLAC file
    sf.write(str(output_fpath), sin_wave, frequency, format="FLAC")

    logging.info(f"Mock FLAC file {str(output_fpath)} created successfully.")


def create_mock_midi(output_fpath: pathlib.Path) -> None:
    assert output_fpath.suffix in (".mid", ".midi")
    # Create a new MIDI file with one track
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Define a sequence of notes (time, type, note, velocity)
    notes = [
        (0, "note_on", 60, 64),  # C4
        (500, "note_off", 60, 64),
        (0, "note_on", 62, 64),  # D4
        (500, "note_off", 62, 64),
    ]

    # Add the notes to the track
    for time, type, note, velocity in notes:
        track.append(Message(type, note=note, velocity=velocity, time=time))

    # Save the MIDI file
    mid.save(output_fpath)

    logging.info(f"Mock MIDI file '{output_fpath}' created successfully.")
