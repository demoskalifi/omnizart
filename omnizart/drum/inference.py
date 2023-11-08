import pretty_midi
import numpy as np
from scipy.signal import find_peaks

# Define the standard MIDI note numbers for the given drum components
MIDI_PITCHES = {
    'Kick': 36,
    'Snare': 38,
    'Closed_Hat': 42,
    'Open_Hat': 46,
    'Low_Tom': 45,
    'Rim': 37,
}

# Define the indices of the instruments in the model's output
MODEL_OUTPUT_INDICES = {
    'Kick': 0,
    'Snare': 1,
    'Closed_Hat': 4,
    'Open_Hat': 6,
    'Low_Tom': 7,
    'Rim': 2,
}

def filter_activations(activations_dict, pred, m_beat_arr):
    LOW_SOUNDS = ['Kick', 'Low_Tom']
    HIGH_SOUNDS = ['Snare', 'Closed_Hat', 'Open_Hat', 'Rim']

    # This dictionary will hold the filtered activations
    filtered_activations = {instrument: [] for instrument in activations_dict}

    # We will iterate over each beat time
    for beat_index, beat_time in enumerate(m_beat_arr):
        # Gather activations for this specific beat
        beat_activations = [(instrument, time_index) for instrument, times in activations_dict.items() for time_index in times if m_beat_arr[time_index] == beat_time]
        
        # Separate low and high sound activations
        low_activations = [act for act in beat_activations if act[0] in LOW_SOUNDS]
        high_activations = [act for act in beat_activations if act[0] in HIGH_SOUNDS]
        
        # Sort them by the prediction value, highest first
        low_activations.sort(key=lambda x: pred[x[1], MODEL_OUTPUT_INDICES[x[0]]], reverse=True)
        high_activations.sort(key=lambda x: pred[x[1], MODEL_OUTPUT_INDICES[x[0]]], reverse=True)
        
        # Select up to 1 low and 2 high activations
        selected_low = low_activations[:1]
        selected_high = high_activations[:2]
        
        # Add the selected activations to the filtered list
        for instrument, time_index in selected_low + selected_high:
            if time_index not in filtered_activations[instrument]:
                filtered_activations[instrument].append(time_index)

    # Sort the filtered activations for each instrument
    for instrument in filtered_activations:
        filtered_activations[instrument].sort()

    return filtered_activations


def inference(pred, m_beat_arr, bass_drum_th, snare_th, hihat_th, default_th=0.95):
    midi_file = pretty_midi.PrettyMIDI()
    drum_inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drum Set")

    hihat_th -= 0.05 
    thresholds = {
        'Kick': bass_drum_th,
        'Snare': snare_th,
        'Closed_Hat': hihat_th,
        'Open_Hat': default_th,
        'Low_Tom': default_th,
        'Rim': default_th,
        'Clap': default_th,
    }

    activations_dict = {}
    for instrument, index in MODEL_OUTPUT_INDICES.items():
        norm_instrument = (pred[:, index] - np.mean(pred[:, index])) / np.std(pred[:, index])
        activations, _ = find_peaks(norm_instrument, height=thresholds[instrument])
        activations_dict[instrument] = activations

    filtered_activations = filter_activations(activations_dict, pred, m_beat_arr)

    for instrument, times in filtered_activations.items():
        for time in times:
            beat_time = m_beat_arr[time]
            note = pretty_midi.Note(
                velocity=100,
                pitch=MIDI_PITCHES[instrument],
                start=beat_time,
                end=beat_time + 0.1
            )
            drum_inst.notes.append(note)

    midi_file.instruments.append(drum_inst)
    return midi_file
