"""
MIDI Processing Module

Handles loading, converting, and saving MIDI files.
"""

import numpy as np
import mido
from typing import Optional, Tuple
from pathlib import Path


class MIDIProcessor:
    """
    Processes MIDI files for music generation.
    
    Handles:
    - Loading MIDI files
    - Converting MIDI to piano roll representation
    - Converting piano roll back to MIDI
    - Extracting musical features from MIDI
    """
    
    def __init__(self, ticks_per_beat: int = 480, 
                 min_note: int = 21, max_note: int = 108):
        """
        Initialize the MIDI processor.
        
        Args:
            ticks_per_beat: MIDI ticks per quarter note
            min_note: Minimum MIDI note number (A0 = 21)
            max_note: Maximum MIDI note number (C8 = 108)
        """
        self.ticks_per_beat = ticks_per_beat
        self.min_note = min_note
        self.max_note = max_note
        self.note_range = max_note - min_note + 1
        
    def load_midi(self, filepath: str) -> mido.MidiFile:
        """
        Load a MIDI file.
        
        Args:
            filepath: Path to MIDI file
            
        Returns:
            Mido MidiFile object
        """
        midi_file = mido.MidiFile(filepath)
        return midi_file
        
    def midi_to_piano_roll(self, filepath: str, 
                          resolution: int = 24) -> np.ndarray:
        """
        Convert MIDI file to piano roll representation.
        
        Args:
            filepath: Path to MIDI file
            resolution: Time resolution (ticks per step)
            
        Returns:
            Piano roll matrix (time x pitch), binary values
        """
        midi_file = self.load_midi(filepath)
        
        # Calculate total length in ticks
        total_ticks = 0
        for track in midi_file.tracks:
            current_tick = 0
            for msg in track:
                current_tick += msg.time
            total_ticks = max(total_ticks, current_tick)
            
        # Create piano roll
        n_steps = int(total_ticks / resolution) + 1
        piano_roll = np.zeros((n_steps, self.note_range), dtype=np.float32)
        
        # Track active notes
        active_notes = {}
        
        for track in midi_file.tracks:
            current_tick = 0
            
            for msg in track:
                current_tick += msg.time
                current_step = int(current_tick / resolution)
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Note starts
                    if self.min_note <= msg.note <= self.max_note:
                        note_idx = msg.note - self.min_note
                        active_notes[msg.note] = current_step
                        
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    # Note ends
                    if msg.note in active_notes:
                        if self.min_note <= msg.note <= self.max_note:
                            note_idx = msg.note - self.min_note
                            start_step = active_notes[msg.note]
                            end_step = min(current_step, n_steps)
                            piano_roll[start_step:end_step, note_idx] = 1.0
                        del active_notes[msg.note]
                        
        return piano_roll
        
    def piano_roll_to_midi(self, piano_roll: np.ndarray, 
                          output_path: str,
                          resolution: int = 24,
                          tempo: int = 500000) -> None:
        """
        Convert piano roll to MIDI file.
        
        Args:
            piano_roll: Piano roll matrix (time x pitch)
            output_path: Where to save MIDI file
            resolution: Time resolution (ticks per step)
            tempo: Microseconds per quarter note (500000 = 120 BPM)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create MIDI file
        midi_file = mido.MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = mido.MidiTrack()
        midi_file.tracks.append(track)
        
        # Add tempo
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        
        # Convert piano roll to note events
        # Threshold piano roll for binary decision
        piano_roll_binary = (piano_roll > 0.5).astype(int)
        
        # Track note states
        note_states = np.zeros(self.note_range, dtype=bool)
        
        for step in range(piano_roll_binary.shape[0]):
            current_notes = piano_roll_binary[step]
            
            # Find notes that need to be turned off
            notes_to_off = np.where(note_states & ~current_notes)[0]
            for note_idx in notes_to_off:
                note = note_idx + self.min_note
                track.append(mido.Message('note_off', note=note, 
                                        velocity=64, time=0))
                note_states[note_idx] = False
                
            # Find notes that need to be turned on
            notes_to_on = np.where(~note_states & current_notes)[0]
            for note_idx in notes_to_on:
                note = note_idx + self.min_note
                track.append(mido.Message('note_on', note=note, 
                                        velocity=80, time=0))
                note_states[note_idx] = True
                
            # Add time delta for next step
            if step < piano_roll_binary.shape[0] - 1:
                track.append(mido.Message('note_on', note=60, 
                                        velocity=0, time=resolution))
                
        # Turn off any remaining notes
        for note_idx in np.where(note_states)[0]:
            note = note_idx + self.min_note
            track.append(mido.Message('note_off', note=note, 
                                    velocity=64, time=0))
            
        # Save MIDI file
        midi_file.save(str(output_path))
        print(f"MIDI saved to {output_path}")
        
    def load_and_convert(self, filepath: str, 
                        resolution: int = 24) -> np.ndarray:
        """
        Load MIDI file and convert to piano roll.
        
        Args:
            filepath: Path to MIDI file
            resolution: Time resolution
            
        Returns:
            Piano roll representation
        """
        piano_roll = self.midi_to_piano_roll(filepath, resolution=resolution)
        return piano_roll
        
    def get_note_density(self, piano_roll: np.ndarray) -> float:
        """
        Calculate note density (percentage of active notes).
        
        Args:
            piano_roll: Piano roll matrix
            
        Returns:
            Note density (0 to 1)
        """
        return np.mean(piano_roll > 0)
        
    def get_pitch_range(self, piano_roll: np.ndarray) -> Tuple[int, int]:
        """
        Get the pitch range used in piano roll.
        
        Args:
            piano_roll: Piano roll matrix
            
        Returns:
            Tuple of (min_pitch, max_pitch) as MIDI note numbers
        """
        active_pitches = np.where(np.any(piano_roll > 0, axis=0))[0]
        
        if len(active_pitches) == 0:
            return (0, 0)
            
        min_pitch = active_pitches[0] + self.min_note
        max_pitch = active_pitches[-1] + self.min_note
        
        return (min_pitch, max_pitch)
        
    def visualize_piano_roll(self, piano_roll: np.ndarray,
                            max_time_steps: int = 500) -> None:
        """
        Visualize a piano roll.
        
        Args:
            piano_roll: Piano roll matrix
            max_time_steps: Maximum time steps to display
        """
        import matplotlib.pyplot as plt
        
        # Limit display to max_time_steps
        display_roll = piano_roll[:max_time_steps, :]
        
        plt.figure(figsize=(15, 6))
        plt.imshow(display_roll.T, aspect='auto', origin='lower', 
                  cmap='binary', interpolation='nearest')
        plt.xlabel('Time Step')
        plt.ylabel('Pitch (MIDI Note)')
        plt.title('Piano Roll Visualization')
        plt.colorbar(label='Active')
        plt.tight_layout()
        plt.show()
