import onnxruntime as ort
import time
import librosa
import kaldi_native_fbank as knf
import numpy as np
import mutagen
from mutagen.wave import WAVE
import os
import psutil
import tempfile
import json 
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import pickle
import dill
from omegaconf import DictConfig, OmegaConf, open_dict
import nemo.collections.asr as nemo_asr


def audio_duration(length):
    hours = length // 3600  # calculate in hours
    length %= 3600
    mins = length // 60  # calculate in minutes
    length %= 60
    seconds = length  # calculate in seconds
    total_duration = hours * 3600 + mins * 60 + seconds
    return float(total_duration)
def get_files_in_folder(folder_path):
    files_array = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file_path is a file (not a folder)
        if os.path.isfile(file_path):
            files_array.append(file_path)
    
    return files_array
def _setup_dataloader_from_config(self,config: dict):
        # Automatically inject args from model config to dataloader config
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='labels')
        dataset = audio_to_text_dataset.get_audio_to_text_char_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            preprocessor_cfg=self._cfg.get("preprocessor", None),
        )

        if dataset is None:
            return None

        if isinstance(dataset, AudioToCharDALIDataset):
            # DALI Dataset implements dataloader interface
            return dataset

        shuffle = config['shuffle']
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )
def _setup_transcribe_dataloader(self, config: dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))
        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'labels': self.joint.vocabulary,
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
        }    
        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer
def main():
    device = 'cpu'
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_en_conformer_transducer_large")
    total_length=0.0
    folder_path = "rasptest"
    files=get_files_in_folder(folder_path)
    max_symbols_per_step=5
    audios=sorted(files)
    start_memory = psutil.virtual_memory().used
    print(f"Memory usage before loading the model: {start_memory} bytes")
    for i in range(len(audios)):
        audio_filepath = audios[i]
        audio = WAVE(audio_filepath)
        # contains all the metadata about the wavpack file
        audio_info = audio.info
        length = int(audio_info.length)
        total_length += audio_duration(length)
        
        with tempfile.TemporaryDirectory() as tmpgdir:
            with open(os.path.join(tmpgdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
                entry = {'audio_filepath': audio_filepath, 'duration': 100000, 'text': 'nothing'}
                fp.write(json.dumps(entry) + '\n')

            config = {'paths2audio_files': [audio_filepath], 'batch_size': 1, 'temp_dir': tmpgdir}
            
    output = _setup_transcribe_dataloader(DictConfig(config))  
    # Save the output to a file
if __name__ == '__main__':
    main()           