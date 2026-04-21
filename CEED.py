"""CEED: California Earthquake Dataset for Machine Learning and Cloud Computing"""
from typing import Dict, List, Optional, Tuple, Union
import datasets
import fsspec
import h5py
import numpy as np
import torch
_CITATION = '@InProceedings{huggingface:dataset,\ntitle = {CEED: California Earthquake Dataset for Machine Learning and Cloud Computing},\nauthor={Zhu et al.},\nyear={2025}\n}\n'
_DESCRIPTION = 'A dataset of earthquake waveforms organized by earthquake events and based on the HDF5 format.\n'
_HOMEPAGE = ''
_LICENSE = ''
_REPO_NC = 'https://huggingface.co/datasets/AI4EPS/quakeflow_nc/resolve/main/waveform_h5'
_FILES_NC = ''
_REPO_SC = 'https://huggingface.co/datasets/AI4EPS/quakeflow_sc/resolve/main/waveform_h5'
_FILES_SC = []
_URLS_2002 = {'full': [f'{_REPO_NC}/{x}' for x in _FILES_NC]}

class CEED(datasets.GeneratorBasedBuilder):
    """CEED: A dataset of earthquake waveforms organized by earthquake events and based on the HDF5 format."""
    VERSION = datasets.Version('1.1.0')
    nt = 8192
    BUILDER_CONFIGS = [datasets.BuilderConfig(name='station', version=VERSION, description='yield station-based samples one by one of whole dataset'), datasets.BuilderConfig(name='event', version=VERSION, description='yield event-based samples one by one of whole dataset'), datasets.BuilderConfig(name='station_train', version=VERSION, description='yield station-based samples one by one of training dataset'), datasets.BuilderConfig(name='event_train', version=VERSION, description='yield event-based samples one by one of training dataset'), datasets.BuilderConfig(name='station_test', version=VERSION, description='yield station-based samples one by one of test dataset'), datasets.BuilderConfig(name='event_test', version=VERSION, description='yield event-based samples one by one of test dataset')]
    DEFAULT_CONFIG_NAME = 'station_test'

    def _info(self):
        if self.config.name == 'station' or self.config.name == 'station_train' or self.config.name == 'station_test':
            features = datasets.Features({'data': datasets.Array2D(shape=(3, self.nt), dtype='float32'), 'phase_time': datasets.Sequence(datasets.Value('string')), 'phase_index': datasets.Sequence(datasets.Value('int32')), 'phase_type': datasets.Sequence(datasets.Value('string')), 'phase_polarity': datasets.Sequence(datasets.Value('string')), 'begin_time': datasets.Value('string'), 'end_time': datasets.Value('string'), 'event_time': datasets.Value('string'), 'event_time_index': datasets.Value('int32'), 'event_location': datasets.Sequence(datasets.Value('float32')), 'station_location': datasets.Sequence(datasets.Value('float32'))})
        elif self.config.name == 'event' or self.config.name == 'event_train' or self.config.name == 'event_test':
            features = datasets.Features({'data': datasets.Array3D(shape=(None, 3, self.nt), dtype='float32'), 'phase_time': datasets.Sequence(datasets.Sequence(datasets.Value('string'))), 'phase_index': datasets.Sequence(datasets.Sequence(datasets.Value('int32'))), 'phase_type': datasets.Sequence(datasets.Sequence(datasets.Value('string'))), 'phase_polarity': datasets.Sequence(datasets.Sequence(datasets.Value('string'))), 'begin_time': datasets.Value('string'), 'end_time': datasets.Value('string'), 'event_time': datasets.Value('string'), 'event_time_index': datasets.Value('int32'), 'event_location': datasets.Sequence(datasets.Value('float32')), 'station_location': datasets.Sequence(datasets.Sequence(datasets.Value('float32')))})
        else:
            raise ValueError(f'config.name = {self.config.name} is not in BUILDER_CONFIGS')
        return datasets.DatasetInfo(description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION)

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_config.extract_dir if hasattr(dl_manager.download_config, 'extract_dir') else None
        user_data_dir = getattr(self.config, 'data_dir', None)
        local_dir = user_data_dir or data_dir
        if local_dir is not None:
            import os
            from glob import glob
            pattern = os.path.join(local_dir, '*.h5')
            files = sorted(glob(pattern))
            if not files:
                urls = _URLS_2002['full']
                files = dl_manager.download_and_extract(urls)
        else:
            urls = _URLS_2002['full']
            files = dl_manager.download_and_extract(urls)
        print(files)
        all_events = self._get_all_events(files)
        train_events, test_events = self._split_events(all_events, train_ratio=0.8)
        if self.config.name in ['station', 'event']:
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'filepath': files, 'split': 'train', 'selected_events': train_events}), datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={'filepath': files, 'split': 'test', 'selected_events': test_events})]
        elif self.config.name in ['station_train', 'event_train']:
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'filepath': files, 'split': 'train', 'selected_events': train_events})]
        elif self.config.name in ['station_test', 'event_test']:
            return [datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={'filepath': files, 'split': 'test', 'selected_events': test_events})]
        else:
            raise ValueError('config.name is not in BUILDER_CONFIGS')

    def _get_all_events(self, files):
        all_events = []
        for file in files:
            with fsspec.open(file, 'rb') as fs:
                with h5py.File(fs, 'r') as fp:
                    event_ids = list(fp.keys())
                    all_events.extend([(file, event_id) for event_id in event_ids])
        return all_events

    def _split_events(self, all_events, train_ratio=0.8):
        import random
        random.seed(42)
        shuffled_events = all_events.copy()
        random.shuffle(shuffled_events)
        split_idx = int(len(shuffled_events) * train_ratio)
        train_events = shuffled_events[:split_idx]
        test_events = shuffled_events[split_idx:]
        print(f'Total number of events: {len(all_events)}')
        print(f'Number of events in the training set: {len(train_events)}')
        print(f'Number of test set events: {len(test_events)}')
        return (train_events, test_events)

    def _generate_examples(self, filepath, split, selected_events=None):
        file_events_map = {}
        for file, event_id in selected_events:
            if file not in file_events_map:
                file_events_map[file] = []
            file_events_map[file].append(event_id)
        for file in filepath:
            if file not in file_events_map:
                continue
            with fsspec.open(file, 'rb') as fs:
                with h5py.File(fs, 'r') as fp:
                    event_ids = file_events_map[file]
                    for event_id in event_ids:
                        if event_id not in fp:
                            continue
                        event = fp[event_id]
                        event_attrs = event.attrs
                        begin_time = event_attrs['begin_time']
                        end_time = event_attrs['end_time']
                        event_location = [event_attrs['longitude'], event_attrs['latitude'], event_attrs['depth_km']]
                        event_time = event_attrs['event_time']
                        event_time_index = event_attrs['event_time_index']
                        station_ids = list(event.keys())
                        if len(station_ids) == 0:
                            continue
                        if 'station' in self.config.name:
                            waveforms = np.zeros([3, self.nt], dtype='float32')
                            for i, sta_id in enumerate(station_ids):
                                waveforms[:, :self.nt] = event[sta_id][:, :self.nt]
                                attrs = event[sta_id].attrs
                                phase_type = attrs['phase_type']
                                phase_time = attrs['phase_time']
                                phase_index = attrs['phase_index']
                                phase_polarity = attrs['phase_polarity']
                                station_location = [attrs['longitude'], attrs['latitude'], -attrs['elevation_m'] / 1000.0]
                                yield (f'{event_id}/{sta_id}', {'data': waveforms, 'phase_time': phase_time, 'phase_index': phase_index, 'phase_type': phase_type, 'phase_polarity': phase_polarity, 'begin_time': begin_time, 'end_time': end_time, 'event_time': event_time, 'event_time_index': event_time_index, 'event_location': event_location, 'station_location': station_location})
                        elif 'event' in self.config.name:
                            waveforms = np.zeros([len(station_ids), 3, self.nt], dtype='float32')
                            phase_type = []
                            phase_time = []
                            phase_index = []
                            phase_polarity = []
                            station_location = []
                            for i, sta_id in enumerate(station_ids):
                                waveforms[i, :, :self.nt] = event[sta_id][:, :self.nt]
                                attrs = event[sta_id].attrs
                                phase_type.append(list(attrs['phase_type']))
                                phase_time.append(list(attrs['phase_time']))
                                phase_index.append(list(attrs['phase_index']))
                                phase_polarity.append(list(attrs['phase_polarity']))
                                station_location.append([attrs['longitude'], attrs['latitude'], -attrs['elevation_m'] / 1000.0])
                            yield (event_id, {'data': waveforms, 'phase_time': phase_time, 'phase_index': phase_index, 'phase_type': phase_type, 'phase_polarity': phase_polarity, 'begin_time': begin_time, 'end_time': end_time, 'event_time': event_time, 'event_time_index': event_time_index, 'event_location': event_location, 'station_location': station_location})