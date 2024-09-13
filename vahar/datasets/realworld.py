import os
import re
import zipfile
from collections import defaultdict
from typing import Dict
from loguru import logger
import numpy as np
import pandas as pd
import polars as pl
from glob import glob

from my_py_utils.my_py_utils.string_utils import rreplace

if __name__ == '__main__':
    from vahar.datasets.base_classes import ParquetDatasetFormatter, NpyWindowFormatter
    from vahar.modal_sync import split_interrupted_dfs
else:
    from .base_classes import ParquetDatasetFormatter, NpyWindowFormatter
    from ..modal_sync import split_interrupted_dfs


class RealWorldConst:
    # modal names
    MODAL_INERTIA = 'inertia'

    # unsupported: 'mag', 'lig', 'mic', 'gps'
    SUBMODAL_SI_UNIT = {
        'acc': 'm/s^2',
        'gyr': 'rad/s'
    }
    RAW_MODALS: list

    CLASS_LABELS = ['walking', 'running', 'sitting', 'standing', 'lying', 'climbingup', 'climbingdown', 'jumping']
    SENSOR_POSITION = {'head', 'chest', 'waist', 'upperarm', 'forearm', 'thigh', 'shin'}

    @classmethod
    def define_att(cls):
        # submodal (or raw modal) name
        cls.RAW_MODALS = list(cls.SUBMODAL_SI_UNIT)


RealWorldConst.define_att()


class RealWorldParquet(ParquetDatasetFormatter):
    def __init__(self, raw_folder: str, destination_folder: str, sampling_rates: dict,
                 used_modals: dict = None, sensor_pos: list = ('waist',),
                 min_length_segment: float = 10, max_interval: dict = None):
        """
        Class for RealWorld2016 dataset.
        In this dataset, raw modals are considered as sub-modals. For example, modal 'inertia' contains 3 sub-modals:
        [acc, gyr, mag], which are also raw modals.

        Args:
            raw_folder: path to unprocessed dataset
            destination_folder: folder to save output
            sampling_rates: a dict containing sampling rates of each modal to resample by linear interpolation.
                - key: modal name
                - value: sampling rate (unit: Hz)
            used_modals: a dict containing sub-modal names of each modal
                - key: modal name (any name), this will be used in output paths
                - value: list of sub-modal names, choices are in RealWorldConst.RAW_MODALS.
                Default: {'inertia': ['acc', 'gyr']}
            sensor_pos: list of sensor positions to take
                (don't use all by default to avoid more interruptions in session)
            min_length_segment: only write segments longer than this threshold (unit: sec)
            max_interval: dict[submodal] = maximum intervals (millisecond) between rows of an uninterrupted segment;
                default = 500 ms
        """
        used_modals = {RealWorldConst.MODAL_INERTIA: ['acc', 'gyr']} if used_modals is None else used_modals
        max_interval = {'acc': 500, 'gyr': 500} if max_interval is None else max_interval

        all_submodals = np.concatenate(list(used_modals.values()))
        assert len(set(all_submodals) - set(RealWorldConst.RAW_MODALS)) == 0, \
            (f'Invalid raw modal: {set(all_submodals) - set(RealWorldConst.RAW_MODALS)}; '
             f'Expected values in: {RealWorldConst.RAW_MODALS}')
        assert len(all_submodals) == len(set(all_submodals)), 'Duplicate sub_modals, please check'
        assert len(set(sensor_pos) - RealWorldConst.SENSOR_POSITION) == 0, \
            (f'Invalid sensor positions found: {set(sensor_pos) - RealWorldConst.SENSOR_POSITION}; '
             f'Expected values in: {RealWorldConst.SENSOR_POSITION}')

        super().__init__(raw_folder, destination_folder, sampling_rates)

        self.sensor_pos = sensor_pos
        self.min_length_segment = min_length_segment * 1000

        self.submodal_2_modal = {submodal: modal
                                 for modal in used_modals
                                 for submodal in used_modals[modal]}
        self.max_interval = max_interval
        self.label_dict = dict(zip(range(len(RealWorldConst.CLASS_LABELS)), RealWorldConst.CLASS_LABELS))

    def get_list_sessions(self) -> pl.DataFrame:
        """
        Scan all session files

        Returns:
            a DF, each row is a session, columns are sub-modal names, each cell contains a file path
        """
        list_submodals = list(self.submodal_2_modal)
        first_submodal = list_submodals[0]
        files = {
            first_submodal: sorted(glob(f'{self.raw_folder}/proband*/data/{first_submodal}_*_csv.zip'))
        }
        for sub_modal in list_submodals[1:]:
            files[sub_modal] = [rreplace(f, first_submodal, sub_modal) for f in files[first_submodal]]
        df = pl.DataFrame(files)
        return df

    def get_info_from_session_file(self, zip_path: str) -> tuple:
        """
        Get info from ZIP data file path

        Args:
            zip_path: path to zip data file

        Returns:
            a tuple of 3 elements: subject ID (int), label (int), zip file split number (str)
        """
        search_text = zip_path.removeprefix(self.raw_folder)
        info = re.search(
            rf'proband([0-9]*)/data/[a-z]{{3}}_({"|".join(RealWorldConst.CLASS_LABELS)})_([1-3]?)_?csv.zip',
            search_text
        )
        subject_id, text_label, zip_file_split = tuple(info.group(i) for i in range(1, 4))

        subject_id = int(subject_id)
        idx_label = RealWorldConst.CLASS_LABELS.index(text_label)
        return subject_id, idx_label, zip_file_split

    def read_csv_in_zip(self, zip_path: str) -> dict:
        """
        Read all csv files in a zip file

        Args:
            zip_path: path to zip file

        Returns:
            dict with:
                - key: sensor position (chest, waist, ...)
                - value: corresponding DF
        """
        result = {}

        # read all csv files in zip file
        with zipfile.ZipFile(zip_path, 'r') as zf:
            compressed_list = [item for item in zf.namelist() if item.endswith('.csv')]

            # for each csv file in the zip file
            for csv_file in compressed_list:
                sensor_pos = csv_file.split('_')[-1].removesuffix('.csv')
                assert sensor_pos in RealWorldConst.SENSOR_POSITION, (f'Unexpected sensor position: {sensor_pos}; '
                                                                      f'Expected: {RealWorldConst.SENSOR_POSITION}')
                if sensor_pos in self.sensor_pos:
                    result[sensor_pos] = pl.read_csv(zf.read(csv_file))

        # clean up DFs
        sub_modal = zip_path.split('/')[-1].split('_')[0]
        if sub_modal in RealWorldConst.SUBMODAL_SI_UNIT:
            # both acc and gyr are already in SI units, no need to convert
            unit = RealWorldConst.SUBMODAL_SI_UNIT[sub_modal]
            for sensor_pos, df in result.items():
                result[sensor_pos] = df.select(
                    pl.col('attr_time').alias('timestamp(ms)'),
                    pl.col('attr_x').alias(f'{sensor_pos}_{sub_modal}_x({unit})'),
                    pl.col('attr_y').alias(f'{sensor_pos}_{sub_modal}_y({unit})'),
                    pl.col('attr_z').alias(f'{sensor_pos}_{sub_modal}_z({unit})')
                )
        else:
            raise ValueError(f'Unsupported sub-modal: {sub_modal}')
        return result

    def split_session_to_segments(self, all_dfs_of_session: dict) -> list:
        """
        Split a session into uninterrupted segments

        Args:
            all_dfs_of_session: a 2-level dict: dict[submodal][sensor position] = raw DF

        Returns:
            list of 1-level dicts, each dict has this format: dict[{submodal}_{sensor position}] = DF
        """
        # convert 2-level dict to 1-level dict
        all_dfs_of_session = {f'{submodal}_{sensor_pos}': df
                              for submodal in all_dfs_of_session.keys()
                              for sensor_pos, df in all_dfs_of_session[submodal].items()}

        # split into uninterrupted segments
        max_interval = {key: self.max_interval[key.split('_')[0]] for key in all_dfs_of_session.keys()}
        sampling_rates = {key: self.sampling_rates[self.submodal_2_modal[key.split('_')[0]]]
                          for key in all_dfs_of_session.keys()}
        segments = split_interrupted_dfs(dfs=all_dfs_of_session, max_interval=max_interval,
                                         min_length_segment=self.min_length_segment, sampling_rates=sampling_rates)
        return segments

    def concat_submodals_to_modal(self, data_dict: Dict[str, pl.DataFrame]) -> dict:
        """
        Concatenate DFs of submodals into DFs of modal;
        Example: concat [acc, gyr] => inertia

        Args:
            data_dict: a dict with format: dict[{submodal}_{sensor position}] = DF

        Returns:
            a dict with format: dict[modal] = DF
        """
        dfs = defaultdict(list)
        # for each submodal
        for submodal, df in data_dict.items():
            modal = self.submodal_2_modal[submodal.split('_')[0]]

            # only keep ts column for 1 DF of each submodal to avoid duplicate column name during concatenation
            if len(dfs[modal]):
                df = df.drop('timestamp(ms)')
            dfs[modal].append(df)

        # concat DFs with the same sensor type, add label column
        dfs = {sensor: pl.concat(list_dfs, how='horizontal') for sensor, list_dfs in dfs.items()}
        return dfs

    def add_label_col(self, data_dict: Dict[str, pl.DataFrame], label: any) -> dict:
        """
        Add a `label` column with only 1 value to each DF in the input

        Args:
            data_dict: data_dict: a dict with format: dict[modal] = DF
            label: label value

        Returns:
            same as input dict, but each DF inside has a new `label` column
        """
        data_dict = {key: val.with_columns(pl.Series(name='label', values=[label] * len(val)))
                     for key, val in data_dict.items()}
        return data_dict

    def run(self):
        unique_modals = set(self.submodal_2_modal.values())

        # scan all sessions
        logger.info('Scanning for sessions...')
        list_sessions = self.get_list_sessions()
        logger.info(f'Found {len(list_sessions)} sessions in total')

        skipped_sessions = 0
        written_files = 0
        # for each session
        for session_row in list_sessions.iter_rows(named=True):
            # get session info
            subject_id, idx_label, zip_file_split = self.get_info_from_session_file(next(iter(session_row.values())))
            session_info = f's{subject_id}_l{idx_label}_z{zip_file_split}'

            # check if already run before
            if all(os.path.isfile(self.get_output_file_path(modal, subject_id, f'{session_info}_segmentlast'))
                   for modal in unique_modals):
                logger.info(f'Skipping session {session_info} because already run before')
                skipped_sessions += 1
                continue
            logger.info(f'Starting session {session_info}')

            # a 2-level dict: dict[submodal][sensor position] = raw DF
            all_dfs_of_session = {}
            session_is_empty = True
            # read all submodal files of this session
            for submodal, submodal_file in session_row.items():
                all_dfs_of_session[submodal] = self.read_csv_in_zip(submodal_file)
                session_is_empty &= (len(all_dfs_of_session[submodal]) == 0)

            if session_is_empty:
                logger.info(f"Skipping session {session_info} because it's empty")
                skipped_sessions += 1
                continue

            # split a session (which may be interrupted), into uninterrupted segments
            segments = self.split_session_to_segments(all_dfs_of_session)

            # for each segment
            for seg_i, seg in enumerate(segments):
                # concat submodal DFs into modal DF
                seg = self.concat_submodals_to_modal(seg)
                # add label column to DFs (each session has only 1 label)
                seg = self.add_label_col(seg, idx_label)

                # number the segments, mark the last one
                if seg_i == len(segments) - 1:
                    seg_i = 'last'
                write_name = f'{session_info}_segment{seg_i}'

                # write to files
                for modal, df in seg.items():
                    if self.write_output_parquet(df, modal, subject_id, write_name):
                        written_files += 1

        logger.info(f'{written_files} file(s) written, {skipped_sessions} session(s) skipped')

        # convert labels from text to numbers
        self.export_label_list()


class RealWorldNpyWindow(NpyWindowFormatter):
    pass


if __name__ == '__main__':
    parquet_dir = f'/mnt/data_partition/UCD/dataset_processed/RealWorld_test'

    RealWorldParquet(
        raw_folder='/mnt/data_partition/downloads/realworld2016_dataset',
        destination_folder=parquet_dir,
        sampling_rates={RealWorldConst.MODAL_INERTIA: 50},
        used_modals={RealWorldConst.MODAL_INERTIA: ['acc']},
        sensor_pos=list(RealWorldConst.SENSOR_POSITION),
        min_length_segment=4,
        max_interval={'acc': 500}
    ).run()

    # dataset_window = RealWorldNpyWindow(
    #     parquet_root_dir=parquet_dir,
    #     window_size_sec=2,
    #     step_size_sec=1,
    #     modal_cols={
    #         RealWorldConst.MODAL_INERTIA: {
    #             'thigh': ['thigh_acc_x(m/s^2)', 'thigh_acc_y(m/s^2)', 'thigh_acc_z(m/s^2)']
    #         }
    #     }
    # ).run()
    _ = 1
