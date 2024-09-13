import os
import re
import zipfile
from collections import defaultdict
from typing import Dict, List
from loguru import logger
import numpy as np
import polars as pl
from glob import glob
import scipy.io as scio

from my_py_utils.my_py_utils.string_utils import rreplace
from vahar.constant import G_TO_MS2, DEG_TO_RAD

if __name__ == '__main__':
    from vahar.datasets.base_classes import ParquetDatasetFormatter, NpyWindowFormatter
    from vahar.modal_sync import split_interrupted_dfs
else:
    from .base_classes import ParquetDatasetFormatter, NpyWindowFormatter
    from ..modal_sync import split_interrupted_dfs


class CZUConst:
    # modal names
    MODAL_INERTIA = 'inertia'

    SENSOR_POSITIONS = ['chest', 'abdomen',
                        'leftElbow', 'leftWrist', 'rightElbow', 'rightWrist',
                        'leftKnee', 'leftAnkle', 'rightKnee', 'rightAnkle']

    ACTIVITIES = {0: 'Right high wave', 1: 'Left high wave', 2: 'Right horizontal wave', 3: 'Left horizontal wave',
                  4: 'Hammer with right hand', 5: 'Grasp with right hand', 6: 'Draw fork with right hand',
                  7: 'Draw fork with left hand', 8: 'Draw circle with right hand', 9: 'Draw circle with left hand',
                  10: 'Right foot kick forward', 11: 'Left foot kick forward', 12: 'Right foot kick side',
                  13: 'Left foot kick side', 14: 'Clap', 15: 'Bend down', 16: 'Wave up and down', 17: 'Sur Place',
                  18: 'Left body turning movement', 19: 'Right body turning movement', 20: 'Left lateral movement',
                  21: 'Right lateral movement'}

    SUBJECT_MAPPING = {'cx': 1, 'cyy': 2, 'myj': 3, 'qyh': 4, 'zyh': 5}


class CZUParquet(ParquetDatasetFormatter):
    def __init__(self, raw_folder: str, destination_folder: str, sampling_rates: dict,
                 used_inertial_modals: tuple = ('acc', 'gyr'), inertial_sensor_pos: list = ('waist',),
                 min_length_segment: float = 10, max_interval: dict = None):
        """
        Class for CZU-MHAD dataset. Only support inertial sensors.

        Args:
            raw_folder: path to unprocessed dataset
            destination_folder: folder to save output
            sampling_rates: a dict containing sampling rates of each modal to resample by linear interpolation.
                - key: modal name
                - value: sampling rate (unit: Hz)
            used_inertial_modals: a tuple of modal names to include.
            inertial_sensor_pos: list of sensor positions to take
                (don't use all by default to avoid more interruptions in session)
            min_length_segment: only write segments longer than this threshold (unit: sec)
            max_interval: dict[modal] = maximum intervals (millisecond) between rows of an uninterrupted segment
        """
        super().__init__(raw_folder, destination_folder, sampling_rates)

        self.used_inertial_modals = used_inertial_modals
        self.used_inertial_sensor_positions = set(inertial_sensor_pos)
        self.min_length_segment = min_length_segment * 1000
        self.max_interval = {CZUConst.MODAL_INERTIA: 500} if max_interval is None else max_interval
        self.label_dict = CZUConst.ACTIVITIES

    def get_info_from_session_file(self, path: str) -> tuple:
        """
        Get session info from file path.

        Args:
            path: path to mat file

        Returns:
            a list: [subject name (str), activity id (int), trial number (int)]
        """
        info = re.search(rf'{os.sep}([a-z]*)_a([0-9]*)_t([0-9]*).mat$', path)
        subject, activity, trial_num = info.groups()

        subject = CZUConst.SUBJECT_MAPPING[subject]
        activity = int(activity) - 1
        trial_num = int(trial_num)

        return subject, activity, trial_num

    def read_inertial_mat_file(self, path: str) -> Dict[str, pl.DataFrame]:
        """
        Read all inertial sensor data of a session.

        Args:
            path: path to mat file of the session

        Returns:
            a dict, key is inertial sensor position, value is dataframe
        """
        data = scio.loadmat(path)['sensor']
        assert len(data) == len(CZUConst.SENSOR_POSITIONS), 'The number of sensor positions does not match'
        result = {}

        # for each sensor position
        for pos_i, sensor_pos in enumerate(CZUConst.SENSOR_POSITIONS):
            if sensor_pos not in self.used_inertial_sensor_positions:
                continue
            sensor_arr = data[pos_i, 0]
            sensor_df = {}

            # convert accelerometer unit
            if 'acc' in self.used_inertial_modals:
                sensor_arr[:, :3] *= G_TO_MS2
                for axis_i, axis in enumerate(['x', 'y', 'z']):
                    sensor_df[f'{sensor_pos}_acc_{axis}(m/s^2)'] = sensor_arr[:, axis_i]

            # convert gyroscope unit
            if 'gyr' in self.used_inertial_modals:
                sensor_arr[:, 3:6] *= DEG_TO_RAD
                for axis_i, axis in enumerate(['x', 'y', 'z']):
                    sensor_df[f'{sensor_pos}_gyr_{axis}(rad/s)'] = sensor_arr[:, axis_i + 3]

            # convert timestamp unit
            sensor_arr[:, -1] *= 1e-3
            # handle timestamp gaps
            gap_idxs = np.nonzero(np.diff(sensor_arr[:, -1]) > 30 * 1e3)[0]
            median_gap = np.median(np.diff(sensor_arr[:, -1]))
            for gap_idx in gap_idxs:
                gap = sensor_arr[gap_idx + 1, -1] - sensor_arr[gap_idx, -1]
                sensor_arr[gap_idx + 1:, -1] -= (gap - median_gap)
            sensor_df['timestamp(ms)'] = sensor_arr[:, -1].astype(int)

            result[sensor_pos] = pl.DataFrame(sensor_df)
        return result

    def process_inertial_dfs(self, inertial_dfs: Dict[str, pl.DataFrame]) -> List[pl.DataFrame]:
        """
        Split an inertial data session into uninterrupted segments.
        Merge all sensor positions into one dataframe.

        Args:
            inertial_dfs: a dict[sensor positions] = df

        Returns:
            a list of uninterrupted session dataframes, each dataframe contains all sensor positions.
        """
        # split a session (which may be interrupted), into uninterrupted segments; a list of dict[sensor pos] = df
        inertial_dfs = split_interrupted_dfs(
            dfs=inertial_dfs,
            max_interval={key: self.max_interval[CZUConst.MODAL_INERTIA] for key in inertial_dfs.keys()},
            min_length_segment=self.min_length_segment,
            sampling_rates={key: self.sampling_rates[CZUConst.MODAL_INERTIA] for key in inertial_dfs.keys()}
        )

        result = []
        for segment in inertial_dfs:
            segment = list(segment.values())
            segment_df = pl.concat(
                items=[segment[0]] + [segment[i].drop('timestamp(ms)') for i in range(1, len(segment))],
                how='horizontal'
            )
            result.append(segment_df)
        return result

    def run(self):
        # to override: process data of any dataset and
        # 1. call self.get_output_file_path for each session to check if it has already been processed
        # 2. call self.write_output_parquet() for every modal of every session
        # 3. call self.export_class_list to export class list to a JSON file

        # scan all sessions
        logger.info('Scanning for sessions...')
        list_sensor_sessions = sorted(glob(f'{self.raw_folder}/sensor_mat/*.mat'))
        logger.info(f'Found {len(list_sensor_sessions)} sessions in total')

        skipped_sessions = 0
        written_files = 0
        # for each session
        for session_file in list_sensor_sessions:
            # get session info
            subject, activity, trial_num = self.get_info_from_session_file(session_file)
            session_info = f's{subject}_a{activity}_t{trial_num}'

            # check if already run before
            if os.path.isfile(self.get_output_file_path(CZUConst.MODAL_INERTIA, subject, f'{session_info}_seglast')):
                logger.info(f'Skipping session {session_info} because already run before')
                skipped_sessions += 1
                continue
            logger.info(f'Starting session {session_info}')

            # read inertial file of this session; a dict[sensor pos] = df
            inertial_dfs = self.read_inertial_mat_file(session_file)
            # process inertial data; a list of uninterrupted dfs
            inertial_dfs = self.process_inertial_dfs(inertial_dfs)

            # for each segment
            for i, df in enumerate(inertial_dfs):
                # number the segments, mark the last one
                if i == len(inertial_dfs) - 1:
                    i = 'last'
                write_name = f'{session_info}_seg{i}'

                # add label column
                df = df.with_columns(pl.lit(activity, pl.Int32).alias('label'))

                # write to files
                if self.write_output_parquet(df, CZUConst.MODAL_INERTIA, subject, write_name):
                    written_files += 1

        logger.info(f'{written_files} file(s) written, {skipped_sessions} session(s) skipped')

        # convert labels from text to numbers
        self.export_label_list()


class CZUNpyWindow(NpyWindowFormatter):
    pass


if __name__ == '__main__':
    parquet_dir = f'/mnt/data_partition/UCD/dataset_processed/CZU-MHAD'

    # CZUParquet(
    #     raw_folder='/mnt/data_partition/downloads/CZU-MHAD',
    #     destination_folder=parquet_dir,
    #     sampling_rates={CZUConst.MODAL_INERTIA: 50},
    #     used_inertial_modals=('acc', 'gyr'),
    #     inertial_sensor_pos=CZUConst.SENSOR_POSITIONS,
    #     min_length_segment=1,
    #     max_interval={CZUConst.MODAL_INERTIA: 100}
    # ).run()

    dataset_window = CZUNpyWindow(
        parquet_root_dir=parquet_dir,
        window_size_sec=2,
        step_size_sec=1,
        modal_cols={
            CZUConst.MODAL_INERTIA: {
                'chest_acc': ['chest_acc_x(m/s^2)', 'chest_acc_y(m/s^2)', 'chest_acc_z(m/s^2)']
            }
        }
    ).run()
    _ = 1
