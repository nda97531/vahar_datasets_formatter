import os
from typing import List, Tuple, Dict
import pandas as pd
import polars as pl
from loguru import logger
from glob import glob

if __name__ == '__main__':
    from vahar.datasets.base_classes import ParquetDatasetFormatter, NpyWindowFormatter
    from vahar.modal_sync import split_interrupted_dfs
else:
    from .base_classes import ParquetDatasetFormatter, NpyWindowFormatter
    from ..modal_sync import split_interrupted_dfs


class Pamap2Const:
    INERTIA_MODAL = 'inertia'
    HEARTRATE_MODAL = 'heartrate'
    RAW_COL_DTYPES = {'timestamp(s)': float, 'label': int, 'heart_rate(bpm)': float}
    SELECTED_HEARTRATE_COLS = ['timestamp(ms)', 'label', 'heart_rate(bpm)']
    SELECTED_IMU_COLS = ['timestamp(ms)', 'label']

    @classmethod
    def define_att(cls):
        imu_cols = [
            'temperature(C)',
            'acc_x(m/s^2)', 'acc_y(m/s^2)', 'acc_z(m/s^2)',
            '6gacc_x(m/s^2)', '6gacc_y(m/s^2)', '6gacc_z(m/s^2)',
            'gyro_x(rad/s)', 'gyro_y(rad/s)', 'gyro_z(rad/s)',
            'mag_x(uT)', 'mag_y(uT)', 'mag_z(uT)',
            'orientation_x(invalid)', 'orientation_y(invalid)', 'orientation_z(invalid)', 'orientation_w(invalid)'
        ]
        selected_imu_cols = [
            'acc_x(m/s^2)', 'acc_y(m/s^2)', 'acc_z(m/s^2)',
            'gyro_x(rad/s)', 'gyro_y(rad/s)', 'gyro_z(rad/s)',
            'mag_x(uT)', 'mag_y(uT)', 'mag_z(uT)',
            'orientation_x(invalid)', 'orientation_y(invalid)', 'orientation_z(invalid)', 'orientation_w(invalid)'
        ]
        positions = ['hand', 'chest', 'ankle']

        cls.RAW_COL_DTYPES.update({f'{position}_{imu}': float
                                   for position in positions
                                   for imu in imu_cols})
        cls.SELECTED_IMU_COLS += [f'{position}_{imu}'
                                  for position in positions
                                  for imu in selected_imu_cols]


Pamap2Const.define_att()


class Pamap2Parquet(ParquetDatasetFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # max gap within an uninterrupted session; unit ms
        self.max_interval = {
            Pamap2Const.INERTIA_MODAL: 100,
            Pamap2Const.HEARTRATE_MODAL: 900
        }
        # drop session shorter than this; unit ms
        self.min_length_segment = 10000

    @staticmethod
    def get_session_info(raw_file_path: str) -> tuple:
        """
        Get session info from data file path

        Args:
            raw_file_path: data file path

        Returns:
            a tuple:
                - subset: 'Optional' or 'Protocol'
                - subject ID: integer from 1 to 9
        """
        subset = raw_file_path.split('/')[-2]
        assert subset in {'Optional', 'Protocol'}

        subject = raw_file_path.split('/subject10')[-1].removesuffix('.dat')
        subject = int(subject)

        return subset, subject

    def read_raw_df(self, path: str) -> Dict[str, pl.DataFrame]:
        """
        Read raw data file into 2 DFs for IMU and heartrate and format the data

        Args:
            path: file path

        Returns:
            a dict: {'inertia': IMU DF, 'heartrate': heartrate DF}
        """
        # read data
        df = pl.read_csv(path, has_header=False, separator=' ', new_columns=Pamap2Const.RAW_COL_DTYPES,
                         dtypes=Pamap2Const.RAW_COL_DTYPES)
        df = df.with_columns((pl.col('timestamp(s)') * 1000).cast(int).alias('timestamp(ms)'))
        df = df.set_sorted('timestamp(ms)')
        # replace NAN with Null
        df = df.fill_nan(None)

        imu_df = df.select(Pamap2Const.SELECTED_IMU_COLS)
        heartrate_df = df.select(Pamap2Const.SELECTED_HEARTRATE_COLS)

        # drop NAN rows
        imu_df = imu_df.drop_nulls()
        heartrate_df = heartrate_df.drop_nulls()

        return {Pamap2Const.INERTIA_MODAL: imu_df, Pamap2Const.HEARTRATE_MODAL: heartrate_df}

    def split_sessions(self, dfs: Dict[str, pl.DataFrame]) -> list:
        """
        Split a DF into uninterrupted DFs

        Args:
            dfs: dict with keys are modal names, values are original dataframes

        Returns:
            list of dicts, each dict has the format:
                - key: modal name
                - value: uninterrupted dataframe
        """
        # move label to a separate DF
        data_dfs = {modal: modal_df.drop('label') for modal, modal_df in dfs.items()}
        label_dfs = {modal: modal_df.select(['timestamp(ms)', 'label']).set_sorted('timestamp(ms)')
                     for modal, modal_df in dfs.items()}
        del dfs

        # split DF where there are big gaps
        data_dfs = split_interrupted_dfs(
            dfs=data_dfs, max_interval=self.max_interval,
            min_length_segment=self.min_length_segment, sampling_rates=self.sampling_rates
        )

        # match label with resampled data
        for i, data_df_dict in enumerate(data_dfs):
            data_dfs[i] = {
                modal: data_df.join_asof(label_dfs[modal], on='timestamp(ms)', strategy='nearest')
                for modal, data_df in data_df_dict.items()
            }
        return data_dfs

    def run(self):
        session_files = sorted(glob(f'{self.raw_folder}/*/subject*.dat'))
        logger.info(f'Found {len(session_files)} sessions')

        for session_file in session_files:
            subset, subject = self.get_session_info(session_file)
            logger.info(f'Processing subject {subject}, session {subset}')

            imu_output_path = self.get_output_file_path(Pamap2Const.INERTIA_MODAL, subject, subset)
            hr_output_path = self.get_output_file_path(Pamap2Const.HEARTRATE_MODAL, subject, subset)
            if os.path.isfile(imu_output_path) and os.path.isfile(hr_output_path):
                logger.info(f'Skipping because already run before')
                continue

            dfs = self.read_raw_df(session_file)
            dfs = self.split_sessions(dfs)
            for i, modal_dfs in enumerate(dfs):
                write_name = f'{subset}_{i}' if (i != len(dfs) - 1) else subset
                for modal in [Pamap2Const.INERTIA_MODAL, Pamap2Const.HEARTRATE_MODAL]:
                    self.write_output_parquet(modal_dfs[modal], modal, subject, write_name)


class Pamap2NpyWindow(NpyWindowFormatter):
    def __init__(self, only_protocol: bool, *args, **kwargs):
        """
        Run sliding window on Pamap 2

        Args:
            only_protocol: only include Protocol sessions
            parquet_root_dir:
            window_size_sec:
            step_size_sec:
            modal_cols:
        """
        super().__init__(*args, **kwargs)
        self.only_protocol = only_protocol

    def run(self) -> pd.DataFrame:
        """
        Main processing method

        Returns:
            a DF, each row is a session, columns are:
                - 'subject': subject ID
                - '<modality 1>': array shape [num window, window length, features]
                - '<modality 2>': ...
                - 'label': array shape [num window]
        """
        parquet_sessions = self.get_parquet_file_list(session_pattern='Protocol*' if self.only_protocol else '*')

        result = []
        for parquet_session in parquet_sessions.iter_rows(named=True):
            _, subject, _ = self.get_parquet_session_info(list(parquet_session.values())[0])

            session_data = self.parquet_to_windows(parquet_session=parquet_session, subject=subject)
            result.append(session_data)

        result = pd.DataFrame(result)
        return result


if __name__ == '__main__':
    parquet_folder = '/mnt/data_drive/projects/processed_parquet/Pamap2'

    # pamap2parquet = Pamap2Parquet(
    #     raw_folder='/mnt/data_drive/projects/raw datasets/PAMAP2_Dataset/',
    #     destination_folder=parquet_folder,
    #     sampling_rates={Pamap2Const.INERTIA_MODAL: 50, Pamap2Const.HEARTRATE_MODAL: 10}
    # )
    # pamap2parquet.run()

    acc_cols = ['acc_x(m/s^2)', 'acc_y(m/s^2)', 'acc_z(m/s^2)']
    gyro_cols = ['gyro_x(rad/s)', 'gyro_y(rad/s)', 'gyro_z(rad/s)']
    positions = ['hand', 'chest', 'ankle']

    windows = Pamap2NpyWindow(
        only_protocol=True,
        parquet_root_dir=parquet_folder,
        window_size_sec=5.12,
        step_size_sec=2.56,
        modal_cols={
            Pamap2Const.INERTIA_MODAL: {
                'chest_acc': ['chest_acc_x(m/s^2)', 'chest_acc_y(m/s^2)', 'chest_acc_z(m/s^2)'],
                'hand_acc': ['hand_acc_x(m/s^2)', 'hand_acc_y(m/s^2)', 'hand_acc_z(m/s^2)'],
                'ankle_acc': ['ankle_acc_x(m/s^2)', 'ankle_acc_y(m/s^2)', 'ankle_acc_z(m/s^2)'],
                'chest_mag': ['chest_mag_x(uT)', 'chest_mag_y(uT)', 'chest_mag_z(uT)'],
                'hand_mag': ['hand_mag_x(uT)', 'hand_mag_y(uT)', 'hand_mag_z(uT)'],
                'ankle_mag': ['ankle_mag_x(uT)', 'ankle_mag_y(uT)', 'ankle_mag_z(uT)']
            },
            Pamap2Const.HEARTRATE_MODAL: {
                'hr': ['heart_rate(bpm)']
            }
        }
    ).run()
    _ = 1
