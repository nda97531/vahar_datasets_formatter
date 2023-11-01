import os
from typing import List
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
    MODAL = 'inertia'
    RAW_COL_DTYPES = {'timestamp(s)': float, 'label': int, 'heart_rate(bpm)': float}
    SELECTED_COLS = ['timestamp(ms)', 'label']
    RAW_INERTIA_FREQ = 100

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
        cls.SELECTED_COLS += [f'{position}_{imu}'
                              for position in positions
                              for imu in selected_imu_cols]


Pamap2Const.define_att()


class Pamap2Parquet(ParquetDatasetFormatter):
    def __init__(self, *args, **kwargs):
        assert 'heart_rate(bpm)' not in Pamap2Const.SELECTED_COLS, 'heart_rate(bpm) is not yet supported'

        super().__init__(*args, **kwargs)
        # max gap within an uninterrupted session; unit ms
        self.max_interval = {Pamap2Const.MODAL: 100}
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

    def read_raw_df(self, path: str) -> pl.DataFrame:
        """
        Read raw data file into a DF and format the data

        Args:
            path: file path

        Returns:
            polars DF
        """
        # read data
        df = pl.read_csv(path, has_header=False, separator=' ', new_columns=Pamap2Const.RAW_COL_DTYPES,
                         dtypes=Pamap2Const.RAW_COL_DTYPES)
        df = df.with_columns((pl.col('timestamp(s)') * 1000).cast(int).alias('timestamp(ms)'))
        df = df.select(Pamap2Const.SELECTED_COLS)
        df = df.set_sorted('timestamp(ms)')

        return df

    def split_sessions(self, df: pl.DataFrame) -> List[pl.DataFrame]:
        """
        Split a DF into uninterrupted DFs

        Args:
            df: original dataframe

        Returns:
            list of uninterrupted dataframes
        """
        # move label to a separate DF
        data_df = df.drop('label')
        label_df = df.select(['timestamp(ms)', 'label']).set_sorted('timestamp(ms)')
        del df

        # drop NAN rows
        data_df = data_df.fill_nan(None)
        data_df = data_df.drop_nulls()

        # split DF where there are big gaps
        data_dfs = split_interrupted_dfs(
            dfs={Pamap2Const.MODAL: data_df}, max_interval=self.max_interval,
            min_length_segment=self.min_length_segment, sampling_rates=self.sampling_rates
        )
        del data_df
        data_dfs = [df[Pamap2Const.MODAL] for df in data_dfs]

        # match label with resampled data
        dfs = [df.join_asof(label_df, on='timestamp(ms)', strategy='nearest')
               for df in data_dfs]
        return dfs

    def run(self):
        session_files = sorted(glob(f'{self.raw_folder}/*/subject*.dat'))
        logger.info(f'Found {len(session_files)} sessions')

        for session_file in session_files:
            subset, subject = self.get_session_info(session_file)
            logger.info(f'Processing subject {subject}, session {subset}')

            output_path = self.get_output_file_path(Pamap2Const.MODAL, subject, subset)
            if os.path.isfile(output_path):
                logger.info(f'Skipping because already run before')
                continue

            uninterrupted_dfs = self.split_sessions(self.read_raw_df(session_file))
            for i, uninterrupted_df in enumerate(uninterrupted_dfs):
                write_path = f'{subset}_{i}' if (i != len(uninterrupted_dfs) - 1) else subset
                self.write_output_parquet(uninterrupted_df, Pamap2Const.MODAL, subject, write_path)


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
        list_sessions = self.get_parquet_file_list(session_pattern='Protocol*' if self.only_protocol else '*')
        list_sessions = list_sessions[Pamap2Const.MODAL].to_list()

        result = []
        for session_file in list_sessions:
            modal, subject, session = self.get_parquet_session_info(session_file)

            session_data = self.parquet_to_windows(parquet_session={modal: session_file}, subject=subject)
            result.append(session_data)

        result = pd.DataFrame(result)
        return result


if __name__ == '__main__':
    parquet_folder = '/mnt/data_drive/projects/processed_parquet/Pamap2'

    # pamap2parquet = Pamap2Parquet(
    #     raw_folder='/mnt/data_drive/projects/raw datasets/PAMAP2_Dataset/',
    #     destination_folder=parquet_folder,
    #     sampling_rates={Pamap2Const.MODAL: 50}
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
        exclude_labels=[0],
        no_transition=False,
        modal_cols={
            Pamap2Const.MODAL: {
                'acc': [f'{pos}_{axis_col}' for pos in positions for axis_col in acc_cols],
                'gyro': [f'{pos}_{axis_col}' for pos in positions for axis_col in gyro_cols]
            }
        }
    ).run()
