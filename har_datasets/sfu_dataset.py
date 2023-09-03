import os
import re
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from glob import glob

from har_datasets.base_classes import ParquetDatasetFormatter, NpyWindowFormatter
from my_py_utils.my_py_utils.pl_dataframe import resample_numeric_df as pl_resample_numeric_df


class SFUConst:
    CLASSES = ['Falls', 'Near_Falls', 'ADLs']
    FALL_CLASS = 'Falls'
    MODAL = 'inertia'
    SENSOR_POSITIONS = ['head', 'sternum', 'waist', 'lthigh', 'rthigh', 'lankle', 'rankle']

    # rename DF column names
    COLUMNS_CONVERT = {
        ' Acceleration X (m/s^2)': '_acc_x(m/s^2)',
        ' Acceleration Y (m/s^2)': '_acc_y(m/s^2)',
        ' Acceleration Z (m/s^2)': '_acc_z(m/s^2)',
        ' Angular Velocity X (rad/s)': '_gyro_x(rad/s)',
        ' Angular Velocity Y (rad/s)': '_gyro_y(rad/s)',
        ' Angular Velocity Z (rad/s)': '_gyro_z(rad/s)',
        ' Magnetic Field X (uT)': '_mag_x(uT)',
        ' Magnetic Field Y (uT)': '_mag_y(uT)',
        ' Magnetic Field Z (uT)': '_mag_z(uT)'
    }


class SFUParquet(ParquetDatasetFormatter):
    def __init__(self, raw_folder: str, destination_folder: str, sampling_rates: dict,
                 extend_fall_msec: tuple = (1000, 1000)):
        """
        Process SFU-IMU dataset to parquet

        Args:
            extend_fall_msec: extend a fall by some msec to 2 sides of the highest acceleration peak
        """
        super().__init__(raw_folder, destination_folder, sampling_rates)
        self.extend_fall_rows = [int(msec * self.sampling_rates[SFUConst.MODAL]) for msec in extend_fall_msec]

    @staticmethod
    def get_info_from_session_folder(path: str) -> tuple:
        """
        Get info from raw session folder path using regex

        Args:
            path: session folder path

        Returns:
            a tuple of integers: [subject, activity, trial]
        """
        info = re.search(rf'sub([0-9]*)/({"|".join(SFUConst.CLASSES)})/(.*).xlsx', path)
        info = (int(info.group(1)), info.group(2), info.group(3))
        return info

    def read_xlsx_session(self, excel_file: str) -> pl.DataFrame:
        """
        Read raw excel file into a DF and resample

        Returns:
            polars dataframe
        """
        df = pl.read_excel(excel_file)

        # convert timestamp unit
        df = df.with_columns((pl.col('Time') / 1000).cast(int))
        df = df.rename({'Time': 'timestamp(ms)'})

        # rename data columns (no need to convert unit)
        old_cols = df.columns
        new_cols = []
        for col in old_cols:
            for old_suffix, new_suffix in SFUConst.COLUMNS_CONVERT.items():
                if col.endswith(old_suffix):
                    new_cols.append(col.replace(old_suffix, new_suffix).replace('.', ''))
                    break
            else:
                new_cols.append(col.replace('.', ''))
        df.columns = new_cols

        df = pl_resample_numeric_df(df, timestamp_col='timestamp(ms)',
                                    new_frequency=self.sampling_rates[SFUConst.MODAL])
        return df

    def add_label_column(self, df: pl.DataFrame, session_label: str) -> pl.DataFrame:
        """
        Add a 'label' column to the DF of a session.
        0 for non-fall
        1 for fall

        Args:
            df: re-sampled DF, so its timestamp should be evenly distributed
            session_label: main class of this session

        Returns:
            a DF with 'label' column added
        """
        assert 'label' not in df.columns, 'label column already exists'

        fall_label = np.zeros(len(df), dtype=int)

        if session_label == SFUConst.FALL_CLASS:
            # find body acceleration peak => fall point
            body_accel = df.select(pl.col([c for c in df.columns
                                           if '_acc_' in c
                                           and c.split('_')[0] in {'waist', 'sternum', 'head'}]))
            assert body_accel.shape[1] == 9, f'Wrong column names: {df.columns}'
            fall_idx = np.abs(body_accel.to_numpy()).mean(1).argmax().item()

            # extend to 2 sides
            fall_label[fall_idx - self.extend_fall_rows[0]:fall_idx + self.extend_fall_rows[1]] = 1

        # add to DF
        df = df.with_columns(pl.Series(name='label', values=fall_label))
        return df

    def run(self):
        logger.info('Scanning for excel files...')
        excel_files = sorted(glob(f'{self.raw_folder}/sub*/*/*.xlsx'))
        logger.info(f'Found {len(excel_files)} excel files in total')

        skip_file = 0
        write_file = 0
        # for each session
        for excel_file in excel_files:
            # get session info
            subject, activity, trial = self.get_info_from_session_folder(excel_file)
            session_info = f'{activity}_{trial}'

            if os.path.isfile(self.get_output_file_path(SFUConst.MODAL, subject, session_info)):
                logger.info(f'Skipping session {session_info} because already run before')
                skip_file += 1
                continue
            logger.info(f'Starting session {session_info}')

            # get data
            df = self.read_xlsx_session(excel_file)

            # add label
            df = self.add_label_column(df, activity)

            # write files
            if self.write_output_parquet(df, SFUConst.MODAL, subject, session_info):
                write_file += 1
            else:
                skip_file += 1

        logger.info(f'{write_file} file(s) written, {skip_file} file(s) skipped')


class SFUNpyWindow(NpyWindowFormatter):
    def run(self) -> pd.DataFrame:
        # get list of parquet files
        parquet_sessions = self.get_parquet_file_list()

        result = []
        # for each session
        for parquet_session in parquet_sessions.iter_rows(named=True):
            # get session info
            _, subject, session_id = self.get_parquet_session_info(list(parquet_session.values())[0])
            session_label = session_id.split('_')[0]
            session_label = int(session_label == SFUConst.FALL_CLASS)

            session_result = self.process_parquet_to_windows(
                parquet_session=parquet_session,
                subject=subject,
                session_label=session_label,
                is_short_activity=bool(session_label)
            )
            result.append(session_result)
        result = pd.DataFrame(result)
        return result


if __name__ == '__main__':
    SFUParquet(
        raw_folder='/mnt/data_partition/UCD/datasets/SFU-IMU',
        destination_folder='/mnt/data_partition/UCD/UCD04 - Virtual sensor fusion/processed_parquet/SFU-IMU',
        sampling_rates={SFUConst.MODAL: 50}
    ).run()

    # col_name_pattern = '{pos}_acc_{axis}(m/s^2)'
    # SFUNpyWindow(
    #     parquet_root_dir='/mnt/data_partition/UCD/UCD04 - Virtual sensor fusion/processed_parquet/SFU-IMU',
    #     window_size_sec=4, step_size_sec=2, min_step_size_sec=0.5, max_short_window=5,
    #     modal_cols={
    #         'inertia': {
    #             pos: [col_name_pattern.format(pos=pos, axis=a) for a in ['x', 'y', 'z']]
    #             for pos in SFUConst.SENSOR_POSITIONS
    #         }
    #     }
    # ).run()
