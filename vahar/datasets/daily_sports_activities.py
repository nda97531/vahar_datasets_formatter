from typing import List
import os
import numpy as np
from loguru import logger
import polars as pl
from glob import glob

from my_py_utils.my_py_utils.pl_dataframe import resample_numeric_df

if __name__ == '__main__':
    from vahar.datasets.base_classes import ParquetDatasetFormatter, NpyWindowFormatter
else:
    from .base_classes import ParquetDatasetFormatter, NpyWindowFormatter


class DailySportConst:
    # modal names; currently only support accelerometer
    MODAL_INERTIA = 'inertia'
    MODAL_ACC = 'acc'
    MODAL_GYR = 'gyr'
    MODAL_MAG = 'mag'
    RAW_FREQ = 25

    LABEL_LIST = ['sitting', 'standing', 'lying on back', 'lying on right side', 'ascending stairs',
                  'descending stairs', 'standing in an elevator', 'moving around in an elevator',
                  'walking in a parking lot', 'walking on a flat treadmill', 'walking on a inclined treadmill',
                  'running on a treadmill', 'exercising on a stepper', 'exercising on a cross trainer',
                  'cycling in horizontal positions', 'cycling in vertical positions', 'rowing', 'jumping',
                  'playing basketball']

    @classmethod
    def define_att(cls):
        cls.ALL_COLS = [f'{pos}_{ss_type}_{axis}({unit})'
                        for pos in ['torso', 'rightArm', 'leftArm', 'rightLeg', 'leftLeg']
                        for ss_type, unit in [[cls.MODAL_ACC, 'm/s^2'], [cls.MODAL_GYR, 'rad/s'], [cls.MODAL_MAG, '?']]
                        for axis in ['x', 'y', 'z']]
        all_cols = np.array(cls.ALL_COLS).reshape([5, 9])
        cls.COLS_BY_SENSOR_TYPES = {
            cls.MODAL_ACC: all_cols[:, 0:3].reshape(-1).tolist(),
            cls.MODAL_GYR: all_cols[:, 3:6].reshape(-1).tolist(),
            cls.MODAL_MAG: all_cols[:, 6:9].reshape(-1).tolist()
        }


DailySportConst.define_att()


class DailySportParquet(ParquetDatasetFormatter):
    def __init__(self, raw_folder: str, destination_folder: str, sampling_rates: dict,
                 sensor_types=(DailySportConst.MODAL_ACC, DailySportConst.MODAL_GYR)):
        """
        Class for DailySport dataset.

        Args:
            raw_folder: path to unprocessed dataset
            destination_folder: folder to save output
            sampling_rates: a dict containing sampling rates of each modal to resample by linear interpolation.
                - key: modal name
                - value: sampling rate (unit: Hz)
            sensor_types: tuple of sensor types to process; accepted values: acc, gyr, mag
        """
        super().__init__(raw_folder, destination_folder, sampling_rates)

        self.label_dict = dict(zip(range(len(DailySportConst.LABEL_LIST)), DailySportConst.LABEL_LIST))
        self.sensor_types = sensor_types

        self.selected_cols = []
        for ss_type, ss_type_cols in DailySportConst.COLS_BY_SENSOR_TYPES.items():
            if ss_type in sensor_types:
                self.selected_cols += ss_type_cols

    def get_info_from_file_path(self, path: str) -> tuple:
        """
        Get session info from file path

        Args:
            path: path to a raw data csv file

        Returns:
            a tuple of (label, subject ID, segment ID)
        """
        info = path.removesuffix('.txt').split(os.sep)[-3:]
        label, subject, segment = info

        label = int(label[1:]) - 1
        subject = int(subject[1:])
        segment = int(segment[1:])
        return label, subject, segment

    def read_raw_file(self, path: str) -> pl.DataFrame:
        """
        Read and format a raw csv file.

        Args:
            path: path to raw text file

        Returns:
            polars Dataframe
        """
        df = pl.read_csv(path, has_header=False, new_columns=DailySportConst.ALL_COLS, separator=',')
        df = df.select(
            pl.lit(np.arange(len(df)) / DailySportConst.RAW_FREQ * 1000).cast(pl.Int64).alias('timestamp(ms)'),
            *self.selected_cols
        )
        return df

    def run(self):
        written_files = 0
        skipped_sessions = 0
        skipped_files = 0

        # for each session
        for file in glob(os.sep.join([self.raw_folder, 'a*', 'p*', 's*.txt'])):
            label, subject_id, segment_id = self.get_info_from_file_path(file)
            session_id = f'sub{subject_id}_cls{label}_seg{segment_id}'

            # check if already run before
            if os.path.isfile(self.get_output_file_path(DailySportConst.MODAL_INERTIA, subject_id, session_id)):
                logger.info(f'Skipping session {session_id} because it has been done before.')
                skipped_sessions += 1
                continue
            logger.info(f'Starting session {session_id}_sub{subject_id}')

            # read file
            df = self.read_raw_file(file)

            # re-sample data
            df = resample_numeric_df(
                df, timestamp_col='timestamp(ms)', new_frequency=self.sampling_rates[DailySportConst.MODAL_INERTIA],
                end_ts=df.item(-1, 'timestamp(ms)') + 1000 / DailySportConst.RAW_FREQ - 1
            )

            # add label column
            df = df.with_columns(pl.lit(label).alias('label'))

            # write DF file
            written = self.write_output_parquet(df, DailySportConst.MODAL_INERTIA, subject_id, session_id)
            written_files += int(written)
            skipped_files += int(not written)

        logger.info(f'{written_files} file(s) written, {skipped_sessions} session(s) skipped, '
                    f'{skipped_files} file(s) skipped')

        # convert labels from text to numbers
        self.export_label_list()


class DailySportNpyWindow(NpyWindowFormatter):
    pass


if __name__ == '__main__':
    parquet_dir = '/home/nda97531/Documents/dataset_parquet/DailySport'

    DailySportParquet(
        raw_folder='/home/nda97531/Documents/daily+and+sports+activities',
        destination_folder=parquet_dir,
        sampling_rates={DailySportConst.MODAL_INERTIA: 50},
        sensor_types=list(DailySportConst.COLS_BY_SENSOR_TYPES)
    ).run()

    # dataset_window = DailySportNpyWindow(
    #     parquet_root_dir=parquet_dir,
    #     window_size_sec=4,
    #     step_size_sec=2,
    #     modal_cols={
    #         DailySportConst.MODAL_INERTIA: {
    #             'acc': ['torso_acc_x(m/s^2)', 'torso_acc_y(m/s^2)', 'torso_acc_z(m/s^2)']
    #         }
    #     }
    # ).run()
    # _ = 1
