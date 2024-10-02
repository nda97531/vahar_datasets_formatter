from typing import List
import os
from loguru import logger
import polars as pl
from glob import glob

if __name__ == '__main__':
    from vahar.datasets.base_classes import ParquetDatasetFormatter, NpyWindowFormatter
    from vahar.modal_sync import split_interrupted_dfs
else:
    from .base_classes import ParquetDatasetFormatter, NpyWindowFormatter
    from ..modal_sync import split_interrupted_dfs


class SonarConst:
    MODAL_INERTIA = 'inertia'

    TS_COL = 'SampleTimeFine'
    LABEL_COL = 'activity'
    SENSOR_POS = {
        'LW': 'leftWrist',
        'RW': 'rightWrist',
        'ST': 'pelvis',
        'LF': 'leftAnkle',
        'RF': 'rightAnkle'
    }

    DELTA_FREQUENCY = 60

    LABEL_LIST = ['null - activity', 'blow-dry', 'change clothes', 'clean up', 'collect dishes', 'comb hair',
                  'deliver food', 'dental care', 'documentation', 'kitchen preparation', 'make bed', 'pour drinks',
                  'prepare bath', 'push wheelchair', 'put accessories', 'put food on plate', 'put medication',
                  'serve food', 'wash at sink', 'wash hair', 'wash in bed', 'wheelchair transfer', 'wipe up']

    @classmethod
    def define_att(cls):
        # columns to read from raw file
        cls.RAW_ACC_COLS = [f'dv[{axis}]_{pos}' for pos in cls.SENSOR_POS.keys() for axis in range(1, 4)]
        cls.RAW_DELTA_QUAT_COLS = [f'dq_{axis}_{pos}' for pos in cls.SENSOR_POS.keys() for axis in ['W', 'X', 'Y', 'Z']]

        # columns to rename after reading
        cls.OUTPUT_COLS = {cls.TS_COL: 'timestamp(ms)', cls.LABEL_COL: 'label'}
        cls.OUTPUT_COLS.update(dict(zip(
            cls.RAW_ACC_COLS,
            [f'{pos}_acc_{axis}(m/s^2)' for pos in cls.SENSOR_POS.values() for axis in ['x', 'y', 'z']]
        )))
        cls.OUTPUT_COLS.update(dict(zip(
            [f'dq_{axis}_{pos}' for pos in cls.SENSOR_POS.keys() for axis in ['X', 'Y', 'Z']],
            [f'{pos}_gyr_{axis}(rad/s)' for pos in cls.SENSOR_POS.values() for axis in ['x', 'y', 'z']]
        )))


SonarConst.define_att()


class SonarParquet(ParquetDatasetFormatter):
    def __init__(self, raw_folder: str, destination_folder: str, sampling_rates: dict,
                 min_length_segment: float = 10, max_interval: dict = None):
        """
        Class for SONAR dataset.

        Args:
            raw_folder: path to unprocessed dataset
            destination_folder: folder to save output
            sampling_rates: a dict containing sampling rates of each modal to resample by linear interpolation.
                - key: modal name
                - value: sampling rate (unit: Hz)
            min_length_segment: only write segments longer than this threshold (unit: sec)
            max_interval: dict[submodal] = maximum intervals (millisecond) between rows of an uninterrupted segment;
                default = 500 ms
        """
        max_interval = {SonarConst.MODAL_INERTIA: 500} if max_interval is None else max_interval
        super().__init__(raw_folder, destination_folder, sampling_rates)

        self.min_length_segment = min_length_segment * 1000
        self.max_interval = max_interval
        self.label_dict = dict(zip(range(len(SonarConst.LABEL_LIST)), SonarConst.LABEL_LIST))

    def get_info_from_file_path(self, path: str) -> tuple:
        """
        Get session info from file path

        Args:
            path: path to a raw data csv file

        Returns:
            a tuple of (session ID, subject ID)
        """
        info = os.path.split(path)[-1]
        info = info.removesuffix('.csv')
        session_id, subject_id = info.split('_sub')
        return int(session_id), int(subject_id)

    def get_gyroscope_exprs(self, df: pl.DataFrame) -> dict:
        """
        Define expressions to convert delta quaternion to gyroscope in a polars DataFrame. Gyroscope axis x, y, z are
        updated into x, y, z columns of delta quaternion.

        Args:
            df: polars Dataframe

        Returns:
            dict with format: key[sensor position]
        """
        expressions = {}

        # for each sensor unit
        for pos in SonarConst.SENSOR_POS.keys():
            # 4 delta quaternion columns of this sensor unit
            quat_cols = [f'dq_{axis}_{pos}' for axis in ['W', 'X', 'Y', 'Z']]

            # check if data is valid
            assert df.select((pl.col(quat_cols[0]) >= 0).all()).item(), \
                'Quaternion W axis must be non-negative by convention.'
            assert df.select(pl.any_horizontal(quat_cols[1:]).all()).item(), \
                'Length of a rotation axis vector is zero, please fix the code to handle this case.'

            # define conversion expressions
            angle = pl.col(quat_cols[0]).arccos() * 2
            axis_length = (pl.sum_horizontal(pl.col(quat_cols[1:]) ** 2)).sqrt()
            scale_factor = angle / axis_length * SonarConst.DELTA_FREQUENCY
            scale_factor = pl.when(scale_factor.is_finite()).then(scale_factor).otherwise(0)

            # add expressions to result dict
            expressions.update({col: pl.col(col) * scale_factor for col in quat_cols[1:]})
        return expressions

    def read_raw_csv(self, path: str) -> pl.DataFrame:
        """
        Read and format a raw csv file.

        Args:
            path: path to csv file

        Returns:
            polars Dataframe
        """
        read_cols = [SonarConst.TS_COL, SonarConst.LABEL_COL, *SonarConst.RAW_ACC_COLS, *SonarConst.RAW_DELTA_QUAT_COLS]
        df = pl.read_csv(path, separator=',', columns=read_cols)

        df = df.with_columns(
            (pl.col(SonarConst.TS_COL) / 1e3).round().cast(pl.Int64),
            pl.col(SonarConst.RAW_ACC_COLS) * SonarConst.DELTA_FREQUENCY,
            pl.col(SonarConst.LABEL_COL).replace(SonarConst.LABEL_LIST, range(len(SonarConst.LABEL_LIST))).cast(
                pl.Int32),
            **self.get_gyroscope_exprs(df)
        )
        df = df.drop_nulls()

        df = df.rename(SonarConst.OUTPUT_COLS)
        df = df.select(SonarConst.OUTPUT_COLS.values())
        return df

    def split_segments(self, df: pl.DataFrame, drop_col: tuple = ()) -> List[pl.DataFrame]:
        """
        Split a dataframe into multiple uninterrupted dataframes.

        Args:
            df: original dataframe
            drop_col: tuple of columns to exclude in the output

        Returns:
            a list of uninterrupted dataframes
        """
        df = df.sort('timestamp(ms)')
        label_df = df.select('timestamp(ms)', 'label')
        df = df.drop('label', *drop_col)

        segment_dfs = split_interrupted_dfs(
            dfs={SonarConst.MODAL_INERTIA: df},
            max_interval=self.max_interval,
            min_length_segment=self.min_length_segment,
            sampling_rates=self.sampling_rates
        )

        results = [
            segment_df[SonarConst.MODAL_INERTIA].join_asof(label_df, on='timestamp(ms)', strategy='nearest')
            for segment_df in segment_dfs
        ]
        return results

    def run(self):
        written_files = 0
        skipped_sessions = 0
        skipped_files = 0

        # for each session
        for file in glob(f'{self.raw_folder}/*.csv'):
            session_id, subject_id = self.get_info_from_file_path(file)

            # check if already run before
            if os.path.isfile(self.get_output_file_path(SonarConst.MODAL_INERTIA, subject_id, f'{session_id}_last')):
                logger.info(f'Skipping session {session_id} because it has been done before.')
                skipped_sessions += 1
                continue
            logger.info(f'Starting session {session_id}_sub{subject_id}')

            df = self.read_raw_csv(file)
            segment_dfs = self.split_segments(df)

            # write each segment DF
            for seg_i, seg_df in enumerate(segment_dfs):
                segment_id = seg_i if seg_i != len(segment_dfs) - 1 else 'last'
                written = self.write_output_parquet(seg_df, SonarConst.MODAL_INERTIA, subject_id,
                                                    f'{session_id}_{segment_id}')
                written_files += int(written)
                skipped_files += int(not written)

        logger.info(f'{written_files} file(s) written, {skipped_sessions} session(s) skipped, '
                    f'{skipped_files} file(s) skipped')

        # convert labels from text to numbers
        self.export_label_list()


class SonarNpyWindow(NpyWindowFormatter):
    pass


if __name__ == '__main__':
    parquet_dir = '/home/nda97531/Documents/dataset_parquet/Sonar'

    SonarParquet(
        raw_folder='/home/nda97531/Documents/SONAR_ML',
        destination_folder=parquet_dir,
        sampling_rates={SonarConst.MODAL_INERTIA: 50},
        min_length_segment=5
    ).run()

    # dataset_window = SonarNpyWindow(
    #     parquet_root_dir=parquet_dir,
    #     window_size_sec=4,
    #     step_size_sec=2,
    #     # modal_cols={
    #     #     RealWorldConst.MODAL_INERTIA: {
    #     #         'waist': ['waist_acc_x(m/s^2)', 'waist_acc_y(m/s^2)', 'waist_acc_z(m/s^2)']
    #     #     }
    #     # }
    # ).run()
    # _ = 1
