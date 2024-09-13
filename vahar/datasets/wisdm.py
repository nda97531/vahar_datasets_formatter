from typing import List
import os
from loguru import logger
import pandas as pd
import polars as pl

if __name__ == '__main__':
    from vahar.datasets.base_classes import ParquetDatasetFormatter, NpyWindowFormatter
    from vahar.modal_sync import split_interrupted_dfs
    from vahar.constant import G_TO_MS2
else:
    from .base_classes import ParquetDatasetFormatter, NpyWindowFormatter
    from ..modal_sync import split_interrupted_dfs
    from ..constant import G_TO_MS2


class WisdmConst:
    # modal names
    MODAL_INERTIA = 'inertia'


class WisdmParquet(ParquetDatasetFormatter):
    def __init__(self, raw_folder: str, destination_folder: str, sampling_rates: dict,
                 min_length_segment: float = 10, max_interval: dict = None):
        """
        Class for WISDM dataset.
        In this dataset, raw modals are considered as sub-modals. For example, modal 'inertia' contains 3 sub-modals:
        [acc, gyr, mag], which are also raw modals.

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
        max_interval = {WisdmConst.MODAL_INERTIA: 500} if max_interval is None else max_interval
        super().__init__(raw_folder, destination_folder, sampling_rates)

        self.min_length_segment = min_length_segment * 1000
        self.max_interval = max_interval

    def read_csv_raw(self, filename: str = 'WISDM_ar_v1.1_raw.txt'):
        """
        Read the raw CSV file into a polars DF

        Args:
            filename: raw csv file name

        Returns:
            dataframe with columns:
        """
        path = f'{self.raw_folder}/{filename}'
        df = pl.from_pandas(pd.read_csv(path, header=None, sep=',', lineterminator=';', skiprows=[343419],
                                        usecols=[0, 1, 2, 3, 4, 5], dtype={0: str}))
        df.columns = ['subject', 'label', 'timestamp(ms)',
                      'pocket_acc_x(m/s^2)', 'pocket_acc_y(m/s^2)', 'pocket_acc_z(m/s^2)']
        df = df.with_columns(pl.col('subject').str.strip_chars())
        df = df.drop_nulls()
        df = df.filter(pl.col('timestamp(ms)') != 0)

        # convert timestamp unit and data unit
        df = df.with_columns(
            (pl.col('timestamp(ms)') * 1e-6).round(),
            pl.col('pocket_acc_x(m/s^2)', 'pocket_acc_y(m/s^2)', 'pocket_acc_z(m/s^2)') / 10 * G_TO_MS2
        )
        df = df.cast({'subject': pl.Int32, 'label': pl.String, 'timestamp(ms)': pl.Int64})
        return df

    def split_segments(self, df: pl.DataFrame, drop_col: tuple = ('subject',)) -> List[pl.DataFrame]:
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
            dfs={WisdmConst.MODAL_INERTIA: df},
            max_interval=self.max_interval,
            min_length_segment=self.min_length_segment,
            sampling_rates=self.sampling_rates
        )

        results = [
            segment_df[WisdmConst.MODAL_INERTIA].join_asof(label_df, on='timestamp(ms)', strategy='nearest')
            for segment_df in segment_dfs
        ]
        return results

    def run(self):
        raw_df = self.read_csv_raw()
        # convert labels from text to numbers
        class_list = raw_df.get_column('label').unique().sort()
        self.label_dict = dict(zip(range(len(class_list)), class_list))
        raw_df = raw_df.with_columns(pl.col('label').replace(class_list, list(self.label_dict.keys())).cast(pl.Int32))

        written_files = 0
        skipped_subjects = 0
        # for each subject
        for subject_id, subject_df in raw_df.group_by(['subject']):
            subject_id = subject_id[0]
            # check if already run before
            if os.path.isfile(self.get_output_file_path(WisdmConst.MODAL_INERTIA, subject_id, 'last')):
                logger.info(f'Skipping subject ID {subject_id} because it has been done before.')
                skipped_subjects += 1
                continue
            logger.info(f'Starting subject ID {subject_id}')

            # split subject DF into uninterrupted segments
            segment_dfs = self.split_segments(subject_df)
            # write each segment DF
            for seg_i, seg_df in enumerate(segment_dfs):
                session_id = seg_i if seg_i != len(segment_dfs) - 1 else 'last'
                self.write_output_parquet(seg_df, WisdmConst.MODAL_INERTIA, subject_id, session_id)
                written_files += 1

        logger.info(f'{written_files} file(s) written, {skipped_subjects} subject(s) skipped')
        self.export_label_list()


class WisdmNpyWindow(NpyWindowFormatter):
    pass


if __name__ == '__main__':
    parquet_dir = '/mnt/data_partition/UCD/dataset_processed/Wisdm'

    WisdmParquet(
        raw_folder='/mnt/data_partition/downloads/WISDM_ar_v1.1',
        destination_folder=parquet_dir,
        sampling_rates={WisdmConst.MODAL_INERTIA: 50}
    ).run()

    # dataset_window = WisdmNpyWindow(
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
