from typing import List
import os

from tqdm import tqdm
import pandas as pd
from loguru import logger
import polars as pl
from glob import glob

if __name__ == '__main__':
    from vahar.datasets.base_classes import ParquetDatasetFormatter, NpyWindowFormatter
    from vahar.modal_sync import split_interrupted_dfs
else:
    from .base_classes import ParquetDatasetFormatter, NpyWindowFormatter
    from ..modal_sync import split_interrupted_dfs


class RealDispConst:
    MODAL_INERTIA = 'inertia'
    MD_ACC = 'acc'
    MD_GYR = 'gyr'
    MD_MAG = 'mag'
    MD_QUAT = 'quat'

    SENSOR_POSITIONS = ['rightLowerArm', 'rightUpperArm', 'back', 'leftUpperArm', 'leftLowerArm', 'rightCalf',
                        'rightThigh', 'leftThigh', 'leftCalf']
    LABEL_LIST = ['unknown', 'Walking', 'Jogging', 'Running', 'Jump up', 'Jump front & back', 'Jump sideways',
                  'Jump leg/arms open/closed', 'Jump rope', 'Trunk twist (arms outstretched)',
                  'Trunk twist (elbows bent)', 'Waist bends forward', 'Waist rotation',
                  'Waist bends (reach foot with opposite hand)', 'Reach heels backwards', 'Lateral bend',
                  'Lateral bend with arm up', 'Repetitive forward stretching',
                  'Upper trunk and lower body opposite twist', 'Lateral elevation of arms', 'Frontal elevation of arms',
                  'Frontal hand claps', 'Frontal crossing of arms', 'Shoulders high-amplitude rotation',
                  'Shoulders low-amplitude rotation', 'Arms inner rotation', 'Knees (alternating) to breast',
                  'Heels (alternating) to backside', 'Knees bending (crouching)', 'Knees (alternating) bending forward',
                  'Rotation on knees', 'Rowing', 'Elliptical bike', 'Cycling']

    @classmethod
    def define_att(cls):
        cls.RAW_COLS = ['sec', 'microsec']
        for pos in cls.SENSOR_POSITIONS:
            cls.RAW_COLS += [
                f'{pos}_{cls.MD_ACC}_x(m/s^2)', f'{pos}_{cls.MD_ACC}_y(m/s^2)', f'{pos}_{cls.MD_ACC}_z(m/s^2)',
                f'{pos}_{cls.MD_GYR}_x(rad/s)', f'{pos}_{cls.MD_GYR}_y(rad/s)', f'{pos}_{cls.MD_GYR}_z(rad/s)',
                f'{pos}_{cls.MD_MAG}_x(?)', f'{pos}_{cls.MD_MAG}_y(?)', f'{pos}_{cls.MD_MAG}_z(?)',
                f'{pos}_{cls.MD_QUAT}_w', f'{pos}_{cls.MD_QUAT}_x', f'{pos}_{cls.MD_QUAT}_y', f'{pos}_{cls.MD_QUAT}_z'
            ]
        cls.RAW_COLS.append('label')


RealDispConst.define_att()


class RealDispParquet(ParquetDatasetFormatter):
    def __init__(self, raw_folder: str, destination_folder: str, sampling_rates: float = 50,
                 min_length_segment: float = 5, max_interval: int = 500,
                 submodals: tuple = (RealDispConst.MD_ACC, RealDispConst.MD_GYR, RealDispConst.MD_MAG)):
        """
        Class for RealDisp dataset.

        Args:
            raw_folder: path to unprocessed dataset
            destination_folder: folder to save output
            sampling_rates: sampling rate to resample by linear interpolation (unit: Hz)
            min_length_segment: only write segments longer than this threshold (unit: sec)
            max_interval: dict[submodal] = maximum intervals (millisecond) between rows of an uninterrupted segment;
                default = 500 ms
            submodals: list of submodal to keep in the output
        """
        sampling_rates = {RealDispConst.MODAL_INERTIA: sampling_rates}
        super().__init__(raw_folder, destination_folder, sampling_rates)

        self.min_length_segment = min_length_segment * 1000
        self.max_interval = {RealDispConst.MODAL_INERTIA: max_interval}
        self.label_dict = dict(zip(range(len(RealDispConst.LABEL_LIST)), RealDispConst.LABEL_LIST))

        self.drop_cols = []
        for submodal in (RealDispConst.MD_ACC, RealDispConst.MD_GYR, RealDispConst.MD_MAG, RealDispConst.MD_QUAT):
            if submodal not in submodals:
                self.drop_cols += [c for c in RealDispConst.RAW_COLS if f'_{submodal}_' in c]

    def get_info_from_file_path(self, path: str) -> tuple:
        """
        Get session info from file path

        Args:
            path: path to a raw data csv file

        Returns:
            a tuple of (subject ID, session name)
        """
        info = path.split(os.sep)[-1]
        session_name = info.removesuffix('.log')
        subject_id = int(session_name.split('_')[0].removeprefix('subject'))

        return subject_id, session_name

    def read_raw_csv(self, path: str) -> pl.DataFrame:
        """
        Read and format a raw csv file.

        Args:
            path: path to csv file

        Returns:
            polars Dataframe
        """
        # read DF
        df = pl.read_csv(path, separator='\t', has_header=False, new_columns=RealDispConst.RAW_COLS)
        df = df.filter(pl.all_horizontal(pl.all().is_finite()))

        # convert timestamp
        df = df.with_columns((pl.col('sec') * 1e3 + pl.col('microsec') / 1e3).cast(pl.Int64).alias('timestamp(ms)'))
        df = df.drop('sec', 'microsec', *self.drop_cols)

        assert df.select(pl.col('timestamp(ms)').is_finite().all()).item(), 'Invalid timestamps!'
        return df

    def split_sessions(self, df: pl.DataFrame) -> List[pl.DataFrame]:
        """
        Split a dataframe into multiple uninterrupted dataframes.

        Args:
            df: original dataframe

        Returns:
            a list of uninterrupted dataframes
        """

        def split_uninterrupted(df: pl.DataFrame) -> List[pl.DataFrame]:
            """
            Split by gap in timestamp column.
            """
            df = df.set_sorted('timestamp(ms)')

            label_df = df.select('timestamp(ms)', 'label')
            df = df.drop('label')

            segment_dfs = split_interrupted_dfs(
                dfs={RealDispConst.MODAL_INERTIA: df},
                max_interval=self.max_interval,
                min_length_segment=self.min_length_segment,
                sampling_rates=self.sampling_rates
            )

            results = [
                segment_df[RealDispConst.MODAL_INERTIA].join_asof(label_df, on='timestamp(ms)', strategy='nearest')
                for segment_df in segment_dfs
            ]
            return results

        # one file may contain multiple sessions, split them into separate DFs first
        df = df.with_columns((pl.col('timestamp(ms)').diff().fill_null(1) < 0).cast(pl.Int32).cum_sum().alias('ss_n'))
        df = df.partition_by('ss_n', include_key=False)

        # check for interruption and interpolate each session
        results = []
        for session_df in df:
            results += split_uninterrupted(session_df)

        return results

    def run(self):
        written_files = 0
        skipped_sessions = 0
        skipped_files = 0

        # for each session
        for file in glob(f'{self.raw_folder}/*.log'):
            subject_id, session_id = self.get_info_from_file_path(file)

            # check if already run before
            if os.path.isfile(self.get_output_file_path(RealDispConst.MODAL_INERTIA, subject_id, f'{session_id}_last')):
                logger.info(f'Skipping session {session_id} because it has been done before.')
                skipped_sessions += 1
                continue
            logger.info(f'Starting session {session_id}')

            df = self.read_raw_csv(file)
            segment_dfs = self.split_sessions(df)

            # write each segment DF
            for seg_i, seg_df in enumerate(segment_dfs):
                segment_id = seg_i if seg_i != len(segment_dfs) - 1 else 'last'
                written = self.write_output_parquet(seg_df, RealDispConst.MODAL_INERTIA, subject_id,
                                                    f'{session_id}_{segment_id}')
                written_files += int(written)
                skipped_files += int(not written)

        logger.info(f'{written_files} file(s) written, {skipped_sessions} session(s) skipped, '
                    f'{skipped_files} file(s) skipped')

        # convert labels from text to numbers
        self.export_label_list()


class RealDispNpyWindow(NpyWindowFormatter):
    def __init__(self, scenarios: List[str] = None, *args, **kwargs):
        """
        Generate windows data from RealDisp dataset.

        Args:
            scenarios: tuple of the dataset's scenario names to get data
        """
        super().__init__(*args, **kwargs)
        if scenarios:
            for scenario in scenarios:
                assert scenario in {'ideal', 'self', 'mutual'}, 'scenario must be one of ideal, self or mutual'
        self.scenarios = scenarios

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
        # get list of parquet files
        if self.scenarios:
            parquet_sessions = []
            for scenario in self.scenarios:
                parquet_sessions.append(self.get_parquet_file_list(session_pattern=f'*_{scenario}*'))
            parquet_sessions = pl.concat(parquet_sessions, how='vertical')
        else:
            parquet_sessions = self.get_parquet_file_list()

        result = []
        # for each session
        for parquet_session in tqdm(parquet_sessions.iter_rows(named=True), total=len(parquet_sessions)):
            # get session info
            _, subject, _ = self.get_parquet_session_info(list(parquet_session.values())[0])

            session_result = self.parquet_to_windows(parquet_session=parquet_session, subject=subject)
            result.append(session_result)
        result = pd.DataFrame(result)
        return result


if __name__ == '__main__':
    parquet_dir = '/home/nda97531/Documents/dataset_parquet/RealDisp'

    # RealDispParquet(
    #     raw_folder='/home/nda97531/Documents/realdisp+activity+recognition+dataset',
    #     destination_folder=parquet_dir,
    #     sampling_rates=50,
    #     min_length_segment=5
    # ).run()

    dataset_window = RealDispNpyWindow(
        parquet_root_dir=parquet_dir,
        window_size_sec=4,
        step_size_sec=2,
        # modal_cols={
        #     RealWorldConst.MODAL_INERTIA: {
        #         'waist': ['waist_acc_x(m/s^2)', 'waist_acc_y(m/s^2)', 'waist_acc_z(m/s^2)']
        #     }
        # },
        scenarios=['mutual'],
    ).run()
    _ = 1
