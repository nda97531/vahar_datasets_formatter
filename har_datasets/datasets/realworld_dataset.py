import re
import zipfile
import numpy as np
import pandas as pd
import polars as pl
from glob import glob

from my_py_utils.my_py_utils.string_utils import rreplace

if __name__ == '__main__':
    from har_datasets.datasets.base_classes import ParquetDatasetFormatter, NpyWindowFormatter
else:
    from .base_classes import ParquetDatasetFormatter, NpyWindowFormatter


class RealWorldConst:
    # modal names
    MODAL_INERTIA = 'inertia'

    RAW_MODALS = ['acc', 'gyr', 'mag', 'lig', 'mic', 'gps']
    CLASS_LABELS = ['walking', 'running', 'sitting', 'standing', 'lying', 'climbingup', 'climbingdown', 'jumping']


class RealWorldParquet(ParquetDatasetFormatter):
    def __init__(self, raw_folder: str, destination_folder: str, sampling_rates: dict,
                 sub_modals: dict = {'inertia': ['acc', 'gyr']}
                 ):
        """
        Class for RealWorld2016 dataset.
        In this dataset, raw modals are considered as sub-modals. For example, modal 'inertia' contains 3 sub-modals:
        [acc, gyr, mag], which are also raw modals.

        Args:
            sub_modals: a dict containing sub-modal names of each modal
                - key: modal name (any name), this will be used in output paths
                - value: list of sub-modal names, choices are in RealWorldConst.RAW_MODALS
        """
        super().__init__(raw_folder, destination_folder, sampling_rates)
        self.sub_modals = sub_modals

    def get_list_sessions(self) -> pl.DataFrame:
        """
        Scan all session files

        Returns:
            a DF, each row is a session, columns are sub-modal names, each cell contains a file path
        """
        first_submodal = RealWorldConst.RAW_MODALS[0]
        files = {
            first_submodal: sorted(glob(f'{self.raw_folder}/proband*/data/{first_submodal}_*_csv.zip'))
        }
        for sub_modal in RealWorldConst.RAW_MODALS[1:]:
            files[sub_modal] = [rreplace(f, first_submodal, sub_modal) for f in files[first_submodal]]
        df = pl.DataFrame(files)
        return df

    def read_csv_in_zip(self, zip_file: str, csv_file: str) -> pl.DataFrame:


    def run(self):
        # scan all sessions
        list_sessions = self.get_list_sessions()

        # for each session
        for session_modals in list_sessions.iter_rows():

            # for each raw modal zip file of the session
            for submodal_file in session_modals:
                with zipfile.ZipFile(submodal_file, 'r') as zf:
                    compressed_list = [item for item in zf.namelist() if item.endswith('.csv')]

                    # for each device csv file in the zip file
                    for csv_file in compressed_list:
                        modal_device_df = pl.read_csv(zf.read(csv_file))

                        _ = 1

        # # write
        # skipped_sessions = 0
        # written_files = 0
        # # for each session
        # for session in whole_dataset:
        #     # get session info
        #     subject = session['subject']
        #     activity = session['activity']
        #     trial = session['trial']
        #     device = '-'.join(session['device'])
        #     session_info = f'subject{subject}_act{activity}_trial{trial}_device{device}'
        #
        #     # check if already run before
        #     if os.path.isfile(self.get_output_file_path(RealWorldConst.MODAL_INERTIA, subject, session_info)):
        #         logger.info(f'Skipping session {session_info} because already run before')
        #         skipped_sessions += 1
        #         continue
        #     logger.info(f'Starting session {session_info}')
        #
        #     # get data DF
        #     data_df = session['data']
        #     # add timestamp and label
        #     data_df = self.add_ts_and_label(data_df, activity)
        #
        #     # write file
        #     if self.write_output_parquet(data_df, RealWorldConst.MODAL_INERTIA, subject, session_info):
        #         written_files += 1
        #
        # logger.info(f'{written_files} file(s) written, {skipped_sessions} session(s) skipped')


class RealWorldNpyWindow(NpyWindowFormatter):
    def get_parquet_file_list(self) -> pl.DataFrame:
        """
        Override parent class method to filter out sessions that don't have required inertial sub-modals
        """
        df = super().get_parquet_file_list()
        if RealWorldConst.MODAL_INERTIA not in df.columns:
            return df

        sub_modals = np.unique([
            col.split('_')[0]
            for col in np.concatenate(list(self.modal_cols[RealWorldConst.MODAL_INERTIA].values()))
        ])
        df = df.filter(pl.all(
            pl.col(RealWorldConst.MODAL_INERTIA).str.contains(submodal)
            for submodal in sub_modals
        ))
        return df

    def run(self, shift_short_activity: bool = True) -> pd.DataFrame:
        """

        Args:
            shift_short_activity: whether to run shifting windows on fall sessions (or just run normal sliding window)

        Returns:
            please see parent class's method
        """
        # get list of parquet files
        parquet_sessions = self.get_parquet_file_list()

        result = []
        # for each session
        for parquet_session in parquet_sessions.iter_rows(named=True):
            # get session info
            modal, subject, session_id = self.get_parquet_session_info(list(parquet_session.values())[0])

            session_label = int(re.search(r'_act([0-9]*)_', session_id).group(1))
            # 0: non-fall; 1: fall
            session_label = session_label >= 100

            session_result = self.parquet_to_windows(
                parquet_session=parquet_session, subject=subject, session_label=int(session_label),
                is_short_activity=session_label if shift_short_activity else False
            )
            result.append(session_result)
        result = pd.DataFrame(result)
        return result


if __name__ == '__main__':
    parquet_dir = '/mnt/data_drive/projects/UCD04 - Virtual sensor fusion/processed_parquet/RealWorld'

    RealWorldParquet(
        raw_folder='/mnt/data_drive/projects/raw datasets/realworld2016_dataset',
        destination_folder=parquet_dir,
        sampling_rates={RealWorldConst.MODAL_INERTIA: 50}
    ).run()

    # dataset_window = RealWorldNpyWindow(
    #     parquet_root_dir=parquet_dir,
    #     window_size_sec=4,
    #     step_size_sec=2,
    #     min_step_size_sec=0.5,
    #     max_short_window=5,
    #     modal_cols={
    #         RealWorldConst.MODAL_INERTIA: {
    #             'waist': ['waist_acc_x(m/s^2)', 'waist_acc_y(m/s^2)', 'waist_acc_z(m/s^2)'],
    #             'wrist': ['wrist_acc_x(m/s^2)', 'wrist_acc_y(m/s^2)', 'wrist_acc_z(m/s^2)']
    #         }
    #     }
    # ).run()
    _ = 1
