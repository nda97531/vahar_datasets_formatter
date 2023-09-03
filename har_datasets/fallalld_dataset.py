import os
import re
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from har_datasets.base_classes import ParquetDatasetFormatter, NpyWindowFormatter
from har_datasets.constant import G_TO_MS2, DEG_TO_RAD
from my_py_utils.my_py_utils.number_array import interp_resample


class FallAllDConst:
    # modal names
    MODAL_INERTIA = 'inertia'

    RAW_SAMPLE_RATE = {'Acc': 238, 'Gyr': 238}
    SENSOR_SI_UNIT = {'Acc': 'm/s^2', 'Gyr': 'rad/s'}


class FallAllDParquet(ParquetDatasetFormatter):
    def __init__(self, raw_folder: str, destination_folder: str, sampling_rates: dict):
        super().__init__(raw_folder, destination_folder, sampling_rates)

    def read_dataset_pkl_file(self, path: str) -> pd.DataFrame:
        """
        Read raw pkl file, filter out unnecessary rows and columns, convert data unit

        Args:
            path: path to dataset pkl file

        Returns:
            a DF with columns: SubjectID, Device, ActivityID, TrialNo, Acc, Gyr;
        """
        df = pd.read_pickle(path)
        logger.info(f'Raw pkl dataframe: {df.shape}')
        df = df[['SubjectID', 'Device', 'ActivityID', 'TrialNo', 'Acc', 'Gyr']]

        # only use 2 sensors: waist and wrist
        df = df.loc[df['Device'].isin({'Waist', 'Wrist'})].reset_index(drop=True)
        logger.info(f'Filter column and device rows: {df.shape}')

        # convert unit to SI (m/s^2 and rad/s)
        acc_conversion_coeff = 8 / 32767 * G_TO_MS2
        gyr_conversion_coeff = 2000 / 32767 * DEG_TO_RAD
        formatted_acc = [arr * acc_conversion_coeff for arr in df['Acc']]
        formatted_gyr = [arr * gyr_conversion_coeff for arr in df['Gyr']]

        # interpolate
        formatted_acc = [
            interp_resample(arr, old_freq=FallAllDConst.RAW_SAMPLE_RATE['Acc'] / 1000,
                            new_freq=self.sampling_rates[FallAllDConst.MODAL_INERTIA])
            for arr in formatted_acc
        ]
        formatted_gyr = [
            interp_resample(arr, old_freq=FallAllDConst.RAW_SAMPLE_RATE['Gyr'] / 1000,
                            new_freq=self.sampling_rates[FallAllDConst.MODAL_INERTIA])
            for arr in formatted_gyr
        ]

        df['Acc'] = formatted_acc
        df['Gyr'] = formatted_gyr
        return df

    def group_sessions(self, dataset_df: pd.DataFrame):
        """
        Group data from different sensors but from the same sessions

        Args:
            dataset_df: a DF with columns: SubjectID, Device, ActivityID, TrialNo, Acc, Gyr

        Returns:
            list of dicts, each dict is a session with the following keys:
                subject, activity, trial, data (data value is a polar DF).
            Device info is added into data column names.
        """
        n_info_cols = 4
        assert list(dataset_df.columns)[:n_info_cols] == ['SubjectID', 'Device', 'ActivityID', 'TrialNo'], \
            f'Wrong dataset columns: {dataset_df.columns}; Expected: [SubjectID, Device, ActivityID, TrialNo]'

        results = []
        groups = dataset_df.groupby(['SubjectID', 'ActivityID', 'TrialNo'])
        # for each unique session
        for (subject, activity, trial), session_group in groups:
            assert len(session_group) == len(np.unique(session_group['Device'])), \
                f'Duplicate sensor in session: Subject {subject}, Activity {activity}, Trial {trial}'

            data_arr = []
            data_cols = []
            device_used = []
            # for each device used in the session
            for _, sensor_data in session_group.iterrows():
                # get info: device position (waist, wrist, ...) and sensor types of this device (acc, gyr, ...)
                device_pos = sensor_data.at['Device'].lower()
                sensor_types = list(sensor_data.index)[n_info_cols:]

                # record data of this device into some lists
                device_used.append(device_pos)
                data_arr += sensor_data.iloc[n_info_cols:].tolist()
                data_cols += [
                    f'{device_pos}_{sensor_type.lower()}_{axis}({FallAllDConst.SENSOR_SI_UNIT[sensor_type]})'
                    for sensor_type in sensor_types
                    for axis in ['x', 'y', 'z']
                ]

            # concat data of all devices into 1 DF
            max_len = max(len(arr) for arr in data_arr)
            data_arr = [arr[:max_len] for arr in data_arr]
            data_df = pl.DataFrame(np.concatenate(data_arr, axis=1), schema=data_cols)

            # record session result
            results.append({
                'subject': subject, 'activity': activity, 'trial': trial, 'device': device_used, 'data': data_df
            })
        logger.info(f'Total number of sessions: {len(results)}')
        return results

    def add_ts_and_label(self, df: pl.DataFrame, label: int) -> pl.DataFrame:
        """
        Add timestamp(ms) and label columns to dataframe (a session)

        Args:
            df: data DF
            label: raw label of session

        Returns:
            DF with 2 new columns 'timestamp(ms)' and 'label'
        """
        # create timestamp array
        ts_arr = (np.arange(len(df)) / self.sampling_rates[FallAllDConst.MODAL_INERTIA]).astype(int)
        # init label array with all 0 values
        label_arr = np.zeros(len(df), dtype=int)

        # if this is a fall session
        if label >= 100:
            # find an accelerometer using a sorted priority list
            all_columns = df.columns
            for device in ['waist', 'wrist', 'neck']:
                device_acc_cols = [c for c in all_columns if f'{device}_acc' in c]
                if len(device_acc_cols) == 3:
                    break
            else:
                raise ValueError(f'No tri-axial accelerometer found in: {all_columns}')

            # find peak acceleration (find fall impact index)
            acc_arr = df.select(device_acc_cols).to_numpy()
            peak_idx = np.argmax((acc_arr ** 2).sum(1))

            # fill fall label (label = 1)
            fall_ts = ts_arr[peak_idx]
            label_arr[(fall_ts - 1000 <= ts_arr) & (ts_arr <= fall_ts + 1000)] = 1

        # add columns to DF
        df = df.with_columns(
            pl.Series(name='timestamp(ms)', values=ts_arr),
            pl.Series(name='label', values=label_arr)
        )
        return df

    def run(self):
        # read and process
        whole_dataset = self.read_dataset_pkl_file(f'{self.raw_folder}/FallAllD.pkl')
        whole_dataset = self.group_sessions(whole_dataset)

        # write
        skipped_sessions = 0
        written_files = 0
        # for each session
        for session in whole_dataset:
            # get session info
            subject = session['subject']
            activity = session['activity']
            trial = session['trial']
            device = '-'.join(session['device'])
            session_info = f'subject{subject}_act{activity}_trial{trial}_device{device}'

            # check if already run before
            if os.path.isfile(self.get_output_file_path(FallAllDConst.MODAL_INERTIA, subject, session_info)):
                logger.info(f'Skipping session {session_info} because already run before')
                skipped_sessions += 1
                continue
            logger.info(f'Starting session {session_info}')

            # get data DF
            data_df = session['data']
            # add timestamp and label
            data_df = self.add_ts_and_label(data_df, activity)

            # write file
            if self.write_output_parquet(data_df, FallAllDConst.MODAL_INERTIA, subject, session_info):
                written_files += 1

        logger.info(f'{written_files} file(s) written, {skipped_sessions} session(s) skipped')


class FallAllDNpyWindow(NpyWindowFormatter):
    def get_parquet_file_list(self) -> pl.DataFrame:
        """
        Override parent class method to filter out sessions that don't have required inertial sub-modals
        """
        df = super().get_parquet_file_list()
        if FallAllDConst.MODAL_INERTIA not in df.columns:
            return df

        sub_modals = np.unique([
            col.split('_')[0]
            for col in np.concatenate(list(self.modal_cols[FallAllDConst.MODAL_INERTIA].values()))
        ])
        df = df.filter(pl.all(
            pl.col(FallAllDConst.MODAL_INERTIA).str.contains(submodal)
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

            session_result = self.process_parquet_to_windows(
                parquet_session=parquet_session, subject=subject, session_label=int(session_label),
                is_short_activity=session_label if shift_short_activity else False
            )
            result.append(session_result)
        result = pd.DataFrame(result)
        return result


if __name__ == '__main__':
    parquet_dir = '/mnt/data_drive/projects/UCD04 - Virtual sensor fusion/processed_parquet/FallAllD'

    # FallAllDParquet(
    #     raw_folder='/mnt/data_drive/projects/raw datasets/FallAllD/',
    #     destination_folder=parquet_dir,
    #     sampling_rates={FalAllDConstant.MODAL_INERTIA: 50}
    # ).run()

    dataset_window = FallAllDNpyWindow(
        parquet_root_dir=parquet_dir,
        window_size_sec=4,
        step_size_sec=2,
        min_step_size_sec=0.5,
        max_short_window=5,
        modal_cols={
            FallAllDConst.MODAL_INERTIA: {
                'waist': ['waist_acc_x(m/s^2)', 'waist_acc_y(m/s^2)', 'waist_acc_z(m/s^2)'],
                'wrist': ['wrist_acc_x(m/s^2)', 'wrist_acc_y(m/s^2)', 'wrist_acc_z(m/s^2)']
            }
        }
    ).run()
    _ = 1
