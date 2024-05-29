import os
import re
from glob import glob
from typing import Tuple
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from scipy.stats import mode
import json
from my_py_utils.my_py_utils.sliding_window import shifting_window, sliding_window
from my_py_utils.my_py_utils.string_utils import rreplace

MODAL_PATH_PATTERN = '{root}/{modal}'
PARQUET_PATH_PATTERN = MODAL_PATH_PATTERN + '/subject_{subject}/{session}.parquet'


class ParquetDatasetFormatter:
    """
    This class processes raw dataset and save as parquet files in a structured directory.
    """

    def __init__(self, raw_folder: str, destination_folder: str, sampling_rates: dict):
        """
        This class transforms public datasets into the same format for ease of use.

        Args:
            raw_folder: path to unprocessed dataset
            destination_folder: folder to save output
            sampling_rates: a dict containing sampling rates of each modal to resample by linear interpolation.
                - key: modal name
                - value: sampling rate (unit: Hz)
        """
        self.raw_folder = raw_folder
        self.destination_folder = destination_folder

        # convert Hz to sample/msec
        self.sampling_rates = {k: v / 1000 for k, v in sampling_rates.items()}

        self.label_dict: dict = {}

    def get_output_file_path(self, modal, subject, session) -> str:
        """
        Get path to an output file (.parquet)

        Args:
            modal: modality
            subject: subject ID
            session: session ID

        Returns:
            path to parquet file
        """
        p = PARQUET_PATH_PATTERN.format(root=self.destination_folder, modal=modal, subject=subject,
                                        session=session)
        return p

    def write_output_parquet(self, data: pl.DataFrame, modal: str, subject: any, session: any) -> bool:
        """
        Write a processed DataFrame of 1 modality, 1 session

        Args:
            data: a DF containing data of 1 modality, 1 session
            modal: modality name (e.g. accelerometer, skeleton)
            subject: subject name/ID
            session: session ID

        Returns:
            boolean, file written successfully or not
        """
        assert 'label' in data.columns, 'No "label" column in output DF'
        assert 'timestamp(ms)' in data.columns, 'No "timestamp(ms)" column in output DF'

        df_interval = data.item(1, 'timestamp(ms)') - data.item(0, 'timestamp(ms)')
        expected_interval = 1 / self.sampling_rates[modal]
        assert df_interval == expected_interval, \
            (f'Unexpected timestamp interval in output DF. '
             f'Expected: {expected_interval}(ms), but actual: {df_interval}(ms)')

        output_path = self.get_output_file_path(modal=modal, subject=subject, session=session)

        # check if there's any NAN
        if np.isnan(data.to_numpy()).sum():
            logger.error(f'NAN in data!! Skipping this file: {output_path}')
            return False

        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        data.write_parquet(output_path)
        logger.info(f'Parquet shape {data.shape} written: {output_path}')
        return True

    def export_label_list(self):
        """
        Write label list to a JSON file.
        """
        assert self.label_dict, 'Class dict not defined.'
        output_path = f'{self.destination_folder}/label_list.json'
        if os.path.isfile(output_path):
            logger.info('Skipping label list before it has been written before.')
            return
        with open(output_path, 'w') as F:
            json.dump(self.label_dict, F)
        logger.info(f'Label list written to file: {output_path}')

    def run(self):
        """
        Main processing method
        """
        # to override: process data of any dataset and
        # 1. call self.get_output_file_path for each session to check if it has already been processed
        # 2. call self.write_output_parquet() for every modal of every session
        # 3. call self.export_class_list to export class list to a JSON file
        raise NotImplementedError()


class NpyWindowFormatter:
    def __init__(self, parquet_root_dir: str,
                 window_size_sec: float, step_size_sec: float,
                 min_step_size_sec: float = None, max_short_window: int = None,
                 exclude_labels: tuple = (), no_transition: bool = False,
                 modal_cols: dict = None):
        """
        This class takes result of `ParquetDatasetFormatter`, run sliding window and return as numpy array.
        If a session is of short activities, only take windows containing those activities and assign those labels
        (without needing to vote within windows). For example: a 20-second session has only 2-second Fall activity,
        then this codes only take windows including that 2-second segment.

        Args:
            parquet_root_dir: path to processed parquet root dir
            window_size_sec: window size in second
            step_size_sec: step size in second
            min_step_size_sec: for short activity sessions, this is used as minimum step size (in shifting window)
            max_short_window: max number of window for short activity sessions (use shifting window)
            exclude_labels: drop any window containing one of these labels BEFORE voting a label for each window;
                not applicable for short activity sessions
            no_transition: if True, drop any window with more than 1 label before voting a label for each window;
                not applicable for short activity sessions
            modal_cols: a 2-level dict;
                1st level key is modal name (match with modal name in parquet file path),
                2nd level key is sub-modal names from the same parquet files (any new name),
                2nd level value is a list of column names of that sub-modal. if None, use all columns.
                Example
                    {
                        'inertia': {
                            'acc': ['acc_x', 'acc_y', 'acc_z'],
                            'gyro': ['gyro_x', 'gyro_y', 'gyro_z'],
                        },
                        'skeleton': {
                            'skeleton': None
                        }
                    }
                    In this case, 'acc' and 'gyro' are from the same parquet file of modal 'inertia'
                if `modal_cols` is None (default),
                sub-modal will be the same as modal in parquet path, and all columns are used
        """
        self.parquet_root_dir = parquet_root_dir
        self.window_size_sec = window_size_sec
        self.step_size_sec = step_size_sec
        self.min_step_size_sec = min_step_size_sec
        self.max_short_window = max_short_window
        self.no_transition = no_transition
        self.exclude_labels = np.array(exclude_labels)
        assert len(exclude_labels) == len(self.exclude_labels)

        # standardise `modal_cols`
        # compose dict of used columns if it has not already been defined
        if modal_cols is None:
            parquet_modals = self.get_parquet_modals()
            modal_cols = {modal: {modal: None} for modal in parquet_modals}
        else:
            for parquet_modal, sub_modal_dict in modal_cols.items():
                if sub_modal_dict is None:
                    modal_cols[parquet_modal] = {parquet_modal: None}
        self.modal_cols = modal_cols

        # flag to control log printing column names
        self.verbose = True

    def get_parquet_modals(self) -> list:
        """
        Get a list of modal names of this dataset

        Returns:
            list of strings
        """
        # scan for modal list first
        modal_folders = sorted(glob(MODAL_PATH_PATTERN.format(root=self.parquet_root_dir, modal='*/')))
        modals = [p.removesuffix('/').split('/')[-1] for p in modal_folders]
        return modals

    def get_parquet_file_list(self, subject_pattern: str = '*', session_pattern: str = '*') -> pl.DataFrame:
        """
        Scan all parquet files in the root dir

        Args:
            subject_pattern: wildcard string for subject ID
            session_pattern: wildcard string for session ID

        Returns:
            a DataFrame, each column is a data modality, each row is a session, cells are paths to parquet files
        """
        modals = list(self.modal_cols.keys())

        # glob first modal
        first_modal_parquets = sorted(glob(PARQUET_PATH_PATTERN.format(
            root=self.parquet_root_dir,
            modal=modals[0], subject=subject_pattern, session=session_pattern
        )))
        if len(modals) == 1:
            return pl.DataFrame({modals[0]: first_modal_parquets})

        # check that all modals have the same number of parquet files
        for modal in modals[1:]:
            next_modal_parquets = glob(PARQUET_PATH_PATTERN.format(
                root=self.parquet_root_dir,
                modal=modal, subject=subject_pattern, session=session_pattern
            ))
            assert len(first_modal_parquets) == len(next_modal_parquets), \
                f'{modals[0]} has {len(first_modal_parquets)} parquet files but {modal} has {len(next_modal_parquets)}'

        # get matching session files of all modals
        result = []
        for pq_path in first_modal_parquets:
            session_dict = {modals[0]: pq_path}
            for modal in modals[1:]:
                session_dict[modal] = rreplace(pq_path, f'/{modals[0]}/', f'/{modal}/')
                assert os.path.isfile(session_dict[modal]), f'Modal parquet file not exist: {session_dict[modal]}'
            result.append(session_dict)

        parquet_files = pl.DataFrame(result)
        logger.info(f'Found {len(parquet_files)} parquets file(s).')
        assert len(parquet_files), 'Cannot find any parquet file, please check data path.'
        return parquet_files

    def get_parquet_session_info(self, parquet_path: str) -> Tuple[str, ...]:
        """
        Get session info from parquet file path

        Args:
            parquet_path: parquet file path

        Returns:
            a tuple: (modality, subject id, session id) all elements are string
        """
        info = re.search(PARQUET_PATH_PATTERN.format(
            root=self.parquet_root_dir,
            modal='(.*)', subject='(.*)', session='(.*)'
        ), parquet_path)
        info = tuple(info.group(i) for i in range(1, 4))
        return info

    def slide_windows_from_modal_df(self, df: pl.DataFrame, modality: str, session_label: int = None,
                                    is_short_activity: bool = False) -> dict:
        """
        Slide windows from 1 session dataframe of 1 modal.
        If main activity of the session is a short activity, run shifting window instead.

        Args:
            df: Dataframe with 'timestamp(ms)' and 'label' columns, others are feature columns
            modality: data modality of this DF
            session_label: main label of this session, only used if `is_short_activity` is True
            is_short_activity: whether this session is of short activities.
                Only support short activities of ONE label in a session

        Returns:
            a dict, keys are sub-modal name from `self.modal_cols` and 'label', values are np array containing windows.
            Example
                {
                    'acc': array [num windows, window length, features]
                    'gyro': array [num windows, window length, features]
                    'label': array [num windows], dtype: NOT float
                }
        """
        # calculate window size row from window size sec
        # Hz can be calculated from first 2 rows because this DF is already interpolated (constant interval timestamps)
        df_sampling_rate = df.head(2).get_column('timestamp(ms)').to_list()
        df_sampling_rate = 1000 / (df_sampling_rate[1] - df_sampling_rate[0])
        window_size_row = int(self.window_size_sec * df_sampling_rate)

        # if this is a session of short activity, run shifting window
        if is_short_activity:
            min_step_size_row = int(self.min_step_size_sec * df_sampling_rate)

            # find short activity indices
            org_label = df.get_column('label').to_numpy()
            bin_label = org_label == session_label
            bin_label = np.concatenate([[False], bin_label, [False]])
            bin_label = np.diff(bin_label)
            start_end_idx = bin_label.nonzero()[0].reshape([-1, 2])
            start_end_idx[:, 1] -= 1

            # shifting window for each short activity occurrence
            windows = np.concatenate([
                shifting_window(df.to_numpy(), window_size=window_size_row,
                                max_num_windows=self.max_short_window,
                                min_step_size=min_step_size_row, start_idx=start, end_idx=end)
                for start, end in start_end_idx
            ])

            # if this is a short activity, assign session label
            windows_label = np.full(shape=len(windows), fill_value=session_label, dtype=int)

        # if this is a session of long activity, run sliding window
        else:
            step_size_row = int(self.step_size_sec * df_sampling_rate)
            windows = sliding_window(df.to_numpy(), window_size=window_size_row, step_size=step_size_row)

            # vote 1 label for each window
            windows_label = windows[:, :, df.columns.index('label')].astype(int)

            if self.no_transition:
                # drop windows containing label transitions
                no_trans_idx = np.array([len(np.unique(windows_label[i])) == 1 for i in range(len(windows_label))])
                windows = windows[no_trans_idx]
                windows_label = windows_label[no_trans_idx]
            if len(self.exclude_labels):
                # drop windows containing disallowed labels
                keep_idx = ~np.array([any(lb in row for lb in self.exclude_labels) for row in windows_label])
                windows = windows[keep_idx]
                windows_label = windows_label[keep_idx]

            windows_label = mode(windows_label, axis=-1, nan_policy='raise', keepdims=False).mode

        # list of sub-modals within the DF
        sub_modals_col_idx = self.modal_cols[modality].copy()
        # get column index from column name for each sub-modal (later used for picking columns in np array)
        for k, v in sub_modals_col_idx.items():
            if v is None:
                # get all feature cols by default
                list_idx = list(range(df.shape[1]))
                list_idx.remove(df.columns.index('timestamp(ms)'))
                list_idx.remove(df.columns.index('label'))
                sub_modals_col_idx[k] = list_idx
            else:
                # get specified cols
                sub_modals_col_idx[k] = [df.columns.index(col) for col in v if col in df.columns]

        if self.verbose:
            for sub_modal, sub_modal_col_idx in sub_modals_col_idx.items():
                logger.info(f'Sub-modal: {sub_modal}; '
                            f'{len(sub_modal_col_idx)} cols: {[df.columns[i] for i in sub_modal_col_idx]}')

        # split windows by sub-modal
        result = {sub_modal: windows[:, :, sub_modal_col_idx]
                  for sub_modal, sub_modal_col_idx in sub_modals_col_idx.items()
                  if sub_modal_col_idx}
        result['label'] = windows_label

        return result

    def parquet_to_windows(self, parquet_session: dict, subject: any, session_label: int = None,
                           is_short_activity: bool = False) -> dict:
        """
        Process from parquet files of ONE session to window data (np array).

        Args:
            parquet_session: dict with keys are modal names, values are parquet file paths
            subject: subject ID
            session_label: main label of this session
            is_short_activity: whether this session is of short activities

        Returns:
            a dict, keys are all sub-modal names of a session, 'subject' and 'label';
            values are np array containing windows.
            Example
                {
                    'acc': array [num windows, window length, features]
                    'gyro': array [num windows, window length, features]
                    'skeleton': array [num windows, window length, features]
                    'label': array [num windows], dtype: int
                    'subject': subject ID
                }
        """
        # dict to be returned
        session_result = {}
        # list of window labels of each modal
        modal_labels = []
        # number of windows, shared among all modals
        min_num_windows = float('inf')

        # for each modal, run sliding window
        for modal, parquet_file in parquet_session.items():
            if modal not in self.modal_cols:
                continue
            # read DF
            df = pl.read_parquet(parquet_file)
            # sliding window
            windows = self.slide_windows_from_modal_df(df=df, modality=modal, session_label=session_label,
                                                       is_short_activity=is_short_activity)

            # append result of this modal
            min_num_windows = min(min_num_windows, len(windows['label']))
            modal_labels.append(windows.pop('label'))
            session_result.update(windows)

        # make all modals have the same number of windows (they may be different because of sampling rates)
        session_result = {k: v[:min_num_windows] for k, v in session_result.items()}

        # add subject info
        session_result['subject'] = int(subject)

        # check if label of all modals are the same
        for modal_label in modal_labels[1:]:
            diff_lb = modal_label[:min_num_windows] != modal_labels[0][:min_num_windows]
            if diff_lb.any():
                modal_labels[0][:min_num_windows] = np.maximum(modal_labels[0][:min_num_windows],
                                                               modal_label[:min_num_windows])
        # add label info; only need to take labels of the first modal because all modalities have the same labels
        session_result['label'] = modal_labels[0][:min_num_windows]

        # only print column names for the first session
        self.verbose = False
        return session_result

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
        # to override:
        # step 1: call self.get_parquet_file_list() to get data parquet file paths of all sessions
        # for each session:
        # step 2: call self.get_parquet_session_info() to get session info if needed
        # step 3: call self.parquet_to_windows() to run sliding window on each session
        raise NotImplementedError()
