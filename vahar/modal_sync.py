import polars as pl
import numpy as np
from loguru import logger

from my_py_utils.my_py_utils.number_array import interval_intersection
from my_py_utils.my_py_utils.pl_dataframe import resample_numeric_df as pl_resample_numeric_df


def split_interrupted_dfs(dfs: dict, max_interval: dict, min_length_segment: float,
                          sampling_rates: dict) -> list:
    """
    Split an interrupted session into multiple uninterrupted sub-sessions with even timestamps (interpolated)

    Args:
        dfs: dict of DFs,
            - key: modal name
            - value: polars DF with a column 'timestamp(ms)' and feature columns
        max_interval: dict
            - key: modal name
            - value: max interval between rows of an uninterrupted DF, unit: millisecond
        min_length_segment: drop sessions shorter than this threshold, unit: millisecond
        sampling_rates: dict:
            - key: modal name
            - value: sampling rate to interpolate, unit: sample/millisecond

    Returns:
        list of dicts, each dict is an uninterrupted segment and has the same format as the input dict
    """
    # split interrupted signals into sub-sessions
    # key: modal; value: list of pairs [start ts, end ts] for each segment
    ts_segments = {}
    for modal, df in dfs.items():
        ts = df.get_column('timestamp(ms)').to_numpy()
        intervals = np.diff(ts)
        interruption_idx = np.nonzero(intervals > max_interval[modal])[0]
        interruption_idx = np.concatenate([[-1], interruption_idx, [len(intervals)]])
        ts_segments[modal] = [
            [ts[interruption_idx[i - 1] + 1], ts[interruption_idx[i]]]
            for i in range(1, len(interruption_idx))
        ]
    combined_ts_segments = interval_intersection(list(ts_segments.values()))
    logger.info(f'Number of segments: ' + '; '.join(f'{k}: {len(v)}' for k, v in ts_segments.items()))

    # crop segments based on timestamps found above
    results = []
    kept_segments = 0
    kept_time = 0
    total_time = 0
    # for each segment
    for combined_ts_segment in combined_ts_segments:
        segment_length = combined_ts_segment[1] - combined_ts_segment[0]
        total_time += segment_length
        if segment_length < min_length_segment:
            continue
        kept_time += segment_length
        kept_segments += 1

        # dict key: modal; value: uninterrupted DF
        segment_dfs = {}
        # for each sensor
        for modal, df in dfs.items():
            # crop the segment in sensor DF
            df = df.filter(pl.col('timestamp(ms)').is_between(combined_ts_segment[0].item(),
                                                              combined_ts_segment[1].item()))

            # interpolate to resample
            df = pl_resample_numeric_df(df, 'timestamp(ms)', sampling_rates[modal],
                                        start_ts=combined_ts_segment[0], end_ts=combined_ts_segment[1])

            segment_dfs[modal] = df

        results.append(segment_dfs)
    logger.info(f'Kept {kept_segments}/{len(combined_ts_segments)} segment(s)')
    logger.info('Kept %.02f/%.02f (sec); %.02f%%' % (kept_time / 1000, total_time / 1000, kept_time / total_time * 100))
    return results
