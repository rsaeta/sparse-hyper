import pandas as pd


def get_real_columns(df: pd.DataFrame):
    """Return the columns that we actually care about."""
    return [c for c in df.columns if '__MIN' not in c and '__MAX' not in c and c != 'Step']


def get_first_drop_step(series: pd.Series, threshold=1e-3):
    """Return the first step in a series where the value drops."""
    for i, value in enumerate(series):
        if value < threshold:
            return i
    return None


def get_first_drop_step_df(df: pd.DataFrame, threshold=1e-3):
    """
    Returns the indices of the first drop in each column of a dataframe that
    we actually care about
    """
    cols = get_real_columns(df)
    df = df[cols]
    return df.apply(get_first_drop_step, threshold=threshold)


def get_instability_metric(series: pd.Series, threshold=1e-3):
    """
    Calculates the instability metric that is the percentage of steps that
    are above the threshold after the first drop.
    """
    i = get_first_drop_step(series, threshold=threshold)
    if i is None:
        return None
    series = series.iloc[i:]
    successful = series[series < threshold]
    return len(successful) / len(series)


def get_instability_metric_df(df: pd.DataFrame, threshold=1e-3):
    """
    Calculates the instability metric for each column in a dataframe that we
    actually care about.
    """
    cols = get_real_columns(df)
    df = df[cols]
    return df.apply(get_instability_metric, threshold=threshold)


if __name__ == '__main__':
    df = pd.read_csv('offset_200_loss.csv')
    f = get_first_drop_step_df(df)
    m = get_instability_metric_df(df)
    print(f)
    print(m)
    breakpoint()
