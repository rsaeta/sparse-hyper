import re
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


def merge_mu_dfs(mu0: pd.DataFrame, mu1: pd.DataFrame):
    """
    Merges the dataframes of the mu's from the same run
    """
    real_cols = get_real_columns(mu0)
    mu0 = mu0[real_cols]
    real_cols = get_real_columns(mu1)
    mu1 = mu1[real_cols]
    return pd.concat([mu0, mu1], axis=1)


def merge_dfs(*dfs: list[pd.DataFrame]):
    """
    Merges the dataframes on step
    """
    df = dfs[0]
    for _df in dfs[1:]:
        df = df.merge(_df, on='Step', how='outer')
    return df


def get_difficulty_metric(df: pd.DataFrame, offset: int):
    """
    Calculates the difficulty metric that is the mininmum distance 
    between a model's mus and the correct offset.
    """
    cols = get_real_columns(df)
    df = df[cols]
    row = []
    for col in cols:
        row.append(df[col].iloc[df[col].first_valid_index()])
    init_mus = pd.Series(row, index=cols)
    mu_offsets = (init_mus - offset).abs()
    return mu_offsets


def get_difficulty_to_time(loss_df, mus_df):
    drops = get_first_drop_step_df(loss_df)
    reg = re.compile('offset (\d+) \((\d+)\).*- loss')
    drop_per_run = {}
    keys = drops.keys()
    for k in keys:
        r = reg.match(k)
        if r is None:
            continue
        offset, run = map(int, r.groups())
        drop_per_run[run] = drops[k]
    
    mus = get_difficulty_metric(mus_df, offset)
    diff_per_run = {}
    for k in drop_per_run.keys():
        mu_cols = [f'offset {offset} ({k}) - mu_{i}' for i in [0, 1]]
        diff_per_run[k] = mus[mu_cols].min()
    
    d = []
    for run in diff_per_run.keys():
        d.append([run, drop_per_run[run], diff_per_run[run]])
    return d


def main_convolv():
    df = pd.read_csv('convolve-simple-sparse-offset-100-loss.csv')
    f = get_first_drop_step_df(df)
    m = get_instability_metric_df(df)
    print(f)
    print(m)


def difficulty_to_drop_viz():
    offsets = [10, 100, 200]
    rows = []
    for offset in offsets:
        df = pd.read_csv(f'offset_{offset}_loss.csv')
        mu_df = merge_mu_dfs(pd.read_csv(f'offset_{offset}_mu_0.csv'), pd.read_csv(f'offset_{offset}_mu_1.csv'))
        diffs = get_difficulty_to_time(df, mu_df)
        for run, drop, diff in diffs:
            rows.append([offset, run, drop, diff])
    df = pd.DataFrame(rows, columns=['offset', 'run', 'drop', 'diff'])
    breakpoint()

if __name__ == '__main__':
    difficulty_to_drop_viz()
    # main_convolv()
    df = pd.read_csv('offset_10_loss.csv')
    f = get_first_drop_step_df(df)
    m = get_instability_metric_df(df)
    print(f)
    print(m)

    mu0_df = pd.read_csv('offset_10_mu_0.csv')
    mu1_df = pd.read_csv('offset_10_mu_1.csv')
    mu_df = merge_mu_dfs(mu0_df, mu1_df)
    print(get_difficulty_to_time(df, mu_df))

