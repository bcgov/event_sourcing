import pandas as pd
import numpy as np
from pytest_check import check
import pytest


# Functions that allow to flag rows that have been
# removed, created or modified between an old and a new
# dataset. "removed", "created" and "modified" functions
# come together in the "track changes function."



def removed(old, new, key, date_col):
    """Detects and labels rows that have been removed.

    This function compares and old and a new dataset. If a 
    new dataset no longer contains unique identifier keys present in the old 
    dataset, it records the row as "removed" in the output with the removal date
    set to the new dataset snapshot date.

    Parameters
    ----------
    old : dataframe
        A dataset preceeding the new dataset.
    new : dataframe
        The newest dataset.
    key : str
        A unique identifier for each row. A "vin" in the case of the ICBC
        data.
    date_col : datetime
        Dataset snapshot date. In the ICBC datasets, typically referred to
        as "snapshot_date".

    Returns
    -------
    dataframe
        A dataframe containing rows removed from the old dataset (i.e. rows
        that appeared in the old dataset but not appearing in the new dataset).
        Column "change" will be set to "removed" for these rows.

    Examples
    --------
    >>> removed(snaps[dates[2]], snaps[dates[3]], 'vin', 'snapshot_date')
    """

    if type(old) is not pd.DataFrame:
        raise TypeError('Argument "old" must be a pandas data frame')

    if type(new) is not pd.DataFrame:
        raise TypeError('Argument "new" must be a pandas data frame')    

    if old.empty:
        raise ValueError('"old" dataframe is empty')
    
    if new.empty:
        raise ValueError('"new" dataframe is empty')
    
    if isinstance(key, (int, float, str)) == False:
        raise TypeError('"key" column name must be a string or a number')
    
    if isinstance(date_col, (int, float, str)) == False:
        raise TypeError('"date_col" column name must be a string or a number')
    
    if key not in old.columns or date_col not in old.columns:
        raise ValueError('Columns "key" and "date_col" must be in the "old" dataframe')
    
    if key not in new.columns or date_col not in new.columns:
        raise ValueError('Columns "key" and "date_col" must be in the "new" dataframe')

    removed_keys = set(old[key]).difference(set(new[key]))
    if len(removed_keys) > 0:
        removed_df = old.loc[old[key].isin(removed_keys)].copy()
        removed_df['change'] = 'removed'
        removed_df['change_date'] = new[date_col].unique()[0]
        return removed_df

    else:
        return pd.DataFrame()

def created(new, key, date_col, old=None):
    """Detects and labels rows that have been created (i.e. added new vins).

    This function compares and old and a new dataset. If a 
    new dataset new unique identifier keys relative to the old 
    dataset, it records the row as "created" in the output with 
    the creation date set to the new dataset snapshot date.

    Parameters
    ----------
    old : dataframe
        A dataset preceeding the new dataset.
    new : dataframe
        The newest dataset.
    key : str
        A unique identifier for each row. A "vin" in the case of the ICBC
        data.
    date_col : datetime
        Dataset snapshot date. In the ICBC datasets, typically referred to
        as "snapshot_date".

    Returns
    -------
    dataframe
        A dataframe containing rows that have been added in the new dataset. 
        Column "change" will be set to "created" for these rows. If the new
        dataset is the first one in the series (i.e. there is no old dataset)
        then all rows will be set to "created".

    Examples
    --------
    >>> created(snaps[dates[2]], snaps[dates[3]], 'vin', 'snapshot_date')
    """
    if old is not None:
        if type(old) is not pd.DataFrame:
            raise TypeError('Argument "old" must be a pandas data frame')

    if type(new) is not pd.DataFrame:
        raise TypeError('Argument "new" must be a pandas data frame')    
    
    if new.empty:
        raise ValueError('"new" dataframe is empty')
    
    if isinstance(key, (int, float, str)) == False:
        raise TypeError('"key" column name must be a string or a number')
    
    if isinstance(date_col, (int, float, str)) == False:
        raise TypeError('"date_col" column name must be a string or a number')
    
    if old is not None:
        if key not in old.columns or date_col not in old.columns:
            raise ValueError('Columns "key" and "date_col" must be in the "old" dataframe')
    
    if key not in new.columns or date_col not in new.columns:
        raise ValueError('Columns "key" and "date_col" must be in the "new" dataframe')

    if old is None:
        created_keys = set(new[key])
    else:
        created_keys = set(new[key]).difference(set(old[key]))
    if len(created_keys) > 0:
        created_df = new.loc[new[key].isin(created_keys)].copy()
        created_df['change'] = 'created'
        created_df['change_date'] = new[date_col].unique()[0]
        return created_df
    else:
        return pd.DataFrame()
    

def modified(old, new, key, date_col):
    """Detects and labels rows that have been modified.

    This function compares and old and a new dataset using the key (vin).
    If the rows for with the same key are different between the old and the new
    dataset, then the row will be marked as "modified".

    To do so the function equates new and old datasets and obtains a comparison matrix. 
    The matrix has the same columns and keys as the datasets being compared (old and new)
    If the values are different, the comparison returns False. 
    If both values are NA, it returns False. If the values are the same, the 
    comparison matrix returns True. Rows in the new dataset that contain 
    False in the matrix are labelled as "modified" in the "change" column.

    Parameters
    ----------
    old : dataframe
        A dataset preceeding the new dataset.
    new : dataframe
        The newest dataset.
    key : str
        A unique identifier for each row. A "vin" in the case of the ICBC
        data.
    date_col : datetime
        Dataset snapshot date. In the ICBC datasets, typically referred to
        as "snapshot_date".

    Returns
    -------
    dataframe
        A dataframe containing rows that have been modified.

    Examples
    --------
    >>> modified(snaps[dates[1]], snaps[dates[2]], 'vin', 'snapshot_date')
    """

    if type(old) is not pd.DataFrame:
        raise TypeError('Argument "old" must be a pandas data frame')

    if type(new) is not pd.DataFrame:
        raise TypeError('Argument "new" must be a pandas data frame')    

    if old.empty:
        raise ValueError('"old" dataframe is empty')
    
    if new.empty:
        raise ValueError('"new" dataframe is empty')
    
    if isinstance(key, (int, float, str)) == False:
        raise TypeError('"key" column name must be a string or a number')
    
    if isinstance(date_col, (int, float, str)) == False:
        raise TypeError('"date_col" column name must be a string or a number')
    
    if key not in old.columns or date_col not in old.columns:
        raise ValueError('Columns "key" and "date_col" must be in the "old" dataframe')
    
    if key not in new.columns or date_col not in new.columns:
        raise ValueError('Columns "key" and "date_col" must be in the "new" dataframe')

    # Find matching keys in the old and new dataset
    same_keys = set(old[key]).intersection(new[key])

    # To compare datasets, they need to be exactly the same
    old = old.loc[old[key].isin(same_keys)].sort_values(by=[key]).reset_index(drop=True)
    new = new.loc[new[key].isin(same_keys)].sort_values(by=[key]).reset_index(drop=True)

    comparison_matrix = (
        old.loc[old[key].isin(same_keys)] == 
        new.loc[new[key].isin(same_keys)]
        )

    # Set dates to true because it is expected that they would be different
    comparison_matrix[date_col] = True

    # If both cells are NA, it is also fine so it is set to True manually
    comparison_matrix[old.isna() & new.isna()] = True

    # Grab keys for rows that have a False in any of the columns in the comparison matrix
    modified_keys = set(new[comparison_matrix.eq(False).any(axis=1)][key])

    # Create column "change" and "change_date" and set "change" to "modified"

    if len(modified_keys) > 0:
        modified_df = new.loc[new[key].isin(modified_keys)].copy()
        modified_df['change'] = 'modified'
        modified_df['change_date'] = modified_df[date_col]
        return modified_df
    else:
        return pd.DataFrame()

def standardize_cols(new, old):
    """Detects added and removed columns.

    New columns are added to the output with NAs for older values.
    Removed column remain in the dataset with NAs for new datasets.

    Parameters
    ----------
    new : dataframe
        The newest dataset.
    old : dataframe
        A dataset preceeding the new dataset.

    Returns
    -------
    new
        A new dataframe with removed columns if applicable.
    old
        An old dataframe with added columns if applicable.

    Examples
    --------
    >>> standardize_cols(snaps[dates[3]], snaps[dates[2]])
    """
    if type(old) is not pd.DataFrame:
        raise TypeError('Argument "old" must be a pandas data frame')

    if type(new) is not pd.DataFrame:
        raise TypeError('Argument "new" must be a pandas data frame')    

    if old.empty:
        raise ValueError('"old" dataframe is empty')
    
    if new.empty:
        raise ValueError('"new" dataframe is empty')

    # Making sure that the original object is not modified
    old = old.copy() 
    new = new.copy()
    
    # Detect added and removed columns
    added_col = list(set(new.columns).difference(set(old.columns)))
    removed_col = list(set(old.columns).difference(set(new.columns)))
    removed_col.sort()

    # If a column is removed then add it to the new dataset and set all values there as NA
    if len(removed_col) > 0 and len(added_col) == 0:
        new[removed_col] = np.nan

    # If a new column is added then add it to the old dataset and set all values to NA
    elif len(added_col) > 0 and len(removed_col) == 0:
        old[added_col] = np.nan

    elif len(added_col) > 0 and len(removed_col) > 0:
        old[added_col] = np.nan
        new[removed_col] = np.nan

    # Sort columns to make sure old and new have identical headers in the same order
    new_cols = new.columns

    old = old[new_cols]
    new = new[new_cols]

    return {'new': new, 'old': old}

def incorrect_key_handler(new, key, duplicate_set = None, old = None):
    """Detects incorrect keys.

    Event sourcing pattern works if unique keys exist. If there are no unique keys, 
    it is impossible to track events.

    This function removes keys that are missing or duplicate.

    Parameters
    ----------
    old : dataframe
        A dataset preceeding the new dataset.
    new : dataframe
        The newest dataset.
    key : str
        A unique identifier for each row. A "vin" in the case of the ICBC
        data
    date_col : datetime
        Dataset snapshot date. In the ICBC datasets, typically referred to
        as "snapshot_date".
    duplicate_set: set
        A set of keys known to be duplicates in previous datasets.

    Returns
    -------
    new
        A new dataframe with missing and duplicate keys removed.
    old
        An old dataframe with missing and duplicate keys removed based on items 
        in the duplicate set.
    new_duplicate_set
        A new set of duplicate keys.

    Examples
    --------
    >>> incorrect_key_handler(snaps[dates[3]], 'vin', 'snapshot_date', duplicate_set = dup_set, old = snaps[dates[2]])
    """
    if old is not None:
        if type(old) is not pd.DataFrame:
            raise TypeError('Argument "old" must be a pandas data frame')

    if type(new) is not pd.DataFrame:
        raise TypeError('Argument "new" must be a pandas data frame')

    if duplicate_set is not None:
        if type(duplicate_set) != set:
            raise TypeError('Argument "duplicate_set" must be a set')  

    if old is not None:
        if old.empty:
            raise ValueError('"old" dataframe is empty')  
    
    if new.empty:
        raise ValueError('"new" dataframe is empty')
    
    if isinstance(key, (int, float, str)) == False:
        raise TypeError('"key" column name must be a string or a number')
    
    if key not in new.columns:
        raise ValueError('Column "key" must be in the "new" dataframe')
    
    if old is not None:
        if key not in old.columns:
            raise ValueError('Column "key" must be in the "old" dataframe')
        
    new = new.copy()

    new_duplicate_keys = set(
        new.loc[(new[key].isna() == False) & (new.duplicated(subset=[key], keep=False)), key].to_list())
    
    if old is not None:
        old = old.copy()
        old_duplicate_keys = set(
            old.loc[(old[key].isna() == False) & (old.duplicated(subset=[key], keep=False)), key].to_list())
        
    else:
        old_duplicate_keys = set()

    if duplicate_set is None:
        duplicate_set = set()

    duplicate_set = duplicate_set.union(new_duplicate_keys).union(old_duplicate_keys)

    new = new.loc[(new[key].isna() == False) & (new[key].isin(duplicate_set) == False)]

    if old is not None:
        old = old.loc[(old[key].isna() == False) & (old[key].isin(duplicate_set) == False)]

    if len(duplicate_set) == 0:
        duplicate_set = set()
     
    return {'new': new, 'old': old, 'new_duplicate_set': duplicate_set}

def col_reorder(df, first_cols, last_cols):

    """Reorders columns. Columns that are not included in the argument 
    are left as is int he middle of the dataframe.
    Parameters
    ----------
    df : dataframe
    first_cols : list
        List of columns that should appear at the beginning
    last_cols : list
        List of columns that should appear at the end
    """
    all_cols = df.columns
    col_order = set(all_cols)
    middle_cols = list(col_order - set(last_cols) - set(first_cols))
    middle_cols.sort()
    col_order = first_cols + middle_cols + last_cols
    return col_order

def track_changes(new, key, date_col, duplicate_set = None, old = None):

    """Detects and labels three types of events for each unique key.

    This function uses removed, created and modified functions to track
    changes between old and new datasets. In the end it outputs rows
    that experienced one of those three events. If the row didn't change
    it is not output.
    
    The function also outputs rows with missing or duplicate  keys
    into a separate dataset.
    
    Parameters
    ----------
    new : dataframe
        The newest dataset.
    key : str
        A unique identifier for each row. A "vin" in the case of the ICBC
        data.
    date_col : datetime
        Dataset snapshot date. In the ICBC datasets, typically referred to
        as "snapshot_date".
    old : dataframe
        A dataset preceeding the new dataset.
    duplicate_set: set
        A set of keys known to be duplicates in previous datasets.

    Returns
    -------
    changes_df
        A dataframe that contains rows with unique keys and corresponding events.
    new_duplicate_set
        A new set of duplicate keys.

    Examples
    --------
    >>> track_changes(snaps[dates[4]], 'vin', 'snapshot_date', old = snaps[dates[3]])[0]
    """
    new = new.copy()
    
    handler_output = incorrect_key_handler(new, key, duplicate_set=duplicate_set, old=old)
    new_no_dups = handler_output['new']
    old_no_dups = handler_output['old']
    new_duplicate_set = handler_output['new_duplicate_set']

    new_with_dups = new.loc[new[key].isin(new_duplicate_set)].copy()
    new_with_na = new.loc[new[key].isna()].copy()

    new_with_dups['change'] = 'untracked_duplicate_key'
    new_with_na['change'] = 'untracked_missing_key'

    incorrect_keys_df = pd.concat([new_with_dups, new_with_na])
    incorrect_keys_df['change_date'] = incorrect_keys_df[date_col]

    if (new_no_dups.empty == True) & (incorrect_keys_df.empty == False):
        changes_df = incorrect_keys_df

    else:
        # First scenario is where the new dataset is the first in the series, old dataset does not exist

        if old_no_dups is None or old_no_dups.empty == True:
            changes_df = created(new_no_dups, key, date_col).reset_index(drop=True)
            col_order = col_reorder(changes_df, [key], ['change_date', 'change'])
            changes_df = changes_df[col_order]
            changes_df = pd.concat([changes_df, incorrect_keys_df]).reset_index(drop=True)

        # Second scenario is where the new datasets is not the first in the series
        else:
            cols_output = standardize_cols(new_no_dups, old_no_dups)
            old_standardized, new_standardized = cols_output['old'], cols_output['new']

            # Now that the old and the new have identical columns, no duplicate or missing keys,
            # these datasets will be compared to detected removed, created and modified rows.
            removed_df = removed(old_standardized, new_standardized, key, date_col)
            created_df = created(new_standardized, key, date_col, old = old_standardized)
            modified_df = modified(old_standardized, new_standardized, key, date_col)

            if removed_df.empty == True and created_df.empty == True and modified_df.empty == True:
                changes_df = pd.DataFrame()
                changes_df = pd.concat([changes_df, incorrect_keys_df]).reset_index(drop=True)
                

            else:
            # Combine all events (removed, created and modified) in a single dataframe
                changes_df = pd.concat([removed_df, created_df, modified_df], ignore_index=True).sort_values(by=key).reset_index(drop=True)
                col_order = col_reorder(changes_df, [key], ['change_date', 'change'])
                changes_df = changes_df[col_order]
                changes_df = pd.concat([changes_df, incorrect_keys_df]).reset_index(drop=True)


    return {'changes_df': changes_df, 'new_duplicate_set': new_duplicate_set}

# Helper functions for formatting data

def format_case(s, case = 'skip', ignore_list = []):
    if len(s.dropna()) != 0:
        output = (
            s[s.notna()] # I am applying this function to non NaN values only. If you do not, they get converted from NaN to nan and are more annoying to work with.
            .astype(str) # Convert to string
            .str.strip() # Strip white spaces (this dataset suffers from extra tabs, lines, etc.)
            )
        
        if case == 'title':
            return output.str.title()
        elif case == 'upper':
            return output.str.upper()
        elif case == 'lower':
            return output.str.lower()
        elif case == 'skip':
            pass

def format_numbers(s):
    if len(s.dropna()) != 0:
        output = pd.to_numeric(
            s[s.notna()] # I am applying this function to non NaN values only. If you do not, they get converted from NaN to nan and are more annoying to work with.
            .astype(str) # Convert to string
            .str.replace(',', '') # Strip white spaces (this dataset suffers from extra tabs, lines, etc.)
            )
        return output

def preprocess_raw_file(file_path, delimiter, file_dtype_dict=None, nrows='all', dropped_columns=None, dtype_numeric_list=None, dtype_date_list=None, text_modif_dict=None):

    if nrows == 'all':
        df = pd.read_csv(
        file_path, 
        delimiter=delimiter,
        na_values=['NIL', 'Unknown', 'unknown', 'UNKNOWN'], dtype=file_dtype_dict)

    else:
        df = pd.read_csv(
            file_path, 
            delimiter=delimiter,
            nrows=nrows, na_values=['NIL', 'Unknown', 'unknown', 'UNKNOWN'], dtype=file_dtype_dict)

        
    df.columns = df.columns.str.lower()
    df.drop(columns=dropped_columns, inplace=True)
    
    numeric_cols = list(set(dtype_numeric_list).intersection(set(df.columns)))

    numeric_cols_w_strings = df[numeric_cols].select_dtypes('object').columns
    for col in numeric_cols_w_strings:
        df[col] = format_numbers(df[col])

    date_cols = list(set(dtype_date_list).intersection(set(df.columns)))
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], yearfirst=True, utc=True).dt.date

    for key in text_modif_dict:
        col_subset = list(set(text_modif_dict[key]).intersection(df.columns))
        if len(col_subset) != 0:
            for col in col_subset:
                df[col] = format_case(df[col], case=key)

    return df

# Tests

def test_removed():

    test_dfs = {}
    col_order = ['date', 'key', 'colour']

    removed_key = {}
    removed_key['one'] = '30b'
    removed_key['three'] = ['10a', '30b', 20]

    test_dfs['base'] = (
        pd.DataFrame({
            'key': ['10a', 20, '30b', '40c', '40c', '50d', np.nan],
            'colour': ['red', 'yellow', 'red', 'green', 'blue', 'yellow', 'yellow']
        })
    )

    test_dfs['base']['date'] = '01-01-2025'
    test_dfs['base'] = test_dfs['base'][col_order]

    new_dfs_names = ['removed_one', 'removed_three', 'removed_na', 'removed_nothing', 'added_one']

    test_dfs['removed_one'] = test_dfs['base'].loc[test_dfs['base']['key'] != removed_key['one']].copy()
    test_dfs['removed_three'] = test_dfs['base'].loc[~test_dfs['base']['key'].isin(removed_key['three'])].copy()
    test_dfs['removed_na'] = test_dfs['base'].loc[test_dfs['base']['key'].isna() == False].copy()
    test_dfs['removed_nothing'] = test_dfs['base'].copy()
    test_dfs['added_one'] = pd.concat(
        [test_dfs['base'],
        pd.DataFrame({'date': ['01-02-2025'], 'key': ['60'], 'colour': ['purple']}).reset_index(
            drop=True)])

    for dfname in new_dfs_names:
        test_dfs[dfname]['date'] = '01-02-2025'

    expected = {}
    expected['removed_one'] = test_dfs['base'].loc[(test_dfs['base']['key'] == removed_key['one'])].copy()
    expected['removed_three'] = test_dfs['base'].loc[test_dfs['base']['key'].isin(removed_key['three'])].copy()
    expected['removed_na'] = test_dfs['base'].loc[test_dfs['base']['key'].isna()].copy()
    expected['removed_nothing'] = pd.DataFrame({'date': [], 'key': [], 'colour': []})

    for dfname in expected:
        expected[dfname]['change'] = 'removed'
        expected[dfname]['change_date'] = '01-02-2025'

    actual = {}

    for dfname in new_dfs_names:
        actual[dfname] = removed(
                test_dfs['base'], 
                test_dfs[dfname], 
                'key', 'date')
    
    with check:
        assert actual['removed_one'].equals(expected['removed_one']) == True, "Dataframe output with one row removed is incorrect."

    with check:
        assert actual['removed_three'].equals(expected['removed_three']) == True, "Dataframe output with three rows removed is incorrect."

    with check:
        assert actual['removed_na'].equals(expected['removed_na']) == True, "Dataframe output with missing key row removed is incorrect."

    with check:
        assert actual['removed_na'].equals(expected['removed_na']) == True, "Dataframe output with missing key row removed is incorrect."
    
    with check:
        assert actual['removed_nothing'].empty == True, "Dataframe output with nothing removed is incorrect."

    with check:
        assert actual['added_one'].empty == True, "Dataframe output with one row added and nothing removed is incorrect."

def test_created():
    test_dfs = {}
    col_order = ['date', 'key', 'colour']

    added_row = {}
    added_row['added_one'] = pd.DataFrame({'date': ['01-02-2025'], 'key': ['60d'], 'colour': ['orange']})
    added_row['added_three'] = pd.DataFrame({
        'key': ['60d', '70f', '80e'],
        'colour': ['gray', 'green', 'yellow']
    })
    added_row['added_na'] = pd.DataFrame(
        {'date': '01-02-2025', 'key': np.nan, 'colour': 'lime'},
        index=[0]
    )

    added_row['added_three']['date'] = '01-02-2025'
    added_row['added_three'] = added_row['added_three'][col_order]

    test_dfs['base'] = (
        pd.DataFrame({
            'key': ['10a', 20, '30b', '40c', '40c', '50d'],
            'colour': ['red', 'yellow', 'red', 'green', 'blue', 'yellow']
        })
    )

    test_dfs['base']['date'] = '01-01-2025'
    test_dfs['base'] = test_dfs['base'][col_order]

    new_dfs_names = ['added_one', 'added_three', 'added_na']

    for name in new_dfs_names:
        test_dfs[name] = pd.concat(
            [
                test_dfs['base'],
                added_row[name]
            ]
        )

        test_dfs[name]['date'] = '01-02-2025'
        test_dfs[name] = test_dfs[name][col_order]

    test_dfs['removed_one'] = test_dfs['base'][test_dfs['base']['key'] != '30b'].copy()

    expected = {}
    for name in new_dfs_names:
        expected[name] = added_row[name].copy()
        expected[name]['change'] = 'created'
        expected[name]['change_date'] = expected[name]['date']


    expected['no_old'] = test_dfs['base'].copy()
    expected['no_old']['change'] = 'created'
    expected['no_old']['change_date'] = expected['no_old']['date']
    expected['removed_one'] = None

    actual = {}

    for name in new_dfs_names:
        actual[name] = created(test_dfs[name], 'key', 'date', old=test_dfs['base'])

    actual['added_na']['key'] = np.float64(actual['added_na']['key'][0])

    actual['no_old'] = created(test_dfs['base'], 'key', 'date')

    actual['created_nothing'] = created(test_dfs['base'], 'key', 'date', old=test_dfs['base'])
    actual['removed_one'] = created(test_dfs['removed_one'], 'key', 'date', old=test_dfs['base'])

    with check:
        assert actual['added_one'].equals(expected['added_one']) == True, "Dataframe output with one row added is incorrect."

    with check:
        assert actual['added_three'].equals(expected['added_three']) == True, "Dataframe output with three rows added is incorrect."

    with check:
        assert actual['added_na'].equals(expected['added_na']) == True, "Dataframe output with NA row added (when no NAs are present in the old dataset) is incorrect."

    with check:
        assert actual['no_old'].equals(expected['no_old']) == True, 'Dataframe output with no "old" dataframe  is incorrect.'

    with check:
        assert actual['created_nothing'].empty == True, 'Dataframe output with nothing added is incorrect.'

    with check:
        assert actual['removed_one'].empty == True, 'Dataframe output with one row removed and nothing added is incorrect.'


def test_removed_error():
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [10, 100, 100]})
    df_new = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 100, 100]})
    empty = pd.DataFrame({'key': []})
    with check:
        with pytest.raises(TypeError, match=r'must be a pandas data frame'):
            removed(1, df, 'a', 'b')

    with check:
        with pytest.raises(TypeError, match=r'must be a pandas data frame'):
            removed(df, 1, 'a', 'b')

    with check:
        with pytest.raises(TypeError, match=r'"key" column name must be a string or a number'):
            removed(df, df, [1], 'col1')

    with check:
        with pytest.raises(TypeError, match=r'"date_col" column name must be a string or a number'):
            removed(df, df, 'col1', [1])

    with check:
        with pytest.raises(ValueError, match=r'dataframe is empty'):
            removed(empty, df, 'a', 'b')

    with check:
        with pytest.raises(ValueError, match=r'dataframe is empty'):
            removed(df, empty, 'a', 'b')

    with check:
        with pytest.raises(ValueError, match=r'Columns "key" and "date_col" must be in the "old" dataframe'):
            removed(df, df, 'a', 'b')

    with check:
        with pytest.raises(ValueError, match=r'Columns "key" and "date_col" must be in the "new" dataframe'):
            removed(df, df_new, 'col1', 'col2')


def test_created_error():
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [10, 100, 100]})
    df_new = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 100, 100]})
    empty = pd.DataFrame({'key': []})
    with check:
        with pytest.raises(TypeError, match=r'must be a pandas data frame'):
            created(1, 'a', 'b', old=df)

    with check:
        with pytest.raises(TypeError, match=r'must be a pandas data frame'):
            created(df, 'a', 'b', old=1)

    with check:
        with pytest.raises(TypeError, match=r'"key" column name must be a string or a number'):
            created(df, [1], 'col1')

    with check:
        with pytest.raises(TypeError, match=r'"date_col" column name must be a string or a number'):
            created(df, 'col1', [1])

    with check:
        with pytest.raises(TypeError, match=r'"key" column name must be a string or a number'):
            created(df, [1], 'col1')

    with check:
        with pytest.raises(TypeError, match=r'"date_col" column name must be a string or a number'):
            created(df, 'col1', [1], old=df)
    
    with check:
        with pytest.raises(TypeError, match=r'"date_col" column name must be a string or a number'):
            created(df, 'col1', [1], old=empty)

    with check:
        with pytest.raises(ValueError, match=r'dataframe is empty'):
            created(empty, 'a', 'b', old=df)

    with check:
        with pytest.raises(ValueError, match=r'Columns "key" and "date_col" must be in the "new" dataframe'):
            created(df_new, 'col1', 'col2', old=df)

    with check:
        with pytest.raises(ValueError, match=r'Columns "key" and "date_col" must be in the "old" dataframe'):
            created(df_new, 'a', 'b',old=df)


def test_modified():

    test_dfs = {}
    col_order = ['date', 'key', 'colour', 'size']

    new_dfs_names = ['nan_to_value', 'value_to_nan', 'one_change', 'two_changes_one_row', 'three_changes']

    test_dfs['base'] = (
        pd.DataFrame({
            'key': ['10a', '20b', '30b', '40c', '40c', '50d'],
            'colour': ['red', 'yellow', 'red', 'green', 'blue', 'yellow'],
            'size': ['medium', np.nan, 'small', 'large', 'medium', 'small']
        })
    )

    test_dfs['base']['date'] = '01-01-2025'
    test_dfs['base'] = test_dfs['base'][col_order]

    for name in new_dfs_names:
        test_dfs[name] = test_dfs['base'].copy()
        test_dfs[name]['date'] = '01-02-2025'
        
    test_dfs['nan_to_value'].loc[test_dfs['nan_to_value']['key'] == '20b', 'size'] = 'xlarge'
    test_dfs['value_to_nan'].loc[test_dfs['nan_to_value']['key'] == '10a', 'colour'] = np.nan
    test_dfs['one_change'].loc[test_dfs['one_change']['key'] == '30b', 'colour'] = 'lime'
    test_dfs['two_changes_one_row'].loc[test_dfs['two_changes_one_row']['key'] == '30b', 'colour'] = 'lime'
    test_dfs['two_changes_one_row'].loc[test_dfs['two_changes_one_row']['key'] == '30b', 'size'] = 'xxsmall'
    test_dfs['three_changes'] = test_dfs['two_changes_one_row'].copy()
    test_dfs['three_changes'].loc[test_dfs['two_changes_one_row']['key'] == '50d', 'size'] = 'xxlarge'

    expected = {}
    expected['nan_to_value'] = test_dfs['nan_to_value'].loc[test_dfs['nan_to_value']['key'] == '20b'].copy()
    expected['value_to_nan'] = test_dfs['value_to_nan'].loc[test_dfs['nan_to_value']['key'] == '10a'].copy()
    expected['one_change'] = test_dfs['one_change'].loc[test_dfs['one_change']['key'] == '30b'].copy()
    expected['two_changes_one_row'] = test_dfs['two_changes_one_row'].loc[test_dfs['two_changes_one_row']['key'] == '30b'].copy()
    expected['three_changes'] = test_dfs['three_changes'].loc[test_dfs['three_changes']['key'].isin(['30b', '50d'])].copy()

    for name in new_dfs_names:
        expected[name]['change'] = 'modified'
        expected[name]['change_date'] = '01-02-2025'

    actual = {}
    actual['no_change'] = modified(test_dfs['base'], test_dfs['base'], 'key', 'date')

    for name in new_dfs_names:
        actual[name] = modified(test_dfs['base'], test_dfs[name], 'key', 'date')

    with check:
        assert actual['no_change'].empty == True, 'Dataframe output with nothing changed is incorrect.'

    with check:
        assert actual['nan_to_value'].equals(expected['nan_to_value']) == True, 'Dataframe output with NA changed to a value is incorrect.'

    with check:
        assert actual['value_to_nan'].equals(expected['value_to_nan']) == True, 'Dataframe output with value changed to NA is incorrect.'

    with check:
        assert actual['one_change'].equals(expected['one_change']) == True, 'Dataframe output with one value changed to another value is incorrect.'

    with check:
        assert actual['two_changes_one_row'].equals(expected['two_changes_one_row']) == True, 'Dataframe output with two values changed in one row is incorrect.'

    with check:
        assert actual['three_changes'].equals(expected['three_changes']) == True, 'Dataframe output with three values changed is incorrect.'


def test_modified_error():
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [10, 100, 100]})
    df_new = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 100, 100]})
    empty = pd.DataFrame({'key': []})
    with check:
        with pytest.raises(TypeError, match=r'must be a pandas data frame'):
            modified(1, df, 'a', 'b')

    with check:
        with pytest.raises(TypeError, match=r'must be a pandas data frame'):
            modified(df, 1, 'a', 'b')

    with check:
        with pytest.raises(TypeError, match=r'"key" column name must be a string or a number'):
            modified(df, df, [1], 'col1')

    with check:
        with pytest.raises(TypeError, match=r'"date_col" column name must be a string or a number'):
            modified(df, df, 'col1', [1])

    with check:
        with pytest.raises(ValueError, match=r'dataframe is empty'):
            modified(empty, df, 'a', 'b')

    with check:
        with pytest.raises(ValueError, match=r'dataframe is empty'):
            modified(df, empty, 'a', 'b')

    with check:
        with pytest.raises(ValueError, match=r'Columns "key" and "date_col" must be in the "old" dataframe'):
            modified(df, df, 'a', 'b')

    with check:
        with pytest.raises(ValueError, match=r'Columns "key" and "date_col" must be in the "new" dataframe'):
            modified(df, df_new, 'col1', 'col2')

def test_standardize_cols():
    test_dfs = {}
    test_dfs['base'] = pd.DataFrame({'A': [1, 2, 1, 3], 'B': ['c', 'c', 'd', 'c'], 'C': [True, False, False, False]})


    test_dfs['one_col'] = test_dfs['base'].copy()
    test_dfs['one_col']['D'] = [999, 1000, 1111, 222]

    test_dfs['two_cols'] = test_dfs['base'].copy()
    test_dfs['two_cols']['X'] = [0, 1, 1, 0]
    test_dfs['two_cols']['Y'] = ['x', 'x', 'y', 'z']

    test_dfs['three_cols'] = test_dfs['one_col'].copy()
    test_dfs['three_cols']['E'] = [1.5, 1.2, 1, 10]
    test_dfs['three_cols']['F'] = ['aaa', 'bbb', 0, 'bbb']

    expected = {}
    expected['one_col'] = {}
    expected['one_col']['added'] = test_dfs['one_col'].copy()
    expected['one_col']['original'] = test_dfs['base'].copy()
    expected['one_col']['original']['D'] = np.nan

    col_order = ['A', 'B', 'C', 'X', 'Y', 'D', 'E', 'F']
    expected['added_removed'] = {}
    expected['added_removed']['old'] = test_dfs['three_cols'].copy()
    expected['added_removed']['old'][['X', 'Y']] = np.nan
    expected['added_removed']['old'] = expected['added_removed']['old'][col_order]
    expected['added_removed']['new'] = test_dfs['two_cols'].copy()
    expected['added_removed']['new'][['D', 'E', 'F']] = np.nan

    actual = {}
    actual['one_col'] = {}
    actual['one_col']['new_col'] = standardize_cols(test_dfs['one_col'], test_dfs['base'])
    actual['one_col']['removed_col'] = standardize_cols(test_dfs['base'], test_dfs['one_col'])
    actual['added_removed'] = standardize_cols(test_dfs['two_cols'], test_dfs['three_cols'])
    actual['no_change'] = standardize_cols(test_dfs['base'], test_dfs['base'])    
    
    with check:
        assert actual['one_col']['new_col']['old'].equals(expected['one_col']['original']) == True, '"old" dataframe output with a new column in "new" is incorrect'

    with check:
        assert actual['one_col']['new_col']['new'].equals(expected['one_col']['added']) == True, '"new" dataframe output with a new column in "new" is incorrect'

    with check:
        assert actual['one_col']['removed_col']['old'].equals(expected['one_col']['added']) == True, '"old" dataframe output with a removed column in "new" is incorrect'

    with check:
        assert actual['one_col']['removed_col']['new'].equals(expected['one_col']['original']) == True, '"new" dataframe output with a removed column in "new" is incorrect'

    with check:
        assert actual['added_removed']['old'].equals(expected['added_removed']['old']) == True, '"old" dataframe output with two removed columns and three added columns is incorrect'

    with check:
        assert actual['added_removed']['new'].equals(expected['added_removed']['new']) == True, '"new" dataframe output with two removed columns and three added columns is incorrect'

    with check:
        assert actual['no_change']['old'].equals(test_dfs['base']) == True, '"old" dataframe output no column differences is incorrect'

    with check:
        assert actual['no_change']['new'].equals(test_dfs['base']) == True, '"new" dataframe output no column differences is incorrect'


def test_standardized_cols_error():
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [10, 100, 100]})
    empty = pd.DataFrame({'key': []})
    with check:
        with pytest.raises(TypeError, match=r'must be a pandas data frame'):
            standardize_cols(1, df)

    with check:
        with pytest.raises(TypeError, match=r'must be a pandas data frame'):
            standardize_cols(df, [2])

    with check:
        with pytest.raises(ValueError, match=r'dataframe is empty'):
            standardize_cols(empty, df)

    with check:
        with pytest.raises(ValueError, match=r'dataframe is empty'):
            standardize_cols(df, empty)

def error_msg(s, opt=''):
    if len(opt) != 0:
        opt = ' (' + opt + ')'
    return s + ' -- is incorrect' + opt

def test_incorrect_key_handler():
    test_dfs = {}
    test_dfs['no_duplicates'] = {}
    test_dfs['no_duplicates'][0] = pd.DataFrame({'key': ['1a', '2b', '3c'], 'colour': ['orange', 'yellow', 'green']})
    test_dfs['no_duplicates'][1] = pd.DataFrame({'key': ['3c', '4d', '5e'], 'colour': ['pink', 'purple', 'green']})
    test_dfs['no_duplicates'][2] = pd.DataFrame({'key': ['3c', np.nan, '5e', np.nan], 'colour': ['pink', 'purple', 'green', 'yellow']})

    test_dfs['duplicates'] = {}
    test_dfs['duplicates'][0] = pd.DataFrame(
        {'key': ['1a', 'yy', 'yy', 'ww', 'ww', 'ww', 'f6'], 
        'colour': ['red', 'green', 'blue', 'purple', 'gray', 'gray', 'pink']})

    test_dfs['duplicates'][1] = pd.DataFrame(
        {'key': ['1a', '2b', '2b', 'yy', 'yy', 'h7'], 
        'colour': ['red', 'green', 'blue', 'purple', 'gray', 'white']})

    test_dfs['duplicates'][2] = pd.DataFrame(
        {'key': ['1a', '2b', '2b', 'yy', 'yy', np.nan], 
        'colour': ['red', 'green', 'blue', 'purple', 'gray', 'white']})

    test_dfs['duplicates'][3] = pd.DataFrame(
        {'key': ['1a', 'yy', 'yy', np.nan, np.nan, 'f6'], 
        'colour': ['red', 'green', 'blue', 'purple', 'gray', 'gray']})

    # case_1: No duplicates, no old, no set
    expected = {}
    expected['case_1'] = {}
    expected['case_1']['text'] = 'case_1: No duplicates, no old, no set'
    expected['case_1']['old'] = None
    expected['case_1']['new_duplicate_set'] = set()
    expected['case_1']['new'] = test_dfs['no_duplicates'][0]

    actual = {}
    actual['case_1'] = incorrect_key_handler(test_dfs['no_duplicates'][0], 'key')

    # case_2: No duplicates, no set
    expected['case_2'] = {}
    expected['case_2']['text'] = 'case_2: No duplicates, no set'
    expected['case_2']['old'] = test_dfs['no_duplicates'][1]
    expected['case_2']['new_duplicate_set'] = set()
    expected['case_2']['new'] = test_dfs['no_duplicates'][0]

    actual['case_2'] = incorrect_key_handler(test_dfs['no_duplicates'][0], 'key', old=test_dfs['no_duplicates'][1])

    # case_3: No duplicates, with set
    dup_set = {}
    dup_set['case_3'] = set(['1a'])

    expected['case_3'] = {}
    expected['case_3']['text'] = 'case_3: No duplicates, with set'
    expected['case_3']['old'] = test_dfs['no_duplicates'][1].loc[~test_dfs['no_duplicates'][1]['key'].isin(dup_set['case_3'])]
    expected['case_3']['new'] = test_dfs['no_duplicates'][0].loc[~test_dfs['no_duplicates'][0]['key'].isin(dup_set['case_3'])]
    expected['case_3']['new_duplicate_set'] = dup_set['case_3']

    actual['case_3'] = incorrect_key_handler(
        test_dfs['no_duplicates'][0], 'key', 
        old=test_dfs['no_duplicates'][1], duplicate_set=dup_set['case_3'])

    # case_4: Duplicates in new, no set, no old
    expected['case_4'] = {}
    expected['case_4']['text'] = 'case_3: Duplicates in new, no set'
    expected['case_4']['old'] = None
    expected['case_4']['new'] = test_dfs['duplicates'][0].drop_duplicates(subset=['key'], keep=False)
    expected['case_4']['new_duplicate_set'] = set(
        test_dfs['duplicates'][0].loc[test_dfs['duplicates'][0].duplicated(subset=['key'], keep=False), 'key'].to_list())

    actual['case_4'] = incorrect_key_handler(test_dfs['duplicates'][0], 'key')

    # case_5: Duplicates in old, new, no set
    expected['case_5'] = {}
    expected['case_5']['text'] = 'Duplicates in old, new, no set'
    expected['case_5']['old'] = test_dfs['duplicates'][1].drop_duplicates(subset=['key'], keep=False)
    expected['case_5']['new'] = test_dfs['duplicates'][0].drop_duplicates(subset=['key'], keep=False)
    expected['case_5']['new_duplicate_set'] = set(
        test_dfs['duplicates'][0].loc[test_dfs['duplicates'][0].duplicated(subset=['key'], keep=False), 'key'].to_list()).union(
            set(test_dfs['duplicates'][1].loc[test_dfs['duplicates'][1].duplicated(subset=['key'], keep=False), 'key'].to_list())
            )

    actual['case_5'] = incorrect_key_handler(test_dfs['duplicates'][0], 'key', old=test_dfs['duplicates'][1])

    # case_6: Duplicates in old, new, set that matches old and new
    dup_set['case_6'] = set(['f6', 'h7'])
    expected['case_6'] = {}
    expected['case_6']['text'] = 'Duplicates in old, new, set that matches old and new'
    expected['case_6']['old'] = expected['case_5']['old'].loc[~expected['case_5']['old']['key'].isin(dup_set['case_6'])]
    expected['case_6']['new'] = expected['case_5']['new'].loc[~expected['case_5']['new']['key'].isin(dup_set['case_6'])]
    expected['case_6']['new_duplicate_set'] = expected['case_5']['new_duplicate_set'].union(dup_set['case_6'])


    actual['case_6'] = incorrect_key_handler(
        test_dfs['duplicates'][0], 'key', 
        old=test_dfs['duplicates'][1], duplicate_set=dup_set['case_6'])

    # case_7. A value is duplicated in old, and the same value appears in new once, no set
    expected['case_7'] = {}
    expected['case_7']['text'] = 'A value is duplicated in old, and the same value appears in new once, no set'
    expected['case_7']['old'] = test_dfs['duplicates'][1].drop_duplicates(subset=['key'], keep=False)
    expected['case_7']['new_duplicate_set'] = set(test_dfs['duplicates'][1].loc[test_dfs['duplicates'][1].duplicated(subset=['key'], keep=False), 'key'].to_list())
    expected['case_7']['new'] = test_dfs['no_duplicates'][0].loc[~test_dfs['no_duplicates'][0]['key'].isin(expected['case_7']['new_duplicate_set'])]

    actual['case_7'] = incorrect_key_handler(
        test_dfs['no_duplicates'][0], 'key', 
        old=test_dfs['duplicates'][1])

    # case_8. Reverse of case 7
    expected['case_8'] = {}
    expected['case_8']['text'] = 'A value is duplicated in new, and the same value appears in old once, no set'
    expected['case_8']['old'] = expected['case_7']['new'].copy()
    expected['case_8']['new'] = expected['case_7']['old'].copy()
    expected['case_8']['new_duplicate_set'] = expected['case_7']['new_duplicate_set']

    actual['case_8'] = incorrect_key_handler(
        test_dfs['duplicates'][1], 'key', 
        old=test_dfs['no_duplicates'][0])

    # case_9. Same as case 8 but with a duplicate set that filters out the same value in both, plus one more in old
    dup_set['case_9'] = set(['1a', 'h7'])
    expected['case_9'] = {}
    expected['case_9']['text'] = 'A value is duplicated in new, and the same value appears in old once, with a duplicate set that filters out the same value in both, plus one more in old'
    expected['case_9']['new'] = expected['case_7']['new'].loc[~expected['case_7']['new']['key'].isin(dup_set['case_9'])]
    expected['case_9']['old'] = expected['case_7']['old'].loc[~expected['case_7']['old']['key'].isin(dup_set['case_9'])]
    expected['case_9']['new_duplicate_set'] = expected['case_7']['new_duplicate_set'].union(dup_set['case_9'])

    actual['case_9'] = incorrect_key_handler(
        test_dfs['no_duplicates'][0], 'key', 
        old=test_dfs['duplicates'][1], duplicate_set=dup_set['case_9'])

    # case_10. Duplicates in old, no duplicates in new, missing values in both, no set
    expected['case_10'] = {}
    expected['case_10']['text'] = 'Duplicates in old, no duplicates in new, missing values in both, no set'
    expected['case_10']['old'] = test_dfs['duplicates'][2].loc[test_dfs['duplicates'][2]['key'].isna() == False].drop_duplicates(subset=['key'], keep=False)
    expected['case_10']['new'] = test_dfs['no_duplicates'][2].loc[test_dfs['no_duplicates'][2]['key'].isna() == False]
    expected['case_10']['new_duplicate_set'] = set(
        test_dfs['duplicates'][2].dropna(subset=['key']).loc[test_dfs['duplicates'][2].duplicated(subset=['key'], keep=False), 'key'].to_list())

    actual['case_10'] = incorrect_key_handler(test_dfs['no_duplicates'][2], 'key', old=test_dfs['duplicates'][2])

    # case_11: Duplicates and missing in both, no set.
    expected['case_11'] = {}
    expected['case_11']['text'] = 'Duplicates and missing in both, no set'
    expected['case_11']['new'] = test_dfs['duplicates'][3].loc[test_dfs['duplicates'][3]['key'].isna() == False].drop_duplicates(subset=['key'], keep=False)
    expected['case_11']['old'] = test_dfs['duplicates'][2].loc[test_dfs['duplicates'][2]['key'].isna() == False].drop_duplicates(subset=['key'], keep=False)
    expected['case_11']['new_duplicate_set'] = set(
        test_dfs['duplicates'][2].dropna(subset=['key']).loc[test_dfs['duplicates'][2].duplicated(subset=['key'], keep=False), 'key'].to_list()).union(
            set(test_dfs['duplicates'][3].dropna(subset=['key']).loc[test_dfs['duplicates'][3].duplicated(subset=['key'], keep=False), 'key'].to_list())
            )

    actual['case_11'] = incorrect_key_handler(test_dfs['duplicates'][3], 'key', old=test_dfs['duplicates'][2])

    # case_12. Same as case_11, but with a duplicate set that filters out one key in old and new
    dup_set['case_12'] = set(['1a', '999'])
    expected['case_12'] = {}
    expected['case_12']['text'] = 'Duplicates and missing in both, with a duplicate set that filters out one key in old and new '
    expected['case_12']['old'] = expected['case_11']['old'].loc[~expected['case_11']['old']['key'].isin(dup_set['case_12'])]
    expected['case_12']['new'] = expected['case_11']['new'].loc[~expected['case_11']['new']['key'].isin(dup_set['case_12'])]
    expected['case_12']['new_duplicate_set'] = expected['case_11']['new_duplicate_set'].union(dup_set['case_12'])

    actual['case_12'] = incorrect_key_handler(test_dfs['duplicates'][3], 'key', old=test_dfs['duplicates'][2], duplicate_set=dup_set['case_12'])

    for case in expected:
        with check:
            assert actual[case]['new_duplicate_set'] == expected[case]['new_duplicate_set'], error_msg(expected[case]['text'], 'new_duplicate_set')

        for df_type in ['old', 'new']:
            if expected[case][df_type] is None:
                with check:
                    assert actual[case][df_type] == None, error_msg(expected[case]['text'], df_type)
            else:
                with check:
                    assert actual[case][df_type].equals(expected[case][df_type]) == True, error_msg(expected[case]['text'], df_type)

def test_incorrect_key_handler_error():
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [10, 100, 100]})
    df_new = pd.DataFrame({'col1': [10, 20, 30], 'col3': [90, 91, 92]})

    empty = pd.DataFrame({'key': []})
    with check:
        with pytest.raises(TypeError, match=r'must be a pandas data frame'):
            incorrect_key_handler(1, 'col1')

    with check:
        with pytest.raises(TypeError, match=r'must be a pandas data frame'):
            incorrect_key_handler(df, 'col1', old=1)

    with check:
        with pytest.raises(ValueError, match=r'dataframe is empty'):
            incorrect_key_handler(empty, 'col1', old=df)

    with check:
        with pytest.raises(ValueError, match=r'dataframe is empty'):
            incorrect_key_handler(df, 'col1', old=empty)

    with check:
        with pytest.raises(TypeError, match=r'"key" column name must be a string or a number'):
            incorrect_key_handler(df_new, [0], old=df)

    with check:
        with pytest.raises(ValueError, match=r'Column "key" must be in the "new" dataframe'):
            incorrect_key_handler(df_new, 'col2', old=df)

    with check:
        with pytest.raises(ValueError, match=r'Column "key" must be in the "old" dataframe'):
            incorrect_key_handler(df_new, 'col3', old=df)

    with check:
        with pytest.raises(TypeError, match=r'Argument "duplicate_set" must be a set'):
            incorrect_key_handler(df_new, 'col1', old=df, duplicate_set=[1])


def test_track_changes():
    test_dfs = {}

    test_dfs[0] = pd.DataFrame(
        {'colour': ['purple', 'green', 'red', 'green'], 'size': [2, 4, 8, 12], 'product_key': ['a1', 'b2', 'c3', 'd4'], 'date': '01-01-2020'})

    test_dfs[1] = test_dfs[0].copy().drop(1).reset_index(drop=True)
    test_dfs[1].loc[test_dfs[1]['product_key'] == 'a1', 'colour'] = 'neon purple'

    added_row = pd.DataFrame({'colour': ['orange'], 'size': [0], 'product_key': ['f5'], 'date': ['01-02-2020']})

    test_dfs[1] = pd.concat([
        test_dfs[1],
        added_row], ignore_index=True)

    test_dfs[1]['date'] = '01-02-2020'

    test_dfs[2] = test_dfs[0].copy().drop(columns=['size'])
    test_dfs[3] = test_dfs[1].copy().drop(columns=['colour'])

    test_dfs[4] = test_dfs[2].copy()
    test_dfs[4].loc[len(test_dfs[4].index)] = {
        'colour': 'cyan',
        'product_key': np.nan,
        'date': '01-01-2020'
    }

    test_dfs[5] = test_dfs[3].copy()
    test_dfs[5].loc[len(test_dfs[5].index)] = {
        'size': 0,
        'product_key': np.nan,
        'date': '01-02-2020'
    }

    test_dfs[5].loc[len(test_dfs[5].index)] = {
        'size': 10,
        'product_key': np.nan,
        'date': '01-02-2020'
    }

    test_dfs[6] = test_dfs[4].copy()
    test_dfs[6].loc[len(test_dfs[6].index)] = {
        'colour': 'lime',
        'product_key': 'r20',
        'date': '01-01-2020'
    }

    test_dfs[6].loc[len(test_dfs[6].index)] = {
        'colour': 'gray',
        'product_key': 'r20',
        'date': '01-01-2020'
    }

    test_dfs[7] = test_dfs[5].copy()

    test_dfs[7] = pd.concat(
        [
            test_dfs[7],
            pd.DataFrame(
                {
                    'size': [2, 0, 16, 8],
                    'product_key': ['r20', 'q11', 'q11', 'q11'],
                    'date': '01-02-2020'
                }
            )
        ]
    )

    test_dfs[8] = pd.DataFrame({
        'size': [10, 10, 12],
        'product_key': [np.nan, np.nan, np.nan],
        'date': '01-02-2020'
    })

    expected = {}
    actual = {}

    # case_1: No old, no set
    expected['case_1'] = {}
    expected['case_1']['text'] = 'case_1: No duplicates, no old, no set'
    expected['case_1']['df'] = test_dfs[0].copy()
    expected['case_1']['df']['change'] = 'created'
    expected['case_1']['df']['change_date'] = '01-01-2020'
    expected['case_1']['df'] = expected['case_1']['df'][col_reorder(expected['case_1']['df'], ['product_key'], ['change_date', 'change'])].reset_index(drop=True)
    expected['case_1']['new_duplicate_set'] = set()

    actual['case_1'] = track_changes(test_dfs[0], 'product_key', 'date')

    # case_2: Old, new with removed, created and modified changes, no set
    expected['case_2'] = {}
    expected['case_2']['text'] = 'case_2: Old, new with removed, created and modified changes, no set'
    expected['case_2']['df'] = pd.concat(
        [
            test_dfs[1],
            test_dfs[0]
        ]
    ).drop_duplicates(subset=['product_key'])

    expected['case_2']['df'] = expected['case_2']['df'].loc[~expected['case_2']['df']['product_key'].isin(['d4', 'c3'])]
    expected['case_2']['df']['change'] = ['modified', 'created', 'removed']
    expected['case_2']['df']['change_date'] = '01-02-2020'
    expected['case_2']['df'] = expected['case_2']['df'][col_reorder(expected['case_2']['df'], ['product_key'], ['change_date', 'change'])].sort_values(by=['product_key']).reset_index(drop=True)
    expected['case_2']['new_duplicate_set'] = set()

    actual['case_2'] = track_changes(test_dfs[1], 'product_key', 'date', old=test_dfs[0])

    # case_3. Same as case 2, but with a set that filters out two values
    dup_set = {}
    dup_set['case_3'] = set(['a1', 'f5'])
    expected['case_3'] = {}
    expected['case_3']['text'] = 'case_3. Same as case 2, but with a set that filters out two values'
    expected['case_3']['changes'] = expected['case_2']['df'].loc[~expected['case_2']['df']['product_key'].isin(dup_set['case_3'])].reset_index(drop=True)
    expected['case_3']['incorrect'] = test_dfs[1].loc[test_dfs[1]['product_key'].isin(dup_set['case_3'])].copy()
    expected['case_3']['incorrect']['change'] = 'untracked_duplicate_key'
    expected['case_3']['incorrect']['change_date'] = test_dfs[1]['date'][0]
    expected['case_3']['df'] = pd.concat([expected['case_3']['changes'], expected['case_3']['incorrect']]).reset_index(drop=True)
    expected['case_3']['new_duplicate_set'] = dup_set['case_3']

    actual['case_3'] = track_changes(test_dfs[1], 'product_key', 'date', old=test_dfs[0], duplicate_set=dup_set['case_3'])


    # case_4. Same as case_2 but with columns colour added and column size removed
    expected['case_4'] = {}
    expected['case_4']['text'] = 'case_4. Same as case_2 but with columns colour added and column size removed'
    expected['case_4']['df'] = test_dfs[2].merge(test_dfs[3], left_on='product_key', right_on='product_key', how='outer', suffixes=(None, '_r')).rename(columns={'date_r': 'change_date'})
    expected['case_4']['df'].loc[expected['case_4']['df']['product_key'] == 'f5', 'date'] = '01-02-2020'
    expected['case_4']['df']['change_date'] = '01-02-2020'
    expected['case_4']['df']['change'] = ['modified', 'removed', 'modified', 'modified', 'created']
    expected['case_4']['df'].loc[expected['case_4']['df']['change'] != 'removed', 'colour'] = np.nan
    expected['case_4']['df'].loc[expected['case_4']['df']['change'] != 'removed', 'date'] = '01-02-2020'
    expected['case_4']['df'] = expected['case_4']['df'][col_reorder(expected['case_4']['df'], ['product_key'], ['change_date', 'change'])].reset_index(drop=True)
    expected['case_4']['new_duplicate_set'] = set()

    actual['case_4'] = track_changes(test_dfs[3], 'product_key', 'date', old=test_dfs[2])

    # case_5. Columns colour added and size removed with na's in both, rows modified, removed, created (same as cases 3 and 4)
    expected['case_5'] = {}
    expected['case_5']['text'] = 'case_5. Columns colour added and size removed with na in both, rows modified, removed, created (same as cases 3 and 4)'
    expected['case_5']['df'] = expected['case_4']['df'].copy()
    expected['case_5']['new_duplicate_set'] = set()

    expected['case_5']['incorrect'] = test_dfs[5].loc[test_dfs[5]['product_key'].isna()].copy()
    expected['case_5']['incorrect']['change_date'] = expected['case_5']['incorrect']['date']
    expected['case_5']['incorrect']['change'] = 'untracked_missing_key'
    expected['case_5']['df'] = pd.concat([expected['case_5']['df'], expected['case_5']['incorrect']]).reset_index(drop=True)

    actual['case_5'] = track_changes(test_dfs[5], 'product_key', 'date', old=test_dfs[4])

    #case_6. Same as case_5 but with duplicates
    expected['case_6'] = {}
    expected['case_6']['text'] = 'case_6: Same as case_5 but with duplicates'
    expected['case_6']['df'] = expected['case_5']['df'].copy()
    expected['case_6']['new_duplicate_set'] = (
        set(
            test_dfs[6].dropna(subset=['product_key']).loc[test_dfs[6].duplicated(subset=['product_key'], keep=False), 'product_key'].to_list())
            .union(set(
                test_dfs[7].dropna(subset=['product_key']).loc[test_dfs[7].dropna(subset=['product_key']).duplicated(subset=['product_key'], keep=False), 'product_key'].to_list()
        ))
    )
    expected['case_6']['incorrect'] = pd.concat(
        [test_dfs[7].loc[test_dfs[7]['product_key'].isin(expected['case_6']['new_duplicate_set'])].copy()]
    )
    expected['case_6']['incorrect']['change_date'] = expected['case_6']['incorrect']['date']
    expected['case_6']['incorrect'].loc[expected['case_6']['incorrect']['product_key'].isin(expected['case_6']['new_duplicate_set']) & (expected['case_6']['incorrect']['product_key'].notna()), 'change'] = 'untracked_duplicate_key'
    expected['case_6']['df'] = pd.concat([expected['case_6']['df'], expected['case_6']['incorrect']]).reset_index(drop=True)
    expected['case_6']['df'] = (
        pd.concat([expected['case_6']['df'][0:5], 
                expected['case_6']['df'][7:], 
                expected['case_6']['df'][5:7]]).reset_index(drop=True)
    )
    actual['case_6'] = track_changes(test_dfs[7], 'product_key', 'date', old=test_dfs[6])

    # case_7. Same as case_6 but with a duplicate set
    dup_set['case_7'] = set(['a1', 'b2'])
    expected['case_7'] = {}
    expected['case_7']['text'] = 'case_7. Same as case_6 but with a duplicate set'
    expected['case_7']['df'] = expected['case_6']['df'].loc[~expected['case_6']['df']['product_key'].isin(dup_set['case_7'])].copy()
    expected['case_7']['new_duplicate_set'] = expected['case_6']['new_duplicate_set'].union(dup_set['case_7'])

    expected['case_7']['df']['size'] = expected['case_7']['df']['size'].astype(int)
    expected['case_7']['df']['colour'] = expected['case_7']['df']['colour'].astype(float)
    expected['case_7']['incorrect'] = test_dfs[7].loc[test_dfs[7]['product_key'].isin(dup_set['case_7'])].copy()
    expected['case_7']['incorrect'].loc[:,'change_date'] = expected['case_7']['incorrect']['date']
    expected['case_7']['incorrect'].loc[:, 'change'] = 'untracked_duplicate_key'
    expected['case_7']['df'] = pd.concat(
        [expected['case_7']['df'], 
        expected['case_7']['incorrect']]).reset_index(drop=True)
    expected['case_7']['df'] = expected['case_7']['df'].reindex([0, 1, 2, 9, 3, 4, 5, 6, 7, 8]).reset_index(drop=True)

    actual['case_7'] = track_changes(
        test_dfs[7], 'product_key', 'date', 
        old=test_dfs[6], duplicate_set=dup_set['case_7'])
    
    # case_8: New with na, duplicates, no old, with a set
    dup_set['case_8'] = set(['r20'])

    expected['case_8'] = {}
    expected['case_8']['text'] = 'case_8: New with na, duplicates, no old, with a set'
    expected['case_8']['df'] = test_dfs[7].copy()
    expected['case_8']['df']['change'] = 'created'
    expected['case_8']['df']['change_date'] = '01-02-2020'
    expected['case_8']['df'] = expected['case_8']['df'].drop_duplicates(subset=['product_key'], keep=False).dropna(subset=['product_key'])
    expected['case_8']['df'] = expected['case_8']['df'][col_reorder(expected['case_8']['df'], ['product_key'], ['change_date', 'change'])]
    expected['case_8']['df'] = expected['case_8']['df'].loc[~expected['case_8']['df']['product_key'].isin(dup_set['case_8'])].reset_index(drop=True)

    expected['case_8']['new_duplicate_set'] = dup_set['case_8'].union(
        set(
                test_dfs[7].dropna(subset=['product_key']).loc[test_dfs[7].dropna(subset=['product_key']).duplicated(subset=['product_key'], keep=False), 'product_key'].to_list()
        )
    )

    expected['case_8']['incorrect'] = (
        test_dfs[7]
            .loc[test_dfs[7]['product_key'].isin(expected['case_8']['new_duplicate_set'])]).copy()


    expected['case_8']['incorrect']['change_date'] = expected['case_8']['incorrect']['date']
    expected['case_8']['incorrect']['change'] = 'untracked_duplicate_key'

    expected['case_8']['incorrect_na'] = test_dfs[7].loc[test_dfs[7]['product_key'].isna()].copy()

    expected['case_8']['incorrect_na']['change_date'] = expected['case_8']['incorrect_na']['date']
    expected['case_8']['incorrect_na']['change'] = 'untracked_missing_key'
    expected['case_8']['df'] = pd.concat([
        expected['case_8']['df'],
        expected['case_8']['incorrect'],
        expected['case_8']['incorrect_na']
    ]).reset_index(drop=True)
    actual['case_8'] = track_changes(test_dfs[7], 'product_key', 'date', duplicate_set=dup_set['case_8'])

    expected['case_9'] = {}
    expected['case_9']['text'] = 'case_9: New with na only, no old, with a set'
    expected['case_9']['df'] = test_dfs[8].copy()
    expected['case_9']['df']['change'] = 'untracked_missing_key'
    expected['case_9']['df']['change_date'] = expected['case_9']['df']['date']
    expected['case_9']['new_duplicate_set'] = dup_set['case_8']

    actual['case_9'] = track_changes(test_dfs[8], 'product_key', 'date', duplicate_set=dup_set['case_8'])

    for case in expected:
        with check:
            assert actual[case]['new_duplicate_set'] == expected[case]['new_duplicate_set'], error_msg(expected[case]['text'], 'new_duplicate_set')

        with check:
            assert actual[case]['changes_df'].equals(expected[case]['df']) == True, error_msg(expected[case]['text'], 'changes_df')



test_removed()
test_removed_error()

test_created()
test_created_error()

test_modified()
test_modified_error()

test_incorrect_key_handler()
test_incorrect_key_handler_error()

test_standardize_cols()
test_standardized_cols_error()

test_track_changes()