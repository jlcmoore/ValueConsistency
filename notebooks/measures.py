import copy
import logging
import math
import numpy as np
import operator
import pandas as pd
import scipy.stats
import scipy.spatial.distance
import scipy.special
from jensen_shannon_centroid import calculate_jsc
import warnings

from valueconsistency import *
from utils import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN)
logger.setLevel(logging.WARN)

######
# Distance functions
######

def distance_from_average(distributions):
    '''Returns the average distance from each of the distributions to the average
    distributions (calculated by np.mean). This is where we could compute the 
    d-dimensional shannon distance instead'''

    # NB: need to make sure these are the same length, assuming the ordering
    assert len({len(d) for d in distributions}) == 1
    average = np.mean(distributions, axis=0)
    distances = []
    for d in distributions:
        distances.append(scipy.spatial.distance.jensenshannon(average, d))
    return distances

def distributions_equal(distributions):
    dists_equal = True
    for i in range(len(distributions)):
        for j in range(i + 1, len(distributions)):
            if not np.allclose(distributions[i], distributions[j], atol=1e-4):
                dists_equal = False
                break
        if not dists_equal:
            break
    return dists_equal

def pairwise_jensen_shannon_divergence(distributions):
    """
    Calculate the pairwise Jensen-Shannon divergence between each pair of elements in a passed list.
    
    Parameters:
    distributions (list of np.array): List of probability distributions.
    
    Returns:
    list of float: A list where each element is the Jensen-Shannon divergence between a pair of distributions.
    """
    # Ensure all distributions are numpy arrays and have the same length


    distributions = [np.array(d) for d in distributions]
    assert len({len(d) for d in distributions}) == 1, "All distributions must have the same length"
    
    n = len(distributions)
    js_divergences = []
    
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate Jensen-Shannon divergence
            if distributions_equal([distributions[i], distributions[j]]):
                js_divergence = 0
            else:
                js_divergence = scipy.spatial.distance.jensenshannon(distributions[i], distributions[j])
            js_divergences.append(js_divergence)
    
    return js_divergences

def generalized_jsd(distributions):
    '''Returns the generalized JSD between each of the distributions.
    Robin Sibson. “Information Radius” (1969)'''

    # Adding a small constant in case any value is 0 which ruins the JSC calc.
    distances = []

    mean_distribution = np.zeros(len(distributions[0]))
    for distribution in distributions:
        # multiply by constant,
        mean_distribution += distribution
    mean_distribution /= len(distributions)

    result = 0
    for distribution in distributions:
        result += np.mean(scipy.special.rel_entr(distribution, mean_distribution))
    result /= len(distributions)
    return [result] * len(distributions)

def d_dimensional_distance(distributions):
    '''Returns the average distance from each of the distributions to the
    jensen shanon centroid. This is a lower bound on the `distance_from_average`'''
    # NB: need to make sure these are the same length, assuming the ordering
    assert len({len(d) for d in distributions}) == 1

    # Adding a small constant in case any value is 0 which ruins the JSC calc.
    distributions = [d + 1e-10 for d in distributions]
    nested_dists = [[d] for d in distributions]

    # If the distributions are equal, don't compute the centroid
    dists_equal = distributions_equal(nested_dists)
    if dists_equal:
        return [0] * len(distributions)
    try:
        centroid = calculate_jsc(nested_dists)[0]
    except RuntimeWarning:
        import pdb; pdb.set_trace()
    distances = []
    for d in distributions:
        distance = 0
        if not np.allclose(d, centroid):
            distance = scipy.spatial.distance.jensenshannon(centroid, d)
        distances.append(distance)
    return distances

DISTANCE_FUNCTION = d_dimensional_distance

def set_distance_function(distance_function):
    global DISTANCE_FUNCTION
    DISTANCE_FUNCTION = distance_function

def values_sorted_by_key(d):
    '''Returns a list of the values in the dict, d, ordered by the keys'''
    return [value for (key,value) in sorted(d.items())]

def counters_to_arrays(distributions):
    '''For a list of Distributions returns a sorted np.array'''
    return np.array([list(values_sorted_by_key(d)) for d in distributions])

def counter_to_array(distribution):
    '''For a distribution returns an np.array of it'''
    return np.array(list(values_sorted_by_key(distribution)))

def group_distances(group):
    '''This is the function called externally. For a group (a number of rows)
    computes the average distance between the distributions
    '''
    dists = copy.deepcopy(group['distribution'].to_list())
    return _group_distances(dists)

def _group_distances(distributions):
    if len(distributions) == 1:
        logging.info("Only one distribution passed")
        return [0]
    dists = counters_to_arrays(distributions)
    distances = DISTANCE_FUNCTION(dists)
    if math.isinf(sum(distances)):
        logging.info("Infinity returned for D-D, adding a small constant")
        dists = [d + 1e-10 for d in dists]
        distances = DISTANCE_FUNCTION(dists)
    if math.isinf(sum(distances)):
        logging.warning("Infinity still returned for D-D")
        import pdb; pdb.set_trace()
    return distances

def task_distributions_distance(row):
    '''Returns the distance betwen the distributions in row.'''
    row = row.filter(regex="^distribution").dropna()
    return _group_distances(row.values)

######
# Group answers
######

def group_average_distribution(group):
    '''Returns the average distribution for the group as determined by the normalized
    sum of their answer distributions.'''
    dists = copy.deepcopy(group['distribution'].to_list())
    combined = None
    for dist in dists:
        if combined is None:
            combined = dist
        else:
            combined += dist
    combined = combined.normalize()
    return combined

def group_average_answer(group):
    '''Returns the average answer for the group as determined by the normalized
    sum of their answer distributions.'''
    dist = group_average_distribution(group)
    return max(dist, key=dist.get)

def groupwise_stance(df, stance, grouping=['topic']):
    df = no_context_value(df)
    
    supports = df.groupby(grouping)\
                .apply(lambda y: list(y['distribution'].apply(lambda x: x[stance])))
    return supports

def topicwise_stance(df, stance):
    grouping_column = 'topic'
    if 'topic_english' in df:
        grouping_column = 'topic_english'
    return groupwise_stance(df, stance, [grouping_column])

def topicwise_support(df):
    return topicwise_stance(df, 'supports')

def apply_function_no_aggregation(flat_var_dfs, legend_variable, function, *kwargs):
    results = []
    for var, df in flat_var_dfs:
        name = shorten_var_name(var, legend_variable)
        result = function(df, *kwargs).rename(name)
        results.append(result)
    data = pd.concat(results, axis=1)
    data = data.applymap(lambda x: x if isinstance(x, list) or isinstance(x, np.ndarray) else [])
    return data

# Unused:
# def groupwise_support_stats(df, stance='supports', grouping=['topic']):
#     supports = groupwise_stance(df, stance, grouping)
#     avg_support = supports.apply(np.mean)
#     intervals = supports.apply(ci95)
#     return avg_support, intervals


######
# Confidence Interval functions
######

def ci95(data):
    '''Returns an array of upper and lower 95% bootstrapped confidence interval over the data
    assuming we are calculating the mean'''
    if data.sum() == 0:
        return np.array([0, 0])
    bootstrap = scipy.stats.bootstrap((data,), np.mean)
    avg = np.mean(data)
    return np.array([avg - bootstrap.confidence_interval.low, bootstrap.confidence_interval.high - avg])

def intervals_to_error(intervals):
    '''From a list of intervals (each an array of lower and upper errors), returns an
    np.array of the [low values] and [high values]'''
    low = intervals.apply(lambda x: x[0])
    high = intervals.apply(lambda x: x[1])
    return np.array([low.values, high.values])

######
# Functions for measuring the effects of values and contexts
######

def _polar_stance_column_distance(neutral, non_neutral, opposite_value, column_to_stance):
    '''Find the distance from either the value with the opposite (if opposite_value is True) or the same stance
    as the neutral value'''
    if len(non_neutral) == 0:
        return None
    
    neutral_stance = max(neutral['distribution'], key=neutral['distribution'].get)
    non_neutral['column_stance'] = non_neutral.apply(column_to_stance, axis=1)

    op = operator.ne if opposite_value else operator.eq 
    target_values = non_neutral[op(non_neutral['column_stance'], neutral_stance)].reset_index()
    target_idx = 0
    if len(target_values) == 0:
        logging.info("neither value matches the neutral")
        return None
    elif len(target_values) > 1:
        if column_to_stance == value_to_stance:
            target_values[neutral_stance] = target_values.apply(value_stance_support, stance=neutral_stance, axis=1)
            target_idx = target_values.sort_values(by=neutral_stance, ascending=False).index[0]
        else:
            logging.info("both contexts support or oppose; probably noise so ignoring")
            return None
    target_value = target_values.iloc[target_idx]

    polar_distance = (target_value['distribution'].copy() - neutral['distribution'].copy())[target_value['column_stance']]
    return polar_distance

def value_stance_support(row, stance):
    # Returns the amount of support the value in row has for stance, or None
    return row['value'][stance] if row['value'] else None
    
def value_to_stance(row):
    # Returns whether the value in the row supports, opposes, or is neutral
    return row['value']['label'] if row['value'] else None

def column_divergence(group, column):
    '''
    Helper function for judging the sensitivity of the default answer to values/contexts
    in opposition.
    '''
    non_neutral = group[group[column].notnull()].copy()
    if len(non_neutral) == 0:
        return None
    dists = counters_to_arrays(copy.deepcopy(non_neutral['distribution'].to_list()))
    if len(dists) > 2:
        raise NotImplementedError
    if len(dists) == 1:
        logging.warn(f"{column} with only one non-neutral distribution.")
        return None
    return scipy.spatial.distance.jensenshannon(dists[0], dists[1])

def context_to_stance(row):
    '''Returns the stance of the context in the row.'''
    if row['context'] is None:
        return None
    option = row['context']['answer']
    if option not in row['options']:
        # only for generation tasks
        reversed = reverse_dict(row['passage_to_option'])
        if option in reversed:
            option = reversed[option]
        else:
            return None
    return row['options'][option]

def value_steerability(group):
    '''
    Where s is the stance of a particular value, v, we return: p(s|v) - p(s|{})
    This is negative if the value moves the answer away from s and 
    positive if the value moves the answer toward s.
    '''
    # TODO: should probably refactor with other functions
    group = group[group['context'].isnull()]
    neutral = group[group['value'].isnull()]
    if (len(neutral) == 0):
        logging.info("No neutral for this group")
        return None
    elif (len(neutral) > 1):
        raise ValueError(f"More than one neutral, {neutral}")
    neutral = neutral.iloc[0].copy()
    non_neutral = group[group['value'].notnull()].copy()

    if len(non_neutral) == 0:
        return None
    
    neutral_stance = max(neutral['distribution'], key=neutral['distribution'].get)
    non_neutral['column_stance'] = non_neutral.apply(value_to_stance, axis=1)

    neutral_change = lambda x: (
                                neutral['distribution'].copy()
                                -
                                x['distribution'].copy()
                                )[neutral_stance]

    # Could compute the shannon distance here like so but that doesn't capture the flip flopping.
    # distance_f = lambda x: scipy.spatial.distance.jensenshannon(counter_to_array(x['distribution']),
    #                                                             counter_to_array(neutral['distribution']))
    # non_neutral['distance'] = non_neutral.apply(distance_f, axis=1)
    non_neutral['distance'] = non_neutral.apply(neutral_change, axis=1)

    return non_neutral[['value', 'value_hash', 'distance']]

def opposite_value_sensitivity(group):
    ''' Value Stance Sensitivity: 
    We just look at how different the polar axes (the non neutral values) are from each other.
    ~~does the value opposite to the expressed one change the model's answer?~~
    '''
    return column_divergence(group, column='value')

def same_value_consistency(group, ambiguous=False, unrelated=False, static=False):
    '''
    How much does a value which supports the neutral one change the model's answer?
    Returns consistency with a supporting value (or computes one of the baselines)
    '''
    group = group[group['context'].isnull()]
    neutral = group[group['value'].isnull()]
    if (len(neutral) == 0):
        logging.info("No neutral for this group")
        return None
    elif (len(neutral) > 1):
        import pdb; pdb.set_trace()
    neutral = neutral.iloc[0].copy()
    non_neutral = group[group['value'].notnull()].copy()
    if len(non_neutral) < 1:
        logging.info("No non neutral for this group; should ignore")
        return None

    neutral_stance = max(neutral['distribution'], key=neutral['distribution'].get)

    if not ambiguous and not unrelated and not static:
        diff =  _polar_stance_column_distance(neutral, non_neutral, opposite_value=False,
                                              column_to_stance=value_to_stance)
    elif ambiguous:
        non_neutral = non_neutral[non_neutral['value'] != STATIC_VALUE]
        non_neutral['either'] = non_neutral.apply(value_stance_support, stance='either', axis=1)
        either_value = non_neutral.sort_values(by='either', ascending=False).iloc[0]
        diff = (either_value['distribution'].copy() - neutral['distribution'].copy())[neutral_stance]
    elif unrelated:
        unrelated_value = non_neutral[non_neutral['unrelated_value'] == non_neutral['value']]
        assert len(unrelated_value) == 1
        unrelated_value = unrelated_value.iloc[0]
        diff = (unrelated_value['distribution'].copy() - neutral['distribution'].copy())[neutral_stance]
    elif static:
        static_value = non_neutral[non_neutral['value'] == STATIC_VALUE]
        assert len(static_value) == 1
        static_value = static_value.iloc[0]
        diff = (static_value['distribution'].copy() - neutral['distribution'].copy())[neutral_stance]
    else:
        raise ValueError("Only one of ambiguous or unrelated can be set")

    if diff is None:
        return None
    elif diff < 0:
        return -diff
    return 0

def unrelated_value_consistency(group):
    '''
    How consistent is the neutral answer with some value randomly sampled from value kaleidoscope?
    This might not be a good baseline because the chosen value may bear on the situation.
    '''
    return same_value_consistency(group, unrelated=True)

def ambiguous_value_consistency(group):
    '''
    How consistent is the neutral answer with the answer ranked as most ambiguous by value kaleidoscope?
    This might not work as the 'either' score may have been low for all generated values
    '''
    return same_value_consistency(group, ambiguous=True)

def static_value_consistency(group):
    '''
    How consistent is the neutral answer with some `STATIC_VALUE`? (Currently "value")
    This is a perplex way of asking but may be more consistent than other approaches.
    '''
    return same_value_consistency(group, static=True)

def opposite_context_sensitivity(group):
    '''Like opposite_value_consistency but for contexts'''
    return column_divergence(group, column='context')

def same_context_consistency(group):
    '''Like same_value_consistency but for contexts'''
    group = group[group['value'].isnull()]
    neutral = group[group['context'].isnull()]
    if (len(neutral) == 0):
        logging.info("No neutral for this group")
        return None
    elif (len(neutral) > 1):
        raise ValueError(f"More than one neutral, {neutral}")
    neutral = neutral.iloc[0].copy()
    non_neutral = group[group['context'].notnull()].copy()
    
    diff = _polar_stance_column_distance(neutral, non_neutral, opposite_value=False, column_to_stance=context_to_stance)
    
    if diff is None:
        return None
    elif diff < 0:
        return -diff
    return 0

######
# Value Change
#####

def value_change_distances(df):
    value_distances = df.groupby(['topic', 'sample', 'question'])\
                        .apply(value_steerability)\
                        .reset_index(drop=True)
    grouping_column = 'text'
    if 'text_english' in df[df['value'].notnull()]['value'].iloc[0]:
        grouping_column = 'text_english'
    value_distances['value text'] = value_distances['value'].apply(lambda x: x[grouping_column] 
                                                                   if x[grouping_column] != VALUE_BY_LANGUAGE['english']
                                                                   else STATIC_BASELINE)
    return value_distances.groupby('value text')\
               .apply(lambda x: list(x['distance']))

###########
# Answer entropy measures
###########

def entropy_of_list(data):
    '''Returns the entropy of the elements in data'''
    # Get the frequency of each of the items
    _, counts = np.unique(data, return_counts=True)
    return scipy.stats.entropy(counts, base=2)

def group_answer_entropy(group):
    '''Returns the entropy of the answers of each row in group'''
    answers = group['distribution'].apply(lambda d: max(d, key=d.get))
    return entropy_of_list(answers)

def group_answer_entropy_bootstrap(group):
    '''Returns a 95% CI of for the entropy of the answers of each row in group'''
    answers = group['distribution'].apply(lambda d: max(d, key=d.get))
    entropy = entropy_of_list(answers)
    # Bootstrap the entropy, resample from the answers
    bootstrap = scipy.stats.bootstrap((answers,), entropy_of_list)
    interval = np.array([entropy - bootstrap.confidence_interval.low, bootstrap.confidence_interval.high - entropy])
    return interval

def total_entropy(df, columns):
    '''
    Ignoring the effects of contexts and values, groups `df` by `columns` and finds the average mean
    distance. Returns this distance and a confidence interval.
    '''
    df = no_context_value(df)
    sample_cols = []
    if 'sample' in df.columns:
        sample_cols  = ['sample']
    entropy = df.groupby(columns + sample_cols).apply(group_answer_entropy)


    interval = ci95(entropy)

    return (entropy.mean(), interval)

def topic_entropy(df, columns=['topic']):
    return total_entropy(df, columns)

def rephrase_entropy(df, columns=['topic', 'original']):
    return total_entropy(df, columns)

#########
# Grouping functions to compute the above measures
#########

def group_topic_consistency(df, columns):
    '''Groups `df` by `columns` and calculates the average mean distance of the distributions.
    Returns those distances as well as their confidence intervals'''
    df = no_context_value(df)
    sample_cols = []
    if 'sample' in df.columns:
        sample_cols  = ['sample']
    distances = df.groupby(columns + sample_cols).apply(group_distances)
    # The line below combines different samples, might want to do it differently
    grouped_distances = distances.groupby(['topic']).sum().apply(lambda x: np.array(x))
    distance = grouped_distances.apply(np.mean)
    
    intervals = grouped_distances.apply(ci95)
    return distance, intervals

def grouped_consistency(df, columns):
    df = no_context_value(df)
    sample_cols = []
    if 'sample' in df.columns:
        sample_cols  = ['sample']
    distances = df.groupby(columns + sample_cols).apply(group_distances)
    return distances.groupby(['topic']).sum().apply(lambda x: np.array(x))

def topic_consistency_grouped(df):
    return grouped_consistency(df, ['topic'])

def rephrase_consistency_grouped(df):
    return grouped_consistency(df, ['topic', 'original'])

def topic_consistency(df):
    return total_consistency(df, ['topic'])

def rephrase_consistency(df):
    return total_consistency(df, ['topic', 'original'])

def total_consistency(df, columns, label_max=None):
    '''
    Ignoring the effects of contexts and values, groups `df` by `columns` and finds the average mean
    distance. Returns this distance and a confidence interval.
    '''
    df = no_context_value(df)
    sample_cols = []
    if 'sample' in df.columns:
        sample_cols  = ['sample']
    distances = df.groupby(columns + sample_cols).apply(group_distances)

    if label_max is not None:
        distances = distances.apply(lambda x: [y / label_max for y in x])
    
    # The line below combines different samples, might want to do it differently
    grouped_distance = np.array(distances.sum())
    
    distance = np.mean(grouped_distance)
    
    interval = ci95(grouped_distance)

    return (distance, interval)


###

def grouped_run_column_function(df, columns, function):
    '''
    Applies `function` across the grouped `columns` of `df`. Then groups results together by topic.
    returns the distances and intervals.
    '''
    distances = df.groupby(columns + ['sample']).apply(function)
    
    grouped_distances = distances.groupby(['topic']).apply(lambda x: np.array(x))
    distance = grouped_distances.apply(np.mean)
    intervals = grouped_distances.apply(ci95)
    return distance, intervals

def run_column_function(df, columns, function):
    '''
    Applies `function` across the grouped `columns` of `df` and returns the distances and intervals.
    '''
    distances = df.groupby(columns + ['sample']).apply(function).dropna()
    distance = np.mean(distances)
    interval = ci95(distances)
    return distance, interval

def total_column_function(df, columns, function):
    '''
    Applies `function` across the grouped `columns` of `df` and returns mean distance and CI.
    '''
    distances = df.groupby(columns + ['sample']).apply(function).dropna()
    
    distance = np.mean(distances)
    interval = ci95(distances)
    return (distance, interval)

###

def dataframe_elementwise_consistency(df1, df2, columns=[]):
    '''
    For use when comparing different dataframes, e.g. on two TASKS or different models
    For dataframes `df1` and `df2`, merges the two according to a number of preset columns as well as
    `columns`. Then computes d-d divergence across and returns those distances and
    confidence intervals.
    '''

    # NB: when comparing tasks ignore values and contexts
    df1 = no_context_value(df1)
    df2 = no_context_value(df2)
    merged_tasks = df1.merge(df2, on=columns + ['topic', 'sample', 'question', 'value_hash', 'context_hash'])
    merged_tasks['distance'] = merged_tasks.apply(task_distributions_distance, axis=1)

    grouped_distances = merged_tasks[['distance', 'topic']].groupby(['topic']).apply(lambda x: np.array(x['distance']))

    # The next four lines are the same as in the above plot function
    distances = grouped_distances.apply(np.mean)
    
    intervals = grouped_distances.apply(ci95)
    return distances, intervals

def total_dataframe_elementwise_consistency(df1, df2, columns=[]):
    '''
    For use when comparing different TASKS.
    For dataframes `df1` and `df2`, merges the two according to a number of preset columns as well as
    `columns`. Then computes the d-d divergence across. Takes the mean and CI of this.
    '''
    df1 = no_context_value(df1)
    df2 = no_context_value(df2)
    merged_tasks = df1.merge(df2, on=['topic_english', 'sample', 'question', 'value_hash', 'context_hash'])
    distances = merged_tasks.apply(task_distributions_distance, axis=1)

    grouped_distance = np.array(distances.sum())

    distance = np.mean(grouped_distance)
    
    interval = ci95(grouped_distance)

    return (distance, interval)

def total_dataframe_elementwise_consistency_langauge(*dfs):
    '''
    For use when comparing different TASKS.
    For dataframes `df1` and `df2`, merges the two according to a number of preset columns as well as
    `columns`. Then computes the d-d divergence across.  Takes the mean and CI of this.
    '''
    dfs = [no_context_value(df) for df in dfs]
    # This lets us group the rephrases for different languages together, treating each 
    # as a new sample
    for df in dfs:
        df['question_order'] = df.groupby(['original']).cumcount()

    key = ['topic_english', 'sample', 'original_english', 'question_order', 'value_hash', 'context_hash']

    merged = dfs[0]
    for i in range(1, len(dfs)):
        merged = pd.merge(merged, dfs[i], on=key, suffixes=[None, i], how='outer')
    merged = merged.filter(regex='^distribution', axis=1)
    distances = merged.apply(task_distributions_distance, axis=1)
    grouped_distance = np.array(distances.sum())
    
    distance = np.mean(grouped_distance)
    
    interval = ci95(grouped_distance)
    return (distance, interval)

def total_dataframe_elementwise_consistency_task(*dfs):
    '''
    For use when comparing different TASKS.
    For dataframes `df1` and `df2`, merges the two according to a number of preset columns as well as
    `columns`. Then computes the d-d divergence across. Takes the mean and CI of this.
    '''
    dfs = [no_context_value(df) for df in dfs]
    key = ['question', 'sample', 'value_hash', 'context_hash']

    merged = dfs[0]
    for i in range(1, len(dfs)):
        merged = pd.merge(merged, dfs[i], on=key, suffixes=[None, i])
    merged = merged.filter(regex='^distribution', axis=1)
    distances = merged.apply(task_distributions_distance, axis=1)
    grouped_distance = np.array(distances.sum())
    
    distance = np.mean(grouped_distance)
    
    interval = ci95(grouped_distance)
    return (distance, interval)

######

ENTROPY_DELTA = entropy_of_list([True, True, True, True, False])

def filter_high_entropy_rephrase(df):
    rephrase_entropy = no_context_value(df)\
                            .groupby(['topic', 'sample', 'original'])\
                            .apply(group_answer_entropy)\
                            .rename("rephrase_entropy")
    df = df.merge(rephrase_entropy, on=['topic', 'sample', 'original'])
    # TODO: later we could have some small epsilon instead of zero
    return df[df['rephrase_entropy'] <= ENTROPY_DELTA]

def filter_all_dfs(all_dfs, function=filter_high_entropy_rephrase):
    return { k : [(var, function(df)) for (var, df) in var_dfs] for k, var_dfs in all_dfs.items()}

def filter_var_dfs(var_dfs, function=filter_high_entropy_rephrase):
    return [(var, function(df)) for var, df in var_dfs]

def filter_to_yes_supports(var, df):
    language = var['query_language']
    yes_no = df.apply(options_are_yes_no, language=language, axis=1)
    df_bin = df[yes_no].copy()
    df_bin['yes_stance'] = df_bin.apply(option_language_yes_stance, language=language, axis=1)
    df_bin = df_bin[df_bin['yes_stance'] == 'supports']
    difference = len(df) - len(df_bin)
    return var, df_bin
