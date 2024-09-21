import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textwrap
from IPython.display import display

from measures import *
from valueconsistency import *
from utils import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN)
logger.setLevel(logging.WARN)

'''
Plotting functions
'''

AVG_MEAN_DIST_LABEL = "Avg. Jen-Shan Dist from Mean Distribution"
YLIM = (0, .5)

INCONSISTENT_MEAURE_LABEL = "inconsistency"

TITLE_CHARS_PER_INCH = 9

PAIRED_COLORS = [matplotlib.colormaps['tab20'](x) for x in range(20)]
SINGLE_COLORS = [matplotlib.colormaps['tab20'](x) for x in range(0, 20, 2)]

# CONDITION_TO_COLOR = {condition : color for condition, color in zip(CONDITION_ORDER, _COLORS[len(AGGREGATIONS):])}

ALL_MODELS = [
'gpt-4o',
None,
'meta-llama/Meta-Llama-3-70B-Instruct',
'meta-llama/Meta-Llama-3-70B',
'meta-llama/Llama-2-70b-chat-hf',
'meta-llama/Llama-2-70b-hf',
'CohereForAI/c4ai-command-r-v01',
None,
'01-ai/Yi-34B-Chat',
'01-ai/Yi-34B',
'stabilityai/japanese-stablelm-instruct-beta-70b',
# 'meta-llama/Llama-2-70b-hf'
None,
'human',
None,
'meta-llama/Meta-Llama-3-8B-Instruct',
'meta-llama/Meta-Llama-3-8B',
'meta-llama/Llama-2-7b-chat-hf',
'meta-llama/Llama-2-7b-hf',
]


ALL_MODEL_COLORS = {model : color for model, color in zip(ALL_MODELS, PAIRED_COLORS)}


#######
# Topic / rephrase / modality consistency
#######

def plot_consistency(var, df, columns, ax=None, title="", color='blue', plot_bound=True):
    '''
    Plots the total d-dimensional distance of df grouped by columns.
    '''

    distance, intervals = group_topic_consistency(df, columns)
    ordered = pd.concat([distance.rename('distance'),
                         intervals.rename('intervals')], axis=1).sort_values(by='distance',
                                                                             ascending=False)
    print(f"mean {np.mean(distance):.2f}, min {min(distance):.2f}, max {max(distance):.2f}")
    error = intervals_to_error(ordered['intervals'])
   
    ax = ordered['distance'].plot.bar(yerr=error,
      color=color,
      ax=ax,
      ylim=YLIM,
      ylabel=INCONSISTENT_MEAURE_LABEL,
      title=title
    )
    if plot_bound:
        plot_upper_bound(ax, var=var)

def plot_models_consistency(all_dfs, models, task, columns=[]):
    '''
    Plots the `columns`-grouped consistency of each model in `all_dfs` in the order of `models` 
    Same function as `plot_consistency`, just grouped by models.
    '''
    results = {}
    errors = {}
    for model in models:
        data = get_matching_data(all_dfs[model], {'task' : task})
        if not data:
            continue
        var, df = data
    
        distances, intervals = group_topic_consistency(df, columns=columns)
        results[model] = distances
        errors[model] = intervals
    
    data_df = pd.DataFrame(results)
    yerr = errors_dict_reshape(errors)

    data_df.plot.bar(yerr=yerr,
                     ylim=YLIM,
                      ylabel=AVG_MEAN_DIST_LABEL,
                      title=f"{task} inconsistency, {columns} grouping")

#######
# Task consistency
#######

def plot_dataframe_elementwise_consistency(df1, df2, title="generation-classification task inconsistency"):
    '''Plots the total consistency between `df` and `df2`'''
    # TODO: need to sort out effect of modality
    distances, intervals = dataframe_elementwise_consistency(df1, df2)
    error = intervals_to_error(intervals)
    
    ax = distances.plot.bar(yerr=error,
                      ylim=YLIM,
                      ylabel=AVG_MEAN_DIST_LABEL,
                      title=title)

    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45)

def plot_models_task_consistency(all_dfs, models):
    '''Plots the task consistency of each model in `all_dfs` in the order of `models` 
    TODO: potentially redundant figure generation'''
    results = {}
    errors = {}

    for model in models:
        grouped_runs = group_tasks_by_run(all_dfs[model])
        data = grouped_runs[0]
        if not data or len(data) <= 1:
            continue
        (_, df1), (_, df2) = data

        distances, intervals = dataframe_elementwise_consistency(df1, df2)
        results[model] = distances
        errors[model] = intervals

    data_df = pd.DataFrame(results)
    yerr = errors_dict_reshape(errors)

    data_df.plot.bar(yerr=yerr,
                     ylim=YLIM,
                      ylabel=AVG_MEAN_DIST_LABEL,
                      title=f"Generation / Classification task inconsistency")
    

#######
# Value / context consistency
#######

def plot_column_function(df, columns, function, title):
    '''
    Plots the projection `df` on `function` grouped by `columns`.
    (E.g. for value or context consistency)
    '''
    distance, intervals = grouped_run_column_function(df, columns, function)
    error = intervals_to_error(intervals)

    distance.plot.bar(yerr=error,
                      ylim=YLIM,
                      title=f"{title} {columns}")

    # NB: how to display this data if not normalizing
    # data_df = distance.rename('avg. distance').to_frame().reset_index()
    # ax = data_df.plot.scatter(x='topic', y='avg. distance', yerr=error, ylim=(-1.05, 1.05),
    #                           title=f"{title} {columns}")
    # ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45)

def plot_models_function_columns(all_dfs, models, task, function, columns=[], title=""):
    '''
    Plots the projection of each model in `all_dfs` ordered by `models`
    on `function` grouped by `columns`. (E.g. for value or context consistency)
    '''
    results = {}
    errors = {}
    for model in models:
        data = get_matching_data(all_dfs[model], {'task': task})
        if not data:
            continue
        var, df = data
    
        distances, intervals = grouped_run_column_function(df, columns, function)
        results[model] = distances
        errors[model] = intervals
    
    data_df = pd.DataFrame(results)

    yerr = errors_dict_reshape(errors)
    
    data_df.plot.bar(yerr=yerr,
                     ylim=YLIM,
                      ylabel=AVG_MEAN_DIST_LABEL,
                      title=f"{title}, {columns} grouping")

#######
# Total consistency
#######

def plot_consistency_by_model(all_dfs, group=True, order=None):
    '''
    Plots all of the total consistency measures by each model in `all_dfs` ordered by `order`.
    '''
    results = {}
    errors = {}
    
    if order is None:
        order = all_dfs.keys()
    for model in order:
        grouped_runs = group_tasks_by_run(all_dfs[model], group)
    
        if len(grouped_runs) > 1:
            logging.info(f"More than one matching pair for {model}, using only the first.")
    
        #########
        # TODO: Right now just using the classificaiton task here but should combine the two
        #########
        if len(grouped_runs) < 1:
            continue
        print(model)
    
        data = grouped_runs[0]
    
        if len(data) == 2:
            (var1, df1), (var2, df2) = data
        else:
            ((var1, df1),) = data
    
        # error = interval_to_error(intervals)
        if model not in results:
            results[model] = {}
        lower_error = {}
        upper_error = {}

        default_grouping = ['topic']

        distance, (lower, upper) = total_consistency(df1, default_grouping + ['original'])
        results[model]['rephrase'] = distance
        lower_error['rephrase'] = lower
        upper_error['rephrase'] = upper

        distance, (lower, upper) = total_consistency(df1, default_grouping)
        results[model]['topic'] = distance
        lower_error['topic'] = lower
        upper_error['topic'] = upper

        if 'modality' in df1.columns:
            default_grouping += ['modality']
            distance, (lower, upper) = total_consistency(df1, default_grouping)
            results[model]['modality'] = distance
            lower_error['modality'] = lower
            upper_error['modality'] = upper
    
        if len(data) > 1:
            # TODO: add modality here
            distance, (lower, upper) = total_dataframe_elementwise_consistency(df1, df2, columns=['question'])
            results[model]['task'] = distance
            lower_error['task'] = lower
            upper_error['task'] = upper
        
        if var1['use_context']:
    
            distance, (lower, upper) = total_column_function(df1, default_grouping + ['question'], same_context_consistency)
            results[model]['context'] = distance
            lower_error['context'] = lower
            upper_error['context'] = upper
    
        if var1['use_values']:
    
            distance, (lower, upper) = total_column_function(df1, default_grouping + ['question'], same_value_consistency)
            results[model]['value'] = distance
            lower_error['value'] = lower
            upper_error['value'] = upper

            if 'unrelated_value' in var1 and var1['unrelated_value']:
                distance, (lower, upper) = total_column_function(df1, default_grouping + ['question'], unrelated_value_consistency)
                results[model]['unrelated v.'] = distance
                lower_error['unrelated v.'] = lower
                upper_error['unrelated v.'] = upper
    
        errors[model] = np.array(list(lower_error.values())), np.array(list(upper_error.values()))
    
    
    results_df = pd.DataFrame(results)
    
    ax = results_df.plot.bar(ylim=YLIM,
                          title=f"inconsistency measures (class. task used by default)",
                           yerr=pd.DataFrame(errors).T.to_numpy())
    
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=0)

def plot_consistency_by_task(var_dfs, title):
    plot_consistency_by_variable(var_dfs, title, 'task')

TWO_LABEL_MAX = np.mean(DISTANCE_FUNCTION(np.array([[1, 0], [0, 1]])))
THREE_LABEL_MAX = np.mean(DISTANCE_FUNCTION(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])))


def plot_consistency_by_variable(var_dfs, title='inconsistency measures', var_name='task',
                                 before_after_filter=False, colors=None,
                                 as_fraction_of_max=False, plot_bound=False,
                                 ax=None):
    '''
    Plots the total consistency measures across `dfs` grouped by `var_name`
    '''
    results = {}
    errors = {}

    # TODO: This is almost the same as the above function, consider refactoring
    if before_after_filter:
        var_dfs = [item for item in var_dfs for _ in range(2)]
    for i in range(len(var_dfs)):
        var, df = var_dfs[i]
        key = shorten_var_name(var, var_name)
        if i % 2 == 0 and before_after_filter:
            key = f"{key} filtered"
            df = filter_high_entropy_rephrase(df)
        if key not in results:
            results[key] = {}
            
        lower_error = {}
        upper_error = {}

        label_max = None
        if as_fraction_of_max:
            label_max = TWO_LABEL_MAX
            if var['allow_abstentions']:
                label_max = THREE_LABEL_MAX
    
        distance, (lower, upper) = total_consistency(df, ['topic'], label_max)
        results[key]['topic'] = distance
        lower_error['topic'] = lower
        upper_error['topic'] = upper

        # TODO: should insert modality consistency here
    
        distance, (lower, upper) = total_consistency(df, ['topic', 'original'], label_max)
        results[key]['paraphrase'] = distance
        lower_error['paraphrase'] = lower
        upper_error['paraphrase'] = upper

        if var['use_context']:
            distance, (lower, upper) = total_column_function(df, ['topic', 'question'], same_context_consistency)
            results[key]['context'] = distance
            lower_error['context'] = lower
            upper_error['context'] = upper

        if var['use_values'] and not is_schwartz(var):

            distance, (lower, upper) = total_column_function(df, ['topic', 'question'], same_value_consistency)
            results[key]['value'] = distance
            lower_error['value'] = lower
            upper_error['value'] = upper

        errors[key] = np.array(list(lower_error.values())), np.array(list(upper_error.values()))

    data_df = pd.DataFrame(results).sort_index(axis=1)
    display(data_df)
    yerr = pd.DataFrame(errors).sort_index(axis=1).T.to_numpy()

    if colors is not None:
        assert len(colors) >= len(var_dfs)
        colors = colors[0:len(var_dfs)]
    
    ax = data_df.plot.bar(ylim=YLIM,
                          yerr=yerr,
                          ylabel=INCONSISTENT_MEAURE_LABEL,
                          title=title,
                          color=colors, 
                          ax=ax)
    if not as_fraction_of_max and plot_bound:
        plot_upper_bound(ax, var=var_dfs[0][0])
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=50)

#########
# Value steerability / sensitivity / change
#########

def plot_upper_bound(axes, var):
    n_labels = 2
    if (('allow_abstentions' in var and var['allow_abstentions']) or 
            ('annotator_abstentions' in var and var['annotator_abstentions'])):
        n_labels = 3
    distributions = []
    for i in range(n_labels):
        dist = [0] * n_labels
        dist[i] = 1
        distributions.append(dist)
    value = np.mean(DISTANCE_FUNCTION(np.array(distributions)))
    xmin, xmax = axes.get_xlim()
    axes.hlines(y=value, xmin=xmin, xmax=xmax, colors='grey', linestyles='--', lw=2, label='Random')

def plot_steerability_by_model(all_dfs, task, order=None):
    '''
    Plots each model in ordered by `order` showing the sensitivity / steerability 
    each model has, on average, when a value/context has an opposite stance to the 
    neutral one.
    '''
    results = {}
    errors = {}
    
    if order is None:
        order = all_dfs.keys()
    for model in order:
        data = get_matching_data(all_dfs[model], {'task': task,
                                                  'use_context' : True,
                                                  'use_values' : True})
        if not data:
            continue
        var, df = data
    
        if model not in results:
            results[model] = {}
        lower_error = {}
        upper_error = {}
    
        distances, (lower, upper) = run_column_function(df, columns=['topic', 'question'],
                                                     function=opposite_context_sensitivity)
        results[model]['context'] = distances
        lower_error['context'] = lower
        upper_error['context'] = upper
    
        distances, (lower, upper) = run_column_function(df, columns=['topic', 'question'],
                                                     function=opposite_value_sensitivity)
        results[model]['value'] = distances
        lower_error['value'] = lower
        upper_error['value'] = upper

        errors[model] = np.array(list(lower_error.values())), np.array(list(upper_error.values()))
    
    data_df = pd.DataFrame(results)
    
    yerr = pd.DataFrame(errors).T.to_numpy()
    
    ax = data_df.plot.bar(ylim=YLIM,
                          title=f"sensitivity measures, {task}",
                             yerr=errors)
    
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=0)

def plot_average_value_change(df):
    '''
    Plots an ordered list of box plots for the by the maximum effect each value in the `df`
    has on the neutral answer.

    This function only really makes sense when we prompt with a limited set of values, 
    ie when using something like `data/wiki_controversial_small_schwartz.jsonl`
    '''
    value_distances = df.groupby(['topic', 'sample', 'question'])\
                       .apply(value_steerability)\
                       .reset_index(drop=True)
    value_distances['value text'] = value_distances['value'].apply(lambda x: x['text'])
    means = value_distances[['value text', 'distance']].groupby(['value text'])\
                                               .mean().reset_index().sort_values(by=['distance'])
    
    # These are only needed if we want to make the plot a histogram with error bars
    # intervals = value_distances[['value text', 'distance']].groupby(['value text'])\
    #                                            .apply(ci95).reset_index()\
    #                                            .loc[means['value text'].index][0] # for sorting
                                    
    # error = intervals_to_error(intervals)
    # means.plot.bar(x='value text', y='distance', yerr=error, ylim=(-1, 1))
    
    grouped = value_distances.groupby('value text')
    boxplot_data = [grouped.get_group(group_name)['distance'].values for group_name in means['value text']]
    
    # Setting labels to the categories in 'value text'
    labels = means['value text']
    
    # Drawing the box plot
    plt.boxplot(boxplot_data, labels=labels)
    plt.axhline(0, color='grey', linestyle='dotted')
    plt.ylabel('distance')
    plt.xticks(rotation=45)
    plt.ylim(-.5, 1)
    plt.title("Avg. change value induces from default answer; filtered high rephrase entropy")
    plt.show()

def plot_paired_boxplots(data, names):
    '''
    Plots the effect each value has in each of the dfs in `data`
    '''
    if len(data) < 1:
        logging.warn("No data passed")
        return
    fig_width = 6 if len(data) < 2 else 12
    fig, ax = plt.subplots(figsize=(fig_width, 6))  # Made the figure twice as wide

    colors = [matplotlib.colormaps['Set2'](x) for x in range(10)]  # Add more colors if needed
    # TODO: could make color a parameter
    legends = []

    for i, (boxplot_data, labels) in enumerate(data):
        boxplot = ax.boxplot(boxplot_data, positions=np.arange(i+1, len(boxplot_data)*(len(data)+1), len(data)+1),
                             widths=0.6, patch_artist=True)
        for box in boxplot['boxes']:
            box.set_facecolor(colors[i % len(colors)])  # Use color in a cyclic manner
        for median in boxplot['medians']:  # Made the "average" bar black
            median.set_color('black')

        legends.append(boxplot["boxes"][0])
        labels = labels

    # Setting labels to the categories in 'value text' only once for each pair
    plt.xticks(np.arange(1, len(labels)*(len(data)+1), len(data)+1), labels, rotation=45)

    # Adding a legend
    plt.legend(legends, names, loc='lower right')

def value_change_boxplot_process(df):
    value_distances = df.groupby(['topic', 'sample', 'question'])\
                       .apply(value_steerability)\
                       .reset_index(drop=True)
    grouping_column = 'text'
    if 'text_english' in df[df['value'].notnull()]['value'].iloc[0]:
        grouping_column = 'text_english'
    value_distances['value text'] = value_distances['value'].apply(lambda x: x[grouping_column])
    means = value_distances[['value text', 'distance']].groupby(['value text'])\
                                               .mean().reset_index().sort_values(by=['value text'])
    grouped = value_distances.groupby('value text')
    boxplot_data = [grouped.get_group(group_name)['distance'].values for group_name in means['value text']]
    return boxplot_data, means['value text']

# TODO: no longer used
def plot_average_value_change_across(var_dfs, legend_variable='model', title_suffix=""):
    data = []
    names = []
    for var, df in var_dfs:
        data.append(value_change_boxplot_process(df))
        name = shorten_var_name(var, legend_variable)
        names.append(name)
    plot_paired_boxplots(data, names,)
    
    plt.axhline(0, color='grey', linestyle='dotted')
    plt.ylabel('distance')
    plt.ylim(-.5, 1)
    plt.title(f"Avg. change value induces from default answer {title_suffix}; filtered high rephrase entropy")
    plt.show()

def plot_value_change_box_subplots(flat_var_dfs, legend_variable, colors=None):
    # Analyze the data
    data = apply_function_no_aggregation(flat_var_dfs, legend_variable, value_change_distances)

    # Make the plots
    nrows = 2
    ncols = math.ceil(len(data)/nrows)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 4),
                             sharey=True, squeeze=False)
    fig.suptitle('Avg. effect of each Schwartz value compared to default', fontsize=16)
    fig.subplots_adjust(hspace=0.5)
    bplots = []
    order = list(set(data.index) - {STATIC_BASELINE})
    if STATIC_BASELINE in data.index:
        order += [STATIC_BASELINE]
    for i in range(len(data.index)):
        idx = order[i]
        row =  i // ncols
        col = i % ncols
        ax = axes[row, col]
        bplot_data = data.loc[idx].values
        labels = data.loc[idx].index
        bplot = ax.violinplot(bplot_data, showmedians=True, widths=.5)
        ax.set_xticks([y + 1 for y in range(len(bplot_data))],
                      labels=labels)
        bplots.append(bplot)
        ax.set_title(idx)
        if col == 0:
            ax.set_ylabel('distance')
        if idx == STATIC_BASELINE:
            ax.spines['bottom'].set_color('red')
            ax.spines['top'].set_color('red')
            ax.spines['right'].set_color('red')
            ax.spines['left'].set_color('red')
    
    if colors is not None:
        for bplot in bplots:
            for patch, color in zip(bplot['bodies'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(.8)
            for part in ['cbars', 'cmins','cmaxes', 'cmedians']:
                bplot[part].set_color('black')
    
    # plt.savefig("figures/XX.pdf", bbox_inches='tight')
    plt.show()

#########
# Answer entropy
#########

def plot_answer_entropy_by_model(all_dfs, models, columns, task):
    '''
    For each model in `models` plots the data from `all_dfs` showing 
    the average entropy for each model grouped by `columns` (e.g. for
    topic and rephrase consistency)
    '''
    results = {}
    errors = {}
    for model in models:
        data = get_matching_data(all_dfs[model], {'task': task})
        if not data:
            continue
        var, df = data
        df = no_context_value(df)

        entropy = df.groupby(columns).apply(group_answer_entropy)
        results[model] = entropy
        intervals = df.groupby(columns).apply(group_answer_entropy_bootstrap)
        errors[model] = intervals
    
    yerr = errors_dict_reshape(errors)
    
    pd.DataFrame(results).plot.bar(yerr=yerr, ylabel="Shannon Entropy",
                                   title=f"Entropy of max answer for {task}, {columns} grouping")

def plot_answer_entropy_by_task(data, columns):
    '''
    For task in `data` caclulates the average entropy as grouped by `columns` and plots.
    '''
    results = {}
    errors = {}
    for var, df in data:
        df = no_context_value(df)

        entropy = df.groupby(columns).apply(group_answer_entropy)
        results[var['task']] = entropy
        intervals = df.groupby(columns).apply(group_answer_entropy_bootstrap)
        errors[var['task']] = intervals
    
    yerr = errors_dict_reshape(errors)
    
    pd.DataFrame(results).plot.bar(yerr=yerr, ylabel="Shannon Entropy",
                                   title=f"Entropy of max answer by task, {columns} grouping")

def plot_total_answer_entropy(all_dfs, models, task):
    '''
    Plots the average answer entropy for rephrase and topic for each model in `all_dfs`
    according to the order of `models` for the task, `task`
    '''
    results = {}
    errors = {}
    for model in models:
        data = get_matching_data(all_dfs[model], {'task': task})
        if not data:
            continue
        var, df = data
        df = no_context_value(df)
    
        results[model] = {}
        lower_error = {}
        upper_error = {}
    
        entropy = df.groupby(['topic', 'original']).apply(group_answer_entropy)
        results[model]['rephrase'] = entropy.mean()
        lower, upper = ci95(entropy)
        lower_error['rephrase'] = lower
        upper_error['rephrase'] = upper
    
        entropy = df.groupby(['topic']).apply(group_answer_entropy)
        results[model]['topic'] = entropy.mean()
        lower, upper = ci95(entropy)
        lower_error['topic'] = lower
        upper_error['topic'] = upper
    
        errors[model] = np.array(list(lower_error.values())), np.array(list(upper_error.values()))
    
    data_df = pd.DataFrame(results)
    yerr = pd.DataFrame(errors).T.to_numpy()
    
    ax = data_df.plot.bar(ylim=YLIM,
                          yerr=yerr,
                          title=f"avg. answer entropy for {task}")
    
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=0)

#######
# Degree of Support
######

def plot_topic_stance(var_dfs, legend_variable, stance, title_suffix=""):
    data = []
    names = []
    # NB: Have to make sure that both dfs have the same topics
    all_topics = set()
    for var, df in var_dfs:
        all_topics |= {*df['topic_english'].unique()}
    for var, df in var_dfs:
        support = topicwise_stance(df, stance)
        these_topics = {*support.index}
        for topic in all_topics - these_topics:
            support[topic] = np.nan
        supports = list(support)
        labels = list(support.index)
        data.append((supports, labels))
        name = shorten_var_name(var, legend_variable)
        names.append(name)
    plot_paired_boxplots(data, names)

    plt.ylabel(stance)
    plt.ylim(0, 1.05)
    plt.title(f"Avg. {stance}, {title_suffix}; filtered high rephrase entropy")
    plt.show()

def plot_topic_support(var_dfs, legend_variable, title_suffix=""):
    plot_topic_stance(var_dfs, legend_variable, 'supports', title_suffix)

def plot_topic_stances(var, df):
    data = []
    names = []
    stances = ['supports', 'opposes']
    if var['allow_abstentions']:
        stances +=  ['neutral']
    for stance in stances:
        support = topicwise_stance(df, stance)
        supports = list(support)
        labels = list(support.index)
        data.append((supports, labels))
        name = stance
        names.append(name)
    plot_paired_boxplots(data, names)

    plt.ylabel('percent support by stance')
    plt.ylim(0, 1.05)
    plt.title(f"Avg. response distributions for {var['model']}; filtered high rephrase entropy")
    plt.show()

def plot_stance_by_var(var_dfs, stance, legend_variable):
    results = []
    for var, df in var_dfs:
        # filter our values, group 
        df = no_context_value(df)
        results.append(df.apply(lambda x: x['distribution'][stance], axis=1)\
                       .rename(var[legend_variable]))
    
    pd.concat(results).plot.box()

def order_data_lists(row):
    # order (last to first): neither have entries, just one has entries, mean difference of entries
    order = sum([len(row[col]) > 0 for col in row.index])
    if order < 2:
        return order
    means = row.apply(np.mean)
    # return the sum of the differences from the mean
    return order + (means - means.mean()).abs().sum()

def plot_topic_support_box_subplots(flat_var_dfs, legend_variable, **kwargs):
    plot_violin_subplots(flat_var_dfs, legend_variable, ylabel='mean % support',
                         grouping_function=topicwise_support,
                         **kwargs)

####
# Generic plotting funcitons
####


def plot_violin_subplots(flat_var_dfs, legend_variable, grouping_function, labels=None,
                         order_function=order_data_lists,
                         title="", colors=None, fig=None, axes=None, ylabel='', ylim=None,
                         ncols=6, nplots=None, rotation=45):
    # Analyze the data
    data = apply_function_no_aggregation(flat_var_dfs, legend_variable, grouping_function)

    # Filter out the empty comparisons
    data = data[data.apply(row_cols_non_empty_lists, axis=1)]

    # Sort the data by the most divergent pairs first
    data['order'] = data.apply(order_function, axis=1)
    for col in data.columns:
        col_data = data[col]
        del data[col]
        data[str(col)] = col_data
    # data = data.sort_index(axis=1)
    data = data.sort_values(by='order', ascending=False)
    del data['order']

    # Make the plots
    fig_width = 10
    
    nplots = len(data) if nplots is None else nplots
    assert nplots <= len(data)
    
    nrows = math.ceil(nplots / ncols)
    height = nrows * 3

    title_chars_per_plot = (fig_width * TITLE_CHARS_PER_INCH) / ncols

    if axes is None:
        if fig is None:
            fig = plt.figure(figsize=(fig_width, height),)
            fig.subplots_adjust(hspace = 0.5)
            fig.suptitle(t=title)
        elif title:
            wrapped_title = "\n".join(textwrap.wrap(title, title_chars_per_plot))
            fig.suptitle(t=wrapped_title, x=.95, y=.7, fontsize="x-large", fontweight="bold")
        
        fig.subplots_adjust(left=0.0, right=1)
    
        axes = fig.subplots(ncols=ncols, nrows=nrows, sharey=True, squeeze=False)
    else:
        ncols = len(axes[0]) if type(axes[0]) == np.ndarray else len(axes)

    bplots = []
    order = data.index

    flattened_axes = axes.flatten()
    for i in range(nplots):
        ax = flattened_axes[i]
        idx = order[i]
        bplot_data = data.loc[idx].values
        if labels is None:
            labels = data.loc[idx].index 
        bplot = ax.violinplot(bplot_data, showmedians=True, widths=.5)
        bplots.append(bplot)

        wrapped_title = "\n".join(textwrap.wrap(idx, title_chars_per_plot))
        ax.set_title(wrapped_title)
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(ylim)

        ax.set_xticks([y + 1 for y in range(len(bplot_data))],
                      labels=labels, rotation=rotation)

    if fig is not None:
        for i in range(nplots, nrows * ncols, 1):
            row =  i // ncols
            col = i % ncols
            ax = axes[row, col]
            ax.axis('off')

    if colors is not None:
        for bplot in bplots:
            for patch, color in zip(bplot['bodies'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(.8)
            for part in ['cbars', 'cmins','cmaxes', 'cmedians']:
                bplot[part].set_color('black')


def plot_measure_by_dfs_filter(var_dfs, measure, title='',
                               nested=False,
                                var_name='model',
                                colors=None,
                                ax=None):
    results = {}
    
    lower_error = {}
    upper_error = {}

    var_dfs = [item for item in var_dfs for _ in range(2)]

    for i in range(len(var_dfs)):
        dfs = []
        if not nested:
            var, df = var_dfs[i]
            dfs = [df]
        else:
            if len(var_dfs[i]) < 2:
                continue
            (var, df), (var2, df2) = var_dfs[i]
            dfs = [df, df2]

        key = shorten_var_name(var, var_name)

        if i % 2 == 0:
            subkey = "filtered"
            dfs = [filter_high_entropy_rephrase(df) for df in dfs]
        else:
            subkey = "raw"

        if subkey not in results:
            results[subkey] = {}
            lower_error[subkey] = {}
            upper_error[subkey] = {}

        distance, (lower, upper) = measure(*dfs)

        results[subkey][key] = distance
        lower_error[subkey][key] = lower
        upper_error[subkey][key] = upper

        # errors[key] = np.array(list(lower_error.values())), np.array(list(upper_error.values()))

    data_df = pd.DataFrame(results)

    lower_error_values = [list(val.values()) for val in lower_error.values()]
    upper_error_values = [list(val.values()) for val in upper_error.values()]
    
    # combine and reshape
    errors = [lower_error_values, upper_error_values]

    if colors is not None:
        assert len(colors) >= len(var_dfs)
        colors = colors[0:len(var_dfs)]
    
    ax = data_df.plot.bar(ylim=YLIM,
                          yerr=errors,
                          ylabel='',
                          title=title,
                          color=colors, 
                          ax=ax)
    
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=0)

def plot_measure_by_dfs(var_dfs, measure, title='',
                       nested=False,
                       var_name='model',
                       use_filter=False,
                       plot_bound=False,
                       ax=None):
    
    results = {}
    errors = {}
    colors = []
    for i in range(len(var_dfs)):
        dfs = []
        if not nested:
            var, df = var_dfs[i]
            dfs = [df]
        else:
            if len(var_dfs[i]) < 2:
                continue
            dfs = []
            for (var, df) in var_dfs[i]:
               dfs.append(df)
        colors.append(ALL_MODEL_COLORS[var['model']])
        key = shorten_var_name(var, var_name)
        
        if use_filter:
            dfs = [filter_high_entropy_rephrase(df) for df in dfs]
    
        lower_error = {}
        upper_error = {}
    
        distance, (lower, upper) = measure(*dfs)
    
        results[key] = distance
        lower_error[key] = lower
        upper_error[key] = upper
    
        errors[key] = np.array(list(lower_error.values())), np.array(list(upper_error.values()))
    
    data_df = pd.DataFrame(list(results.items()),columns = ['Label','Value'])
    display(data_df)


    yerrs = [[val[0][0] for val in errors.values()], [val[1][0] for val in errors.values()]] 
    
    ax = data_df.plot.bar(x='Label',
                          y='Value',
                          ylim=YLIM,
                          yerr=yerrs,
                          ylabel=INCONSISTENT_MEAURE_LABEL,
                          xlabel='',
                          title=title,
                          ax=ax,
                          color=colors)

    var = var_dfs[0][0][0] if nested else var_dfs[0][0]
    if plot_bound:
        plot_upper_bound(ax, var=var)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=55)
    return ax

####### Group topicwise

def order_means_ascending(row):
    return row.apply(np.mean).mean()

def order_means_descending(row):
    return -order_means_ascending(row)

def plot_topic_consistency_grouped(flat_var_dfs, legend_variable, order_function=order_means_ascending, **kwargs):
    plot_violin_subplots(flat_var_dfs, legend_variable, ylabel=INCONSISTENT_MEAURE_LABEL,
                         grouping_function=topic_consistency_grouped, order_function=order_function, ylim=(0, .8),
                         **kwargs)

def plot_rephrase_consistency_grouped(flat_var_dfs, legend_variable, order_function=order_means_ascending, **kwargs):
    plot_violin_subplots(flat_var_dfs, legend_variable, ylabel=INCONSISTENT_MEAURE_LABEL,
                         grouping_function=rephrase_consistency_grouped, order_function=order_function, ylim=(0, .8),
                         **kwargs)

########
# Robustness checks
########

def plot_answer_logprob_distribution(var, df, title="", ax=None):
    if 'answer_logprobs' not in df.columns or df['answer_logprobs'].isnull().all():
        raise ValueError("Called on a run that did not save log probs")
    num_options = 3 if var['allow_abstentions'] or ('annotator_abstentions' in var and var['annotator_abstentions']) else 2
    answers = list(OPTIONS)[0:num_options] # should be greater if a neutral answer is allowed
    logprobs = df['answer_logprobs'].apply(get_highest_option_mass_distribution,
                                           num_options=num_options,)\
                                    .apply(filter_distribution_for_answers,
                                           num_options=num_options,
                                           weight_non_answers=True)

    results = []
    for answer in answers + [None]:
        results.append(logprobs.apply(lambda x: x[answer] if answer in x else 0).rename(str(answer)))
    pd.concat(results, axis=1).plot.box(ylim=(0, 1.05),
                                        ax=ax,
                                        ylabel="Probability",
                                        title=title
                                        )

def plot_top_token_logprobs(var, df):
    if 'answer_logprobs' not in df.columns:
        raise ValueError("Called on a run that did not save log probs")
    num_options = 3 if var['allow_abstentions'] else 2
    distributions = df['answer_logprobs'].apply(get_highest_option_mass_distribution,
                                                num_options=num_options, as_option=False)
    combined_distribution = distributions.sum().normalize()
    pd.Series(combined_distribution)\
      .sort_values(ascending=False)[0:20]\
      .plot.bar(title="top 20 tokens from logprobs", ylabel='probability', xlabel='token')

def plot_robustness_checks(var, df):
    name = MODEL_NAMES_SHORT[var['model']]
    name += f", {var['task']}"
    if var['task'] == 'generation':
        name += f" ({var['annotator']})"
    name += f", {data_filename(var['filename'])}"
    name += ', abstain' if 'allow_abstentions' in var and var['allow_abstentions'] else ''

    print(name)

    df['text'].apply(get_option).value_counts()\
             .plot.bar(title="Counts of chosen answer (arg max stripped from first token letter)")

    plt.show()

    if len(df[df['answer_logprobs'].notnull()]) > 0:
        plot_top_token_logprobs(var, df)

        plt.show()
        if 'example_answer' in df.columns:
            (df['example_answer'] == df['text'].apply(get_option)).value_counts()\
                                        .plot.bar(title=" ex. ans. aggreement")
            plt.show()

        plot_answer_logprob_distribution(var, df, title=name)

        plt.show()
