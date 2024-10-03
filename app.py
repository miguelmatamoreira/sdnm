import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from itertools import chain
import plotly.express as px
from gene_centric import data_preprocessing as dp 
from gene_centric import statistical_tests as st
from pattern_centric import problem_mapping as pm
from pattern_centric import biclustering as bc
from pattern_centric import triclustering as tc
from dash.dependencies import Input, Output
import math





#######################################################################################################################################
# GLOBAL VARIABLES                                                     
#######################################################################################################################################
# define the information retrieval genes
with open("data/ir/filtered_genes_hs.txt", "r") as f:
    genes_read = [line.strip() for line in f.readlines()]
    genes_ir = sorted(genes_read)
# define the dopamine genes
genes_dopa = sorted([
    "SLC18A2", "DRD2", "DRD3", "SLC6A2", "DBH", 
    "COMT", "LRTOMT", "DRD5", "SLC6A3", "TH", "DRD4"
])
# define the gaba genes
genes_gaba = sorted([
    "GABBR2", "GABRA4", "GABRA6", "GABRB2", "GABRG1", "GABRB1", "GABRB3",
    "GABRA3", "GABRE", "GABBR1", "GABRD", "GABRA5", "GABRR2", "GABRA1",
    "GABRG2", "GABRR1", "GPR156", "GABRA2", "GABRP", "GABRQ", "GABRG3", "GABRR3"
])
# define the genes ir as default
genes = genes_ir

# define the brain tissues
brain_tissues = ["Amygdala", "Anterior cingulate cortex (BA24)", "Caudate (basal ganglia)", "Cerebellar Hemisphere", 
                "Cerebellum", "Cortex", "Frontal Cortex (BA9)", "Hippocampus", "Hypothalamus", "Nucleus accumbens (basal ganglia)",
                "Putamen (basal ganglia)", "Spinal cord (cervical c-1)", "Substantia Nigra", "Pituitary"]
# define the peripheral tissues
peripheral_tissues = ["Whole Blood", "Skin (Sun Exposed)", "Adipose (Visceral)", "Skin (Not Sun Exposed)"]
# define the tissues as the sum of brain and peripheral tissues
tissues = brain_tissues + peripheral_tissues

# get the values from information retrieval genes
values_ir, tissue_samples_ir, tissue_indiv_ir = dp.get_values(genes_ir, tissues)
# get the values from dopamine and gaba genes
values_string, tissue_samples_string, tissue_indiv_string = dp.get_values(genes_dopa + genes_gaba, tissues)
# select the ir system as default
values, tissue_samples, tissue_indiv = values_ir, tissue_samples_ir, tissue_indiv_ir

# get the samples
samples = dp.get_samples_dataset()





#######################################################################################################################################
# GLOBAL VARIABLES - GENE TAB                                                
#######################################################################################################################################
# define the possible colors for the plots
colors = ['darkblue', 'darkgreen', 'darkorange', 'darkred', '#D789D8'] #, '#654321']
# define the table for genes
df1 = pd.DataFrame()
# define the table for tissues
df2 = pd.DataFrame()
# p-values dictionary
p_values_dict = {}
# colors table genes
colors_table_genes = ['rgb(200,225,204)', 'rgb(186,255,201)', 'rgb(255,255,186)', 'rgb(255,223,186)', 'rgb(255,179,186)']
# selected tissues and data global for the table pop-ups
selected_tissues_global = []
selected_data_global = {}
# information for pop-up expressions
result_expressions = {}
# ir or string selected
ir_selected = True





#######################################################################################################################################
# GLOBAL VARIABLES - PATTERN TAB                                              
#######################################################################################################################################
# define the table for clusters found
df3 = pd.DataFrame()
# information for pop-up clusters
cluster_information = {}
# tissues selected in pattern tab
pattern_tissues = []
# algorithm chosen
pattern_alg = ""
# invididuals selected in triclustering
pattern_indiv = []
# cube for triclustering
pattern_cube = []
# for the tissue context dropdown
samples_tricluster_atm = []
samples_tricluster_atm_idx = []
genes_tricluster_atm = []
genes_tricluster_atm_idx = []
tissues_tricluster_atm = []
tissues_tricluster_atm_idx = []





#######################################################################################################################################
# AUXILARY FUNCTIONS - GENE TAB                                           
#######################################################################################################################################
def decide_expression_conclusion(selected_tissues, selected_tissues_f, heatmap_values, selected_gene):
    '''
    Decide the expression conclusion based on the number of input and output tissues selected.
    We need to do the maximum of each sublist and the minimum of the final, [[1, 2], [3, 4]] -> [2, 4] -> 2.

    Parameters:
    - selected_tissues (list of 2 lists): Contains the tisses selected in input and output.
    - selected_tissues_f (list): Represents the selected_tisues flatten.
    - heatmap_values (numpy array): Contains the p-values retrieved from the statistical study.
    - selected_gene (str): Gene selected by the user.

    Returns:
    - footnote (str): Text that describes the decision.
    - overall_minimum (float): Final result of p-value.
    '''
    footnote = ""
    overall_minimum = 0
    if not selected_tissues[0] or not selected_tissues[1]:
        footnote = "<b>No conclusion with the filters selected</b>"
        return footnote, overall_minimum
    if len(selected_tissues[0]) > 0 and len(selected_tissues[1]) > 0:
        values = []
        for input_tissue in selected_tissues[0]:
            values_input = []
            for output_tissue in selected_tissues[1]:
                v = heatmap_values[selected_tissues_f.index(input_tissue)][selected_tissues_f.index(output_tissue)]
                values_input.append(v)
            values.append(values_input)

        # add information to dict
        p_values_dict[selected_gene] = [item for sublist in values for item in sublist]

        # find the maximum in each sublist
        max_values = [max(sublist) for sublist in values]
        # find the minimum in the final
        overall_minimum = min(max_values)
        if overall_minimum < 0.05:
            footnote = f"The <b>input tissue set</b> and <b>output tissue set</b> are <b>differently expressed</b> (p-value = {overall_minimum})"
        else:
            footnote = f"<b>No conclusion</b> (p-value = {overall_minimum})"
        return footnote, overall_minimum
    return footnote, overall_minimum



def draw_plot_one_gene(selected_data, selected_gene, selected_tissues, selected_profile):
    '''
    Draws the heatmap and the kde (when one gene is selected).

    Parameters:
    - selected_data (list of 2 lists of tuples): It is selected_data = [([expressions], [samples]), ([expressions], [samples]), ...].
    - selected_gene (str): Gene selected by the user.
    - selected_tissues (list of 2 lists): List of tissues selected by the user [[input tissues], [output tissues]].
    - selected_profile (list of str): Profiles selected by the user.

    Returns:
    - fig (figure): Figure to be displayed, heatmap + kde.
    '''
    # create subplots
    delay = 0
    fig = make_subplots(rows=1, cols=2)
    selected_tissues_f = list(chain(*selected_tissues))
    selected_data_f = list(chain(*selected_data))
    tissues_not_considered = []
    traces_none = 0

    # create the histogram trace
    traces = []
    for i, (expression, sample) in enumerate(zip(selected_data_f, selected_tissues_f)):
        # print(f"Number of samples {i} - {len(expression[0])}")
        # print(expression[0])
        # not enough values to construct the kde line
        if len(expression[0]) <= 3 or (all(element == 0 for element in expression[0])):
            delay += 1
            tissues_not_considered.append(i)
            continue
        # create histogram trace
        # if i == 5:
        #    hist_trace = go.Histogram(x=expression[0], nbinsx=50, name=f'{sample}', histnorm='probability density', opacity=0.75, legendgroup=f'group{i}', marker=dict(color='brown'))
        # else:
        hist_trace = go.Histogram(x=expression[0], nbinsx=50, name=f'{sample}', histnorm='probability density', opacity=0.75, legendgroup=f'group{i}')
        traces.append(hist_trace)
        # create kde line trace
        kde_line = ff.create_distplot([expression[0]], group_labels=[selected_gene], bin_size=50, show_rug=False)
        kde_trace = kde_line['data'][1]
        # print((i-delay)%5)
        kde_trace['line']['color'] = colors[(i-delay)%5]
        kde_trace['legendgroup'] = f'group{i}'
        traces.append(kde_trace)

    # handle tissues that do not have representation
    tissues_considered_f = [value for index, value in enumerate(selected_tissues_f) if index not in tissues_not_considered]
    # print(f"depois de filtrado {tissues_considered_f}")
    dif = set(selected_tissues_f).difference(set(tissues_considered_f))
    # print(f"O dif {dif}")
    data_considered_f = [value for index, value in enumerate(selected_data_f) if index not in tissues_not_considered]
    for el in dif:
        for sublist in selected_tissues:
            if el in sublist:
                sublist.remove(el)
    # print(f"os selected tissues - {selected_tissues}")

    # create the heatmap trace
    heatmap_values = get_values_heatmap(tissues_considered_f, data_considered_f)
    p_values_text = [[f'p-value = {value}' for value in row] for row in heatmap_values]
    heatmap_trace = go.Heatmap(z=heatmap_values, x=tissues_considered_f, y=tissues_considered_f, text=p_values_text, colorscale='Greens_r', zmin=0, zmax=1, legendgroup='', showscale=False, hoverinfo='text')

    # footnote based on the number of tissues selected
    footnote_text, _ = decide_expression_conclusion(selected_tissues, tissues_considered_f, heatmap_values, selected_gene)
    annotation = dict(
        xref='paper', yref='paper',
        x=0.5, y=1.1,
        xanchor='center', yanchor='bottom',
        text=footnote_text,
        showarrow=False,
        font=dict(size=13, family="Consolas, monospace"),
    )

    # to be more concise when the gene is the only dropdown selected 
    if (len(tissues_considered_f) == 0 and not selected_profile):
        fig.add_trace(heatmap_trace, row=1, col=1)
        fig.add_trace(go.Scatter(), row=1, col=2)
        text = "The tissues ["
        for el in selected_tissues_f:
            text += f"{el}, "
        text = text[:-2]
        text += "] are not being considered due to an insufficient number of samples for representation"
        return fig, dbc.Alert(text, color="danger", style={'font-family': 'Consolas, monospace', 'font-size': '13px', 'marginBottom': 0})

    # add histogram and heatmap trace to subplot
    if len(traces) != 0:
        for trace in traces:
            if 'type' in trace and trace['type'] == 'histogram':
                fig.add_trace(trace, row=1, col=2)
            else:
                fig.add_trace(trace, row=1, col=2)
        fig.add_trace(heatmap_trace, row=1, col=1)
    else:
        traces_none = 1
    fig.update_layout(showlegend=True)
    fig.update_xaxes(title_text='Gene Expression Value', title_font=dict(size=12), row=1, col=2)
    fig.update_yaxes(title_text='Probability Density', title_font=dict(size=12), row=1, col=2)
    fig.update_layout(
        annotations=[annotation],
        font=dict(family="Consolas, monospace")
    )

    if list(dif):
        text = "The tissues ["
        for el in dif:
            text += f"{el}, "
        text = text[:-2]
        text += "] are not being considered due to an insufficient number of samples for representation"
        if traces_none == 0:
            return fig, dbc.Alert(text, color="danger", style={'font-family': 'Consolas, monospace', 'font-size': '13px', 'marginBottom': 0})
        else:
            return None, dbc.Alert(text, color="danger", style={'font-family': 'Consolas, monospace', 'font-size': '13px', 'marginBottom': 0})
    if traces_none == 0:
        return fig, None
    else:
        return None, None



def draw_plot_multiple_gene(selected_data, selected_genes, selected_tissues):
    '''
    Draws the tables (when more that one gene is selected).

    Parameters:
    - selected_data (dict that has list of 2 lists of tuples): It is selected_data[gene] = [([expressions], [samples]), ([expressions], [samples]), ...].
    - selected_genes (list of str): Genes selected by the user.
    - selected_tissues (list of 2 lists): List of tissues selected by the user [[input tissues], [output tissues]].

    Returns:
    - table1, table2 (dash_table.DataTable): Tables to be displayed.
    '''
    # empty, no data at all
    if all(value == [[([], [])], [([], [])]] for value in selected_data.values()):
        return None, None
    # if gene is not representative enough in one tissue do not consider it
    for gene, value in selected_data.items():
        # reset
        selected_tissues_input = len(selected_tissues[0])
        selected_tissues_output = len(selected_tissues[1])
        for t in value[0]:
            if all(element == 0 for element in t[0]):
                selected_tissues_input -= 1
        for t in value[1]:
            if all(element == 0 for element in t[0]):
                selected_tissues_output -= 1
        # no representation for the gene to be presented
        if selected_tissues_input == 0 or selected_tissues_output == 0:
             # print(f"removed {gene}")
             selected_genes.remove(gene)
        for t in value[0]+value[1]:
            if len(t[0]) <= 3:
                selected_genes.remove(gene)
                break
    # print(selected_genes)
    if not selected_genes:
        return None, None

    # data for the table genes
    gene_names = [gene for gene in selected_genes]
    # print(gene_names)
    scores_genes = []
    count_genes = np.zeros(len(selected_genes))
    for gene in selected_genes:
        selected_tissues_f = list(chain(*selected_tissues))
        selected_data_gene_f = list(chain(*selected_data[gene]))
        heatmap_values_gene = get_values_heatmap(selected_tissues_f, selected_data_gene_f)
        _, overall_minimum = decide_expression_conclusion(selected_tissues, selected_tissues_f, heatmap_values_gene, gene)

        scores_genes.append(overall_minimum)

    # data for the table matrix
    results = np.zeros((len(selected_tissues[0]), len(selected_tissues[1])))
    # print(f"Results: {results}")
    # print(f"tissues1 {len(selected_tissues[0])} - tissues2 {len(selected_tissues[1])}")
    for g, gene in enumerate(selected_genes):
        for i, tissue1 in enumerate(selected_tissues[0]):
            # print(f"i={i}-{tissue1}")
            for j, tissue2 in enumerate(selected_tissues[1]):
                # print(f"j={j}-{tissue2}")
                p_value = st.statistical_study(selected_data[gene][0][i][0], selected_data[gene][1][j][0])
                # print(f"p-value between {tissue1}-{tissue2}={p_value}")
                if p_value < 0.05:
                    results[i][j] += 1
                    # print(f"results atm:\n{results}")
                    # print(f"Results in the last if in i={i} and j={j}:\n{results}")
                    count_genes[g] += 1
    # for the column tissue expressions of the table genes
    global brain_tissues
    global peripheral_tissues

    # define the table for genes
    global df1
    df1 = pd.DataFrame({
        'Selected Genes': gene_names,
        'Occurrences': count_genes,
        'Scores (p-values)': scores_genes,
    })
    global result_expressions
    result_expressions = {}
    for gene in gene_names:
        for t, tissue in zip(selected_data[gene][0]+selected_data[gene][1], selected_tissues[0]+selected_tissues[1]):
            mean = np.mean(t[0])
            std = np.std(t[0])
            # print(f"For gene-{gene}, tissue-{tissue}: mean={mean}, std={std}")
            if gene not in result_expressions:
                result_expressions[gene] = [(tissue, mean, std)]
            else:
                result_expressions[gene].append((tissue, mean, std))
    # print(result_expressions)
    df1["Tissue Expressions"] = "More details here..."
    df1 = df1.sort_values(by=['Scores (p-values)', 'Selected Genes'], ascending=[True, True])
    table1 = dash_table.DataTable(
        id='table1',
        columns=[{"name": i, "id": i} for i in df1.columns],
        data=df1.to_dict('records'),
        page_size=10,
        page_current=0,
        style_cell={'textAlign': 'center', 'fontSize': '13px', 'cursor': 'pointer'},  
        row_selectable=False,
        style_data_conditional=[
            {
                'if': {'filter_query': '{Scores (p-values)} < 0.001'},
                'backgroundColor': colors_table_genes[0],
                'color': 'black'
            },
            {
                'if': {'filter_query': '{Scores (p-values)} >= 0.001 and {Scores (p-values)} < 0.01'},
                'backgroundColor': colors_table_genes[1],
                'color': 'black'
            },
            {
                'if': {'filter_query': '{Scores (p-values)} >= 0.01 and {Scores (p-values)} < 0.05'},
                'backgroundColor': colors_table_genes[2],
                'color': 'black'
            },
            {
                'if': {'filter_query': '{Scores (p-values)} >= 0.05 and {Scores (p-values)} < 0.10'},
                'backgroundColor': colors_table_genes[3],
                'color': 'black'
            },
            {
                'if': {'filter_query': '{Scores (p-values)} >= 0.10'},
                'backgroundColor': colors_table_genes[4],
                'color': 'black'
            },
            {
                'if': {'state': 'active'},
                'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                'border': '1px solid rgba(0, 116, 217, 0.3)'
            }
        ],
        style_cell_conditional = [
                {
                    'if': {'column_id': c},
                    'minWidth': '250px' 
                } for c in df1.columns
        ],
    )

    # define the table for tissues
    global df2
    df2 = pd.DataFrame({
        'Input - Output': selected_tissues[0],
    })
    for output_tissue in selected_tissues[1]:
        df2[output_tissue] = None
    # print(results)
    for row_index, row_values in enumerate(results):
        df2.iloc[row_index, 1:] = row_values 
    table2 = dash_table.DataTable(
        id='table2',
        columns=[{"name": i, "id": i} for i in df2.columns],
        data=df2.to_dict('records'),
        style_table={'overflowX': 'auto', 'maxWidth': '100%', 'minWidth': '100%'},
        editable=False,
        style_cell={'textAlign': 'center', 'fontSize': '13px'},  
        row_selectable=False,
        
        style_data_conditional=[
            {
                'if': {'column_id': df2.columns[0]},
                'backgroundColor': 'rgb(248, 248, 248)'
            },
            {
                'if': {'state': 'active'},
                'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                'border': '1px solid rgba(0, 116, 217, 0.3)'
            }
        ],
        fixed_columns={'headers': True, 'data': 1},
        fixed_rows={'headers': True},
        style_cell_conditional = [
                {
                    'if': {'column_id': c},
                    'minWidth': '250px' 
                } for c in df2.columns
        ],
    )
    return table1, table2



def get_profile_values(selected_data, selected_profile):
    '''
    Calculates the correct data given the set of selected profiles.
    First, it performs the union of the same characteristics, then it does the intersection of different ones.

    Parameters:
    - selected_data (list of 2 lists of tuples): Each least is [([expressions], [samples]), ([expressions], [samples]), ...].
    - selected_profile (list of str): List of profiles selected by the user.

    Returns:
    - final_result (list of 2 lists of tuples): Contains the new for each list [([expressions], [samples])].
    '''
    final_result = [[], []]
    for k in range(2):
        for tissue in selected_data[k]:
            indices_to_add_sex = []
            indices_to_add_age = []
            indices_to_add_agonal = []
            for profile in selected_profile:
                for i, (_, sample) in enumerate(zip(tissue[0], tissue[1])):
                    sample = samples.loc[samples["sample"] == sample]
                    sample_values = sample[['sex', 'age', 'agonal_state']].values.tolist()[0]
                    # match each profile
                    if profile == "sex-male" and sample_values[0] == 1:
                        indices_to_add_sex.append(i)
                    elif profile == "sex-female" and sample_values[0] == 2:
                        indices_to_add_sex.append(i)
                    elif profile == "age-20-29" and sample_values[1] == "20-29":
                        indices_to_add_age.append(i)
                    elif profile == "age-30-39" and sample_values[1] == "30-39":
                        indices_to_add_age.append(i)
                    elif profile == "age-40-49" and sample_values[1] == "40-49":
                        indices_to_add_age.append(i)
                    elif profile == "age-50-59" and sample_values[1] == "50-59":
                        indices_to_add_age.append(i)
                    elif profile == "age-60-69" and sample_values[1] == "60-69":
                        indices_to_add_age.append(i)
                    elif profile == "age-70-79" and sample_values[1] == "70-79":
                        indices_to_add_age.append(i)
                    elif profile == "agonal-0" and sample_values[2] == 0:
                        indices_to_add_agonal.append(i)
                    elif profile == "agonal-1" and sample_values[2] == 1:
                        indices_to_add_agonal.append(i)
                    elif profile == "agonal-2" and sample_values[2] == 2:
                        indices_to_add_agonal.append(i)
                    elif profile == "agonal-3" and sample_values[2] == 3:
                        indices_to_add_agonal.append(i)
                    elif profile == "agonal-4" and sample_values[2] == 4:
                        indices_to_add_agonal.append(i)
            # create a list of lists
            indices_to_add = [indices_to_add_sex, indices_to_add_age, indices_to_add_agonal]
            # filter the empty lists
            non_empty_indices_to_add = [list for list in indices_to_add if list]
            # check if there are any non-empty lists
            if non_empty_indices_to_add:
                # find the intersection using set intersection
                indices_to_add = list(set(non_empty_indices_to_add[0]).intersection(*non_empty_indices_to_add[1:]))
                profile_selected_values = ([], [])
                # add the final result
                for index in indices_to_add:
                    profile_selected_values[0].append(tissue[0][index])
                    profile_selected_values[1].append(tissue[1][index])
            final_result[k].append(profile_selected_values)
    return final_result



def get_values_heatmap(selected_tissues, selected_data):
    '''
    Returns the p-values for the selected data to display in the heatmap.

    Parameters:
    - selected_tissues (list of str): Tissues selected by the user flatten.
    - selected_data (list of tuples): Contains the [([expressions], [samples]), ([expressions], [samples]), ...] flatten.

    Returns:
    - results_statistical_study (numpy array): Contains the p-values for tissue-tissue for the gene selected.
    '''
    results_statistical_study = np.ones((len(selected_tissues), len(selected_tissues)))
    # p-value calculations
    for i, _ in enumerate(selected_tissues):
        for j, _ in enumerate(selected_tissues):
            # print(f"Tissue1-{tissue1}, Tissue2-{tissue2}")
            p_value = 0
            # avoid redundancy + shapiro only works with 3 or more samples
            if i < j and len(selected_data[i][0]) >= 3 and len(selected_data[j][0]) >= 3:
                p_value = st.statistical_study(selected_data[i][0], selected_data[j][0])
                p_value = p_value.round(10)
                results_statistical_study[i][j] = p_value
                results_statistical_study[j][i] = p_value
                # print(f"p_value: {p_value} for {tissue_i}-{tissue_j}")
    # print(results_statistical_study)
    return results_statistical_study



def popup_score(gene):
    '''
    Create the pop-up boxplot for the given gene.

    Parameters:
    - gene (str): Gene selected.

    Returns:
    - dcc.Graph (dcc.Graph): Figure to be displayed.
    '''
    fig = px.box(y=p_values_dict[gene], labels={'y': 'p-value'})
    fig.update_layout(
        font=dict(size=12, family="Consolas, monospace"),
        yaxis=dict(title_font=dict(size=12))
    )
    return dcc.Graph(figure=fig)



def popup_gene(gene):
    '''
    Create the pop-up histogram + kde for the given gene.

    Parameters:
    - gene (str): Gene selected.

    Returns:
    - dcc.Graph (dcc.Graph): Figure to be displayed.
    '''
    # define the figure
    fig = go.Figure()
    global selected_tissues_global
    global selected_data_global
    selected_tissues_f = list(chain(*selected_tissues_global))
    selected_data_f = list(chain(*selected_data_global[gene]))
    delay = 0
    # create the histogram trace
    traces = []
    for i, (expression, sample) in enumerate(zip(selected_data_f, selected_tissues_f)):
        # not enough values to construct the kde line
        if len(expression[0]) <= 3 or (all(element == 0 for element in expression[0])):
            delay += 1
            continue
        # create histogram trace
        #if i == 5:
            #hist_trace = go.Histogram(x=expression[0], nbinsx=50, name=f'{sample}', histnorm='probability density', opacity=0.75, legendgroup=f'group{i}', marker=dict(color='brown'))
        #else:
        hist_trace = go.Histogram(x=expression[0], nbinsx=50, name=f'{sample}', histnorm='probability density', opacity=0.75, legendgroup=f'group{i}')
        traces.append(hist_trace)
        # create kde line trace
        kde_line = ff.create_distplot([expression[0]], group_labels=[gene], bin_size=50, show_rug=False)
        kde_trace = kde_line['data'][1]
        kde_trace['line']['color'] = colors[(i-delay)%5]
        kde_trace['legendgroup'] = f'group{i}'
        traces.append(kde_trace)
    for trace in traces:
        fig.add_trace(trace)
    fig.update_layout(font=dict(family="Consolas, monospace"))
    fig.update_xaxes(title_text='Gene Expression Value', title_font=dict(size=12))
    fig.update_yaxes(title_text='Probability Density', title_font=dict(size=12))
    return dcc.Graph(figure=fig)



def popup_expressions(gene):
    '''
    Create the pop-up text for the tissue expressions.

    Parameters:
    - gene (str): Gene selected.

    Returns:
    - dcc.Markdown(text) (dcc.Markdown): Text to be displayed.
    '''
    tissues = []
    text = ""
    global result_expressions
    for t in result_expressions[gene]:
        if t[0] not in tissues:
            tissues.append(t[0])
            text += f"**{t[0]}**  \nμ={t[1]}, σ={t[2]}\n\n"
    return dcc.Markdown(text)





#######################################################################################################################################
# CALLBACK FUNCTIONS - GENE TAB                                                    
#######################################################################################################################################
@callback(
    [Output("tissue1-dropdown", 'value'), Output("tissue2-dropdown", 'value'), Output("gene-dropdown", 'value', allow_duplicate=True), Output("profile-dropdown", 'value'), Output("gene-dropdown", "options")],
    Input('ir-checkbox', 'value'),
    prevent_initial_call=True
)
def update_output(checked_values):
    global values
    global tissue_samples
    global tissue_indiv

    global values_ir
    global tissue_samples_ir
    global tissue_indiv_ir 
    global genes_ir
    
    global values_string
    global tissue_samples_string
    global tissue_indiv_string
    global genes_dopa
    global genes_gaba

    global ir_selected

    # add "dopaminergic system" and "GABAergic system to dropdowns"
    if 'checked' in checked_values:
        ir_selected = False
        values = values_string
        tissue_samples = tissue_samples_string
        tissue_indiv = tissue_indiv_string
        options_genes = [{'label': gene, 'value': gene} for gene in ["All genes", "Genes from dopaminergic system", "Genes from GABAergic system"] + genes_dopa + genes_gaba]
    # use the ir system
    else:
        ir_selected = True
        values = values_ir
        tissue_samples = tissue_samples_ir
        tissue_indiv = tissue_indiv_ir
        options_genes = [{'label': gene, 'value': gene} for gene in ["All genes"] + genes_ir]
    return None, None, None, None, options_genes



@callback(
    dash.dependencies.Output('tablegene-popup', 'children'),
    [dash.dependencies.Input('table1', 'active_cell'), dash.dependencies.Input('table1', 'page_current')],
    prevent_initial_call=True
)
def display_popups_table_genes(active_cell, page_current):
    '''
    For the table genes, displays a pop-up depending on the cell that is active.

    Parameters:
    - active_cell (active_cell): Cell that is active at the moment.

    Returns:
    - tablegene-popup (children): Pop-up to appear.
    '''
    global df1
    df1 = df1.sort_values(by=['Scores (p-values)', 'Selected Genes'], ascending=[True, True])
    # print(df1.head())
    if active_cell:
        row = active_cell['row']
        # print(f"ROW-{row}")
        # print(f"value-{row+10*page_current}")
        col = active_cell['column_id']
        # print(f"COL-{col}")
        value = df1.iloc[row][col]
        # print(f"VALUE-{value}")
        gene = df1.iloc[active_cell['row']+(10*page_current)]['Selected Genes']
        # print(f"GENE-{gene}")
        # print(f"Current Page: {page_current}")
        # pop-up gene histogram + kde
        if col == "Selected Genes":
            return dbc.Modal(
                    [
                        dbc.ModalHeader(f"Histogram for {gene}", style={"fontFamily": "Consolas, monospace"}),
                        dbc.ModalBody(popup_gene(gene)),
                    ],
                    id="modal",
                    is_open=True,
                    centered=True,
                    backdrop="static",
                    scrollable=True,
                    class_name={"max-width": "1200px", "width": "80%"}
                )
        # pop-up p-value boxplot
        elif col == "Scores (p-values)":
            if len(p_values_dict[gene]) >= 3:
                return dbc.Modal(
                    [
                        dbc.ModalHeader(f"Boxplot for {gene}", style={"fontFamily": "Consolas, monospace"}),
                        dbc.ModalBody(popup_score(gene)),
                    ],
                    id="modal",
                    is_open=True,
                    centered=True,
                    backdrop="static",
                    scrollable=True
                )
            else:
                return None
        # pop-up expressions text
        elif col == "Tissue Expressions":
            return dbc.Modal(
                [
                    dbc.ModalHeader(f"Tissue Expressions for {gene}", style={"fontFamily": "Consolas, monospace"}),
                    dbc.ModalBody(popup_expressions(gene), style={'fontFamily': 'Consolas, monospace', "fontSize": "13px"}),
                ],
                id="modal",
                is_open=True,
                centered=True,
                backdrop="static",
                scrollable=True
            )
    else:
        return None



@callback(
    [Output("tissue1-dropdown", "options"), Output("tissue2-dropdown", "options")],
    [Input("tissue1-dropdown", "value"), Input("tissue2-dropdown", "value"), Input("gene-dropdown", "value")]
)
def update_dropdowns(selected_tissues1, selected_tissues2, selected_genes):
    '''
    Filters both tissue dropdowns based on the selected values.

    Parameters:
    - selected_tissues1 (list of str): Tissue selected in the first tissue dropdown.
    - selected_tissues2 (list of str): Tissues selected in the second tissue dropdown.
    - selected_genes (list of str): Genes selected by the user.

    Returns:
    - options1 (options): Options filtered for the first tissue dropdown.
    - options2 (options): Options filtered for the second tissue dropdown.
    '''
    # only allow different comparisons when we have one gene
    if selected_tissues1:
        remaining_tissues2 = [tissue for tissue in tissues if tissue not in selected_tissues1]
        options2 = [{'label': tissue, 'value': tissue} for tissue in list(reversed(remaining_tissues2))]
    else:
        options2 = [{'label': tissue, 'value': tissue} for tissue in list(reversed(tissues))]
    if selected_tissues2:
        remaining_tissues1 = [tissue for tissue in tissues if tissue not in selected_tissues2]
        options1 = [{'label': tissue, 'value': tissue} for tissue in remaining_tissues1]
    else:
        options1 = [{'label': tissue, 'value': tissue} for tissue in tissues]
    # have at maximum 3 tissues selected in each dropdown (when we have the view of one gene)
    #if selected_tissues1 and selected_tissues2 and (selected_genes == None or len(selected_genes) == 1):
    #    if len(selected_tissues1) == 3:
    #        options1 = [{'label': tissue, 'value': tissue} for tissue in selected_tissues1]
    #    if len(selected_tissues2) == 3:
    #        options2 = [{'label': tissue, 'value': tissue} for tissue in selected_tissues2]
    return options1, options2



@callback(
    Output("gene-dropdown", "value", allow_duplicate=True),
    [Input("gene-dropdown", "value")],
    prevent_initial_call=True
)
def update_gene_dropdown(selected_genes):
    '''
    Updates the selected genes in the gene dropdown when "All genes" or other options are selected.

    Parameters:
    - selected_genes (list): Genes currently selected in the gene dropdown.

   Returns:
    - updated_genes (list): Updated list of selected genes.
    '''
    if selected_genes:
        if "All genes" in selected_genes:
            return ["All genes"]
        elif "Genes from dopaminergic system" in selected_genes:
            return ["Genes from dopaminergic system"]
        elif "Genes from GABAergic system" in selected_genes:
            return ["Genes from GABAergic system"]
    return selected_genes



@callback(
    [Output('graph', 'figure'), Output("output-gene-alert", "children"), Output('table1-container', 'children'), Output('table2-container', 'children'), Output('graph', 'style'), Output('table1-container', 'style'), Output('table2-container', 'style')],
    [Input('tissue1-dropdown', 'value'), Input('tissue2-dropdown', 'value'), Input('gene-dropdown', 'value'), Input('profile-dropdown', 'value')]
)
def update_figure_gene_tab(selected_tissues1, selected_tissues2, selected_genes, selected_profile):
    '''
    Logic to update the figure in the gene tab.

    Parameters:
    - selected_tissues1 (list of str): Input tissues selected by the user.
    - selected_tissues2 (list of str): Output tissues selected by the user.
    - selected_gene (list of str): Gene(s) selected by the user.
    - selected_profile (list of str): Profiles selected by the user.

    Returns:
    - fig (figure): Updated figure to be displayed in the app.
    '''
    global selected_tissues_global
    global selected_data_global
    global ir_selected
    selected_tissues = [[], []]
    # check if there is a gene and tissues selected
    if (selected_genes and selected_tissues1 and selected_tissues2):
        if selected_genes != None and selected_genes != []:
            if selected_genes == ["All genes"]:
                if ir_selected:
                    selected_genes = genes
                else:
                    selected_genes = genes_dopa + genes_gaba
            elif selected_genes == ["Genes from dopaminergic system"]:
                selected_genes = genes_dopa
            elif selected_genes == ["Genes from GABAergic system"]:
                selected_genes = genes_gaba
        if selected_tissues1:
            selected_tissues[0] = selected_tissues1 
        if selected_tissues2:
            selected_tissues[1] = selected_tissues2
        # select the data (based on filter or not)
        selected_data = {}
        for selected_gene in selected_genes:
            selected_data_gene = [[], []]
            if selected_profile:
                for k in range(2):
                    for tissue in selected_tissues[k]:
                        selected_data_gene[k].append(values[selected_gene][tissue])
                selected_data_gene = get_profile_values(selected_data_gene, selected_profile)
            else:
                for k in range(2):
                    for tissue in selected_tissues[k]:
                        selected_data_gene[k].append(values[selected_gene][tissue])
            selected_data[selected_gene] = selected_data_gene
        # keep the global versions for the tables pop-up
        selected_tissues_global = selected_tissues
        selected_data_global = selected_data
        # view for one gene selected 
        if len(selected_data) == 1:
            res = draw_plot_one_gene(next(iter(selected_data.values())), selected_genes[0], selected_tissues, selected_profile)
            if res[0] == None:
                style_for_graph = {'display': 'none'}
            else:
                style_for_graph = {'display': 'block'}
            return res[0], res[1], dash_table.DataTable(), dash_table.DataTable(), style_for_graph, {'display': 'none', 'marginTop':'0px', 'marginBottom':'0px'}, {'display': 'none', 'marginTop':'0px', 'marginBottom':'0px'}
        # view for multiple genes selected
        else:
            res = draw_plot_multiple_gene(selected_data, selected_genes, selected_tissues)
            if res[0] == None:
                style_table1 = {'display': 'none'}
            else:
                style_table1 = {'display': 'block', 'marginTop':'40px', 'marginBottom':'40px'}
            if res[1] == None:
                style_table2 = {'display': 'none'}
            else:
                style_table2 = {'display': 'block', 'marginTop':'40px', 'marginBottom':'40px'}
            return go.Figure(), None, res[0], res[1], {'display': 'none'}, style_table1, style_table2
    # return none if nothing is selected
    return dash.no_update, None, dash.no_update, dash.no_update, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
    




#######################################################################################################################################
# AUXILARY FUNCTIONS - PATTERN TAB                                           
#######################################################################################################################################
@callback(
    Output('tissue-tricluster-table', 'children'),
    [Input('tissue-tricluster-dropdown', 'value')]
)
def update_table_tissue_tricluster(selected_tissue):
    '''
    Updates the table of TriCluster depending on the tissue selected.

    Parameters:
    - selected_tissue (str): Tissue context selected by user.

    Returns:
    - table (dash_table.DataTable): Table to display.
    '''
    global pattern_cube
    global pattern_tissues
    global samples_tricluster_atm
    global samples_tricluster_atm_idx
    global genes_tricluster_atm
    global genes_tricluster_atm_idx
    global tissues_tricluster_atm
    global tissues_tricluster_atm_idx

    if selected_tissue:
        # median case
        if selected_tissue == "Median of the samples":
            # get the data for the table
            med_tissues = []
            for idi, tissue in enumerate(pattern_cube):
                if idi in tissues_tricluster_atm_idx:
                    med_genes = {}
                    for idx, sample in enumerate(tissue):
                        if idx in samples_tricluster_atm_idx:
                            for idy, gene in enumerate(sample):
                                if idy in genes_tricluster_atm_idx:
                                    if idy not in med_genes:
                                        med_genes[idy] = [gene]
                                    else:
                                        med_genes[idy].append(gene)

                    med_tissue = [np.median(values) for values in med_genes.values()]
                    med_tissues.append(med_tissue)
           
            # print(f"Normal:\n{med_tissues}")
            med_tissues_t = [list(row) for row in zip(*med_tissues)]
            column_names = ["Genes - Tissues"] + tissues_tricluster_atm
            # print(column_names)
            # print(column_names)
            table_data = {}
            for col in column_names:
                if col == "Genes - Tissues":
                    table_data[col] = genes_tricluster_atm
                else:
                    table_data[col] = []
            # iterate over the bicluster and add it to the table
            for row_values in med_tissues_t:
                for col_index, value in enumerate(row_values):
                    column_name = column_names[col_index+1]
                    table_data[column_name].append(value)
            table_data_1 = pd.DataFrame(table_data)
            return dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in table_data_1.columns],
                    data=table_data_1.to_dict('records'),
                    style_table={'overflowX': 'auto', 'maxWidth': '100%', 'minWidth': '100%'},
                    editable=False,
                    # page_size=10,
                    style_cell={'textAlign': 'center', 'fontSize': '13px'},  
                    row_selectable=False,
                    style_data_conditional=[
                        {
                            'if': {'column_id': table_data_1.columns[0]},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        },
                        {
                            'if': {'state': 'active'},
                            'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                            'border': '1px solid rgba(0, 116, 217, 0.3)'
                        }
                    ],
                    fixed_columns={'headers': True, 'data': 1},
                    fixed_rows={'headers': True},
                    style_cell_conditional = [
                        {
                            'if': {'column_id': c},
                            'minWidth': '260px' 
                        } for c in table_data_1.columns
                    ],
                )
        # tissue selected case
        else:
            i = pattern_tissues.index(selected_tissue)
            tissue_values = pattern_cube[i]
            # create the data for the table
            datatable_data = []
            for idx, sample in enumerate(tissue_values):
                if idx in samples_tricluster_atm_idx:
                    list_sample = []
                    for idy, gene in enumerate(sample):
                        if idy in genes_tricluster_atm_idx:
                            list_sample.append(gene)
                    datatable_data.append(list_sample)
            # data for the table
            column_names = ["Samples - Genes"] + genes_tricluster_atm
            # print(column_names)
            table_data = {}
            for col in column_names:
                if col == "Samples - Genes":
                    table_data[col] = samples_tricluster_atm
                else:
                    table_data[col] = []
            # iterate over the bicluster and add it to the table
            for row_values in datatable_data:
                for col_index, value in enumerate(row_values):
                    column_name = column_names[col_index+1]
                    table_data[column_name].append(value)
        
            table_data_1 = pd.DataFrame(table_data)
            return dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in table_data_1.columns],
                    data=table_data_1.to_dict('records'),
                    style_table={'overflowX': 'auto', 'maxWidth': '100%', 'minWidth': '100%'},
                    editable=False,
                    # page_size=10,
                    style_cell={'textAlign': 'center', 'fontSize': '13px'},  
                    row_selectable=False,
                    style_data_conditional=[
                        {
                            'if': {'column_id': table_data_1.columns[0]},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        },
                        {
                            'if': {'state': 'active'},
                            'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                            'border': '1px solid rgba(0, 116, 217, 0.3)'
                        }
                    ],
                    fixed_columns={'headers': True, 'data': 1},
                    fixed_rows={'headers': True},
                    style_cell_conditional = [
                        {
                            'if': {'column_id': c},
                            'minWidth': '260px' 
                        } for c in table_data_1.columns
                    ],
                )
    else:
        return None



def clusters_popup(cluster):
    '''
    Create the pop-up text for the cluster selected.

    Parameters:
    - cluster (str): Cluster selected.

    Returns:
    - modal (dbc.Modal): Modal to be displayed.
    '''
    global cluster_information
    global genes
    global pattern_tissues
    global pattern_indiv
    
    # biclustering case
    if cluster.startswith("B"):
        cluster_num = int(cluster.replace("Bicluster", ""))
        c_info = cluster_information[str(int(cluster_num)-1)]

        # text of the pop-up
        text = f"**Dimensions**  \n({len(c_info[0])}, {len(c_info[1])})\n\n"
        text += f"**X**  \n("
        for el in c_info[0]:
            text += f"{genes[el]}, "
        if len(c_info[0]) != 0:
            text = text[:-2]
        text += ")\n\n"
        text += f"**Y**  \n("
        for el in c_info[1]:
            text += f"{pattern_tissues[el]}, "
        if len(c_info[1]) != 0:
            text = text[:-2]
        text += ")\n\n"

        # data for the table
        column_names = ["Genes - Tissues"] + [pattern_tissues[el] for el in c_info[1]]
        # print(column_names)
        table_data = {}
        for col in column_names:
            if col == "Genes - Tissues":
                table_data[col] = [genes[el] for el in c_info[0]]
            else:
                table_data[col] = []
        # iterate over the bicluster and add it to the table
        for row_values in c_info[2]:
            for col_index, value in enumerate(row_values):
                column_name = column_names[col_index+1]
                table_data[column_name].append(value)
        # print(table_data)
        # for key, value in table_data.items():
            # print(f"Length of '{key}': {len(value)}")
        table_df = pd.DataFrame(table_data)
        # data table
        table = dash_table.DataTable(
            id='modal-table',
            columns=[{"name": i, "id": i} for i in table_df.columns],
            data=table_df.to_dict('records'),
            style_table={'overflowX': 'auto', 'maxWidth': '100%', 'minWidth': '100%'},
            editable=False,
            #page_size=10,
            style_cell={'textAlign': 'center', 'fontSize': '13px'},  
            row_selectable=False,
            style_data_conditional=[
                {
                    'if': {'column_id': table_df.columns[0]},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                    'if': {'state': 'active'},
                    'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                    'border': '1px solid rgba(0, 116, 217, 0.3)'
                }
            ],
            fixed_columns={'headers': True, 'data': 1},
            fixed_rows={'headers': True},
            style_cell_conditional = [
                {
                    'if': {'column_id': c},
                    'minWidth': '260px' 
                } for c in table_df.columns
            ],
        )
        # create the modal
        modal = dbc.Modal(
            [
                dbc.ModalHeader(f"Information about the bicluster", style={"fontFamily": "Consolas, monospace"}),
                dbc.ModalBody([
                    dcc.Markdown(text, style={'fontFamily': 'Consolas, monospace', "fontSize": "13px", 'marginBottom':'25px'}),
                    table
                ])
            ],
            id="modal",
            is_open=True,
            centered=True,
            backdrop="static",
            scrollable=True,
            size="lg"
        )
        return modal
    # triclustering case
    else:
        cluster_num = int(cluster.replace("Tricluster", ""))
        c_info = cluster_information[str(int(cluster_num)-1)]

        global samples_tricluster_atm
        global samples_tricluster_atm_idx
        global genes_tricluster_atm
        global genes_tricluster_atm_idx
        global tissues_tricluster_atm
        global tissues_tricluster_atm_idx
        samples_tricluster_atm = [pattern_indiv[el] for el in c_info[1]]
        samples_tricluster_atm_idx = [el for el in c_info[1]]
        genes_tricluster_atm = [genes[el] for el in c_info[2]]
        genes_tricluster_atm_idx = [el for el in c_info[2]]
        tissues_tricluster_atm = [pattern_tissues[el] for el in c_info[0]]
        tissues_tricluster_atm_idx = [el for el in c_info[0]]

        # text of the pop-up
        text = f"**Dimensions**  \n({len(c_info[1])}, {len(c_info[2])}, {len(c_info[0])})\n\n"
        text += f"**X**  \n("
        for el in c_info[1]:
            text += f"{pattern_indiv[el]}, "
        if len(c_info[1]) != 0:
            text = text[:-2]
        text += ")\n\n"
        text += f"**Y**  \n("
        for el in c_info[2]:
            text += f"{genes[el]}, "
        if len(c_info[2]) != 0:
            text = text[:-2]
        text += ")\n\n"
        text += f"**Z**  \n("
        for el in c_info[0]:
            text += f"{pattern_tissues[el]}, "
        if len(c_info[0]) != 0:
            text = text[:-2]
        text += ")\n\n"

        modal = dbc.Modal(
            [
                dbc.ModalHeader(f"Information about the tricluster", style={"fontFamily": "Consolas, monospace"}),
                dbc.ModalBody([
                    dcc.Markdown(text, style={'fontFamily': 'Consolas, monospace', "fontSize": "13px", 'marginBottom':'25px'}),
                    dcc.Dropdown(["Median of the samples"] + [pattern_tissues[el] for el in c_info[0]], id='tissue-tricluster-dropdown', placeholder="Select an tissue context...", style={'fontFamily': 'Consolas, monospace', 'fontSize': '13px', 'marginBottom':'25px'}),
                    html.Div(id='tissue-tricluster-table')
                ])
            ],
            id="modal",
            is_open=True,
            centered=True,
            backdrop="static",
            scrollable=True,
            size="lg"
        )
        return modal
        # return dcc.Markdown(text)





#######################################################################################################################################
# CALLBACK FUNCTIONS - PATTERN TAB                                                    
#######################################################################################################################################
@callback(
    [Output("num-clusters-dropdown", 'value'), Output("pattern-type-dropdown", 'value'), Output("mapper-strategy-dropdown", 'value'), Output("min-size-individuals-dropdown", 'value'), Output("min-size-genes-dropdown", 'value'), Output("min-size-tissues-dropdown", 'value')],
    Input("alg-dropdown", "value")
)
def update_dropdowns(_):
    '''
    When algorithm changes, remove the values already selected for consistency.
    '''
    return None, None, None, None, None, None



@callback(
    [Output("num-clusters-dropdown-container", "style"), Output("pattern-type-container", "style"), Output("mapper-strategy-container", "style"), Output("min-size-tissues-container", "style"), Output("min-size-individuals-container", "style"), Output("min-size-genes-container", "style")],
    [Input("alg-dropdown", "value")]
)
def update_dropdowns_based_on_algorithm(selected_alg):
    '''
    Updates the dropdowns depending on the algorithm chosen.

    Parameters:
    - selected_alg (str): Algorithm selected by the user.

    Returns:

    '''
    if selected_alg and (selected_alg == "Biclustering - Spectral Coclustering" or selected_alg == "Biclustering - Spectral Biclustering"):
        return {'display':'block'}, {"display":"none"}, {"display":"none"}, {"display":"none"}, {"display":"none"}, {"display":"none"}
    elif selected_alg and selected_alg == "Biclustering - BicPAMS":
        return {'display':'none'}, {"display":"block"}, {"display":"block"}, {"display":"none"}, {"display":"none"}, {"display":"none"}
    elif selected_alg and selected_alg == "Triclustering - TriCluster":
        return {'display':'none'}, {"display":"none"}, {"display":"none"}, {"display":"block"}, {"display":"block"}, {"display":"block"}
    else:
        return {'display':'none'}, {"display":"none"}, {"display":"none"}, {"display":"none"}, {"display":"none"}, {"display":"none"}



@callback(
    [Output("tissues-pattern-dropdown", "value"), Output("num-clusters-dropdown", "options")],
    [Input("tissues-pattern-dropdown", "value")]
)
def update_tissue_dropdown(selected_tissues):
    '''
    Updates the selected tissues in the tissues pattern dropdown when "All tissues" is selected and the options for num-clusters-dropdown.

    Parameters:
    - selected_tissues (list): Tissues currently selected in the dropdown.

    Returns:
    - updated_tissues (list): Updated list of selected tissues.
    - options (options): Options for the maximum number of clusters to find.
    '''
    if selected_tissues:
        if "All tissues" in selected_tissues:
            return ["All tissues"], [{'label': str(i), 'value': i} for i in range(2, 19)]
        elif len(selected_tissues) == 1:
            return selected_tissues, [{"label": "No options", "value": "No options"}]
        else:
            return selected_tissues, [{'label': str(i), 'value': i} for i in range(2, len(selected_tissues)+1)]
    return selected_tissues, [{"label": "No options", "value": "No options"}]



@callback(
    [Output("min-size-tissues-dropdown", "options"), Output("min-size-individuals-dropdown", "options")],
    [Input("tissues-pattern-dropdown", "value")]
)
def update_min_tissues_dropdown(selected_tissues):
    '''
    Update the minimum size tissues and individuals for the TriCluster algorithm.

    Parameters:
    - selected_tissues (list): Tissues currently selected in the dropdown.

    Returns:
    - options1 (options): Options filtered for the minimum size tissues dropdown.
    - options2 (options): Options filtered for the minimum size individuals dropdown.
    '''
    if selected_tissues:
        # for the tissues
        if selected_tissues == ["All tissues"]:
            selected_tissues = [t for t in tissues]
        list_t_length = len(selected_tissues)
        options1 = [{'label': str(i), 'value': i} for i in range(2, list_t_length + 1)]
        # print(options1)
        # for the individuals
        _, individuals = pm.create_cube(values, genes, [t for t in selected_tissues], tissue_indiv)
        # print(individuals)
        list_i_length = len(individuals)
        options2 = [{'label': str(i), 'value': i} for i in range(2, list_i_length + 1)]
        # print(options2)
        return options1, options2
    else:
        return [{"label": "No options", "value": "No options"}], [{"label": "No options", "value": "No options"}]



@callback(
    dash.dependencies.Output('clusters-popup', 'children'),
    [dash.dependencies.Input('table3', 'active_cell'), dash.dependencies.Input('table3', 'page_current')],
    prevent_initial_call=True
)
def display_clusters_popup(active_cell, page_current):
    '''
    For the clusters found table.

    Parameters:
    - active_cell (active_cell): Cell that is active at the moment.

    Returns:
    - clusters-popup (children): Pop-up to appear.
    '''
    global df3
    df3 = df3.sort_values(by=['Significances', 'Clusters Retrieved'], ascending=[True, True])
    global pattern_alg
    if active_cell:
        row = active_cell['row']
        col = active_cell['column_id']
        # value = df1.iloc[row][col]
        cluster = df3.iloc[active_cell['row']+(10*page_current)]['Clusters Retrieved']
        # print(f"O cluster {cluster}")
        # information cluster pop-up 
        if col == "Clusters Retrieved":
            return clusters_popup(cluster)
    else:
        return None



@callback(
    [Output('table3-container', 'children'), Output("output-pattern-alert", "children")],
    [Input('alg-dropdown', 'value'), Input('num-clusters-dropdown', 'value'), Input('tissues-pattern-dropdown', 'value'), Input("pattern-type-dropdown", "value"), Input("mapper-strategy-dropdown", "value"), Input("min-size-tissues-dropdown", "value"), Input("min-size-individuals-dropdown", "value"), Input("min-size-genes-dropdown", "value")],
    prevent_initial_call=True
)
def update_figure_pattern_tab(selected_alg, selected_max_num_clusters, selected_pattern_tissues, selected_pattern_type, selected_mapper_strategy, minsize_tissues, minsize_indiv, minsize_genes):
    '''
    Logic to update the figure in the pattern tab.

    Parameters:
    - selected_alg (str): Algorithm selected for pattern discovery.
    - selected_max_num_clusters (int): Maximum number of clusters for pattern discovery algorithm.
    - selected_pattern_tissues (list of str): Tissues selected for pattern discovery.
    - selected_pattern_type (str): Pattern type selected for BicPAMS.
    - selected_mapper_strategy (str): Mapper strategy selected for BicPAMS.
    - minsize_tissues (int): Minimum size for clustering tissues for TriCluster.
    - minsize_indiv (int): Minimum size for clustering individuals for TriCluster.
    - minsize_genes (int): Minimum size for clustering genes for TriCluster.

    Returns:
    - table3 (dash_table.DataTable): Table displaying the clusters found.
    '''
    # get the tissues
    global pattern_tissues
    global pattern_indiv
    global genes
    global values
    if selected_pattern_tissues:
        if selected_pattern_tissues == ["All tissues"]:
            selected_pattern_tissues = [t for t in tissues]
        if len(selected_pattern_tissues) == 1:
            return None, dbc.Alert("Please select more than one tissue to proceed with the algorithm", color="danger", style={'font-family': 'Consolas, monospace', 'font-size': '13px', 'marginBottom': 0})
        pattern_tissues = selected_pattern_tissues

    # choose the algorithm
    if selected_alg == "Biclustering - Spectral Coclustering" and selected_pattern_tissues and selected_max_num_clusters:
        matrix = pm.create_matrix(values, genes, selected_pattern_tissues)
        df_matrix = pd.DataFrame(matrix, index=genes, columns=selected_pattern_tissues)
        # df_matrix.to_csv('data/matrix_spectralco.csv', index=False)
        row_indices, column_indices, biclusters = bc.spectral_coclustering(matrix, selected_max_num_clusters)
        significances = bc.calculate_significances(matrix, row_indices, column_indices, biclusters)
    elif selected_alg == "Biclustering - Spectral Biclustering" and selected_pattern_tissues and selected_max_num_clusters:
        matrix = pm.create_matrix(values, genes, selected_pattern_tissues)
        df_matrix = pd.DataFrame(matrix, index=genes, columns=selected_pattern_tissues)
        # df_matrix.to_csv('data/matrix_spectralbi.csv', index=False)
        row_indices, column_indices, biclusters = bc.spectral_biclustering(matrix, selected_max_num_clusters)
        significances = bc.calculate_significances(matrix, row_indices, column_indices, biclusters)
    elif selected_alg == "Biclustering - BicPAMS" and selected_pattern_tissues and selected_pattern_type and selected_mapper_strategy:
        matrix = pm.create_matrix(values, genes, selected_pattern_tissues)
        df_matrix = pd.DataFrame(matrix, index=genes, columns=selected_pattern_tissues)
        # df_matrix.to_csv('data/matrix_bicpams.csv', index=False) 
        mincols = max(2, math.floor(math.sqrt(len(selected_pattern_tissues))))
        try:
            row_indices, column_indices, biclusters, significances = bc.bicpams(df_matrix, selected_pattern_type, selected_mapper_strategy, mincols)
        except KeyError as e:
            # print(f"ERRO:\n{e}")
            return None, dbc.Alert("Error running the algorithm", color="danger", style={'font-family': 'Consolas, monospace', 'font-size': '13px', 'marginBottom': 0})
    elif selected_alg == "Triclustering - TriCluster" and selected_pattern_tissues and minsize_tissues and minsize_indiv and minsize_genes:
        global pattern_cube
        cube, individuals = pm.create_cube(values, genes, [t for t in selected_pattern_tissues], tissue_indiv)
        pattern_indiv = individuals
        pattern_cube = cube
        triclusters = tc.tricluster(cube, minsize_tissues, minsize_indiv, minsize_genes)
        if not triclusters:
            return None, dbc.Alert("Algorithm executed with success", color="success", style={'font-family': 'Consolas, monospace', 'font-size': '13px', 'marginBottom': 0})
        significances_tri = tc.calculate_significances(cube, triclusters)
        # print(f"SIG AQUI: {significances_tri}")
    else:
        return None, None

    # define the information about the clusters retrieved
    global cluster_information
    global pattern_alg
    pattern_alg = selected_alg
    cluster_information = {}
    i = 0
    # biclustering case
    significances_to_remove = []
    if pattern_alg == "Biclustering - Spectral Coclustering" or pattern_alg == "Biclustering - Spectral Biclustering" or pattern_alg == "Biclustering - BicPAMS":
        prefix = selected_alg[:2]
        j = 0
        for row_i, column_i, bicluster, significance in zip(row_indices, column_indices, biclusters, significances):
            if row_i.tolist() and column_i.tolist() and not np.isnan(significance):
                # print(f"ENTREI {i}")
                cluster_information[str(i)] = [row_i, column_i, bicluster, significance]
                i += 1
            else:
                significances_to_remove.append(j)
            j += 1
        new_significances = [significances[p] for p in range(len(significances)) if p not in significances_to_remove]
    # triclustering case
    else:
        prefix = selected_alg[:3]
        if triclusters:
            for tricluster in triclusters:
                cluster_information[str(i)] = [tricluster[0], tricluster[1], tricluster[2]]
                i += 1

    # define the table
    global df3
    # biclustering case
    if pattern_alg == "Biclustering - Spectral Coclustering" or pattern_alg == "Biclustering - Spectral Biclustering" or pattern_alg == "Biclustering - BicPAMS":
        df3 = pd.DataFrame({
            'Clusters Retrieved': [f"{prefix}cluster{m}" for m in range(1, i+1)],
            'Significances': new_significances
        })
    # triclustering case
    else:
        # print(f"SIGNI2: {significances_tri}")
        df3 = pd.DataFrame({
            'Clusters Retrieved': [f"{prefix}cluster{m}" for m in range(1, i+1)],
            'Significances': significances_tri
        })
    # print(f"ANTES:\n{df3.head(14)}")
    df3 = df3.sort_values(by=['Significances', 'Clusters Retrieved'], ascending=[True, True])
    # print(f"cr2: {cr_2} - s_2: {s_2}")
    # print(f"DEPOIS:\n{df3.head(14)}")
    table3 = dash_table.DataTable(
        id='table3',
        columns=[{"name": i, "id": i} for i in df3.columns],
        data=df3.to_dict('records'),
        page_size=10,
        page_current=0,
        style_cell={'textAlign': 'center', 'fontSize': '13px', 'cursor': 'pointer'},  
        row_selectable=False,
        style_data_conditional=[
            {
                'if': {'state': 'active'},
                'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                'border': '1px solid rgba(0, 116, 217, 0.3)'
            }
        ],
        style_cell_conditional = [
                {
                    'if': {'column_id': c},
                    'minWidth': '250px' 
                } for c in df3.columns
        ],
    )
    return table3, dbc.Alert("Algorithm executed with success", color="success", style={'font-family': 'Consolas, monospace', 'font-size': '13px', 'marginBottom': 0})





#######################################################################################################################################
# APP                                                   
#######################################################################################################################################
@callback(
    Output('tab-output', 'children'), 
    [Input('tabs', 'value'), Input('tissue1-dropdown', 'value'), Input('tissue2-dropdown', 'value'), Input('gene-dropdown', 'value'), Input('profile-dropdown', 'value'), Input('alg-dropdown', 'value'), Input('num-clusters-dropdown', 'value'), Input('tissues-pattern-dropdown', 'value'), Input("pattern-type-dropdown", "value"), Input("mapper-strategy-dropdown", "value"),  Input("min-size-tissues-dropdown", "value"), Input("min-size-individuals-dropdown", "value"), Input("min-size-genes-dropdown", "value")]
)
def update_tab(selected_tab, selected_tissues1, selected_tissues2, selected_genes, selected_profile, selected_alg, selected_max_num_clusters, selected_pattern_tissues, selected_pattern_type, selected_mapper_strategy, minsize_tissues, minsize_indiv, minsize_genes):
    '''
    Choose the right tab, depending on the one selected.

    Parameters:
    - selected_tab (str): Can be "gene-centric" or "pattern-centric".
    - selected_tissues1 (list of str): Input tissues selected by the user.
    - selected_tissues2 (list of str): Output tissues selected by the user.
    - selected_gene (str): Gene selected by the user.
    - selected_profile (list of str): Profiles selected by the user.
    - selected_alg (str): Algorithm selected for pattern discovery.
    - selected_pattern_tissues (list of str): Tissues selected for pattern discovery.
    - selected_max_num_clusters (int): Maximum number of clusters for sklearn.
    - selected_pattern_type (str): Pattern type selected for BicPAMS.
    - selected_mapper_strategy (str): Mapper strategy selected for BicPAMS.
    - minsize_tissues (int): Minimum value size for tissues.
    - minsize_indiv (int): Minimum value size for individuals.
    - minsize_genes (int): Minimum value size for genes.

    Returns:
    - tab-output (children): Tab to select.
    '''
    global values
    global tissue_samples
    global tissue_indiv
    global genes

    global values_ir
    global tissue_samples_ir
    global tissue_indiv_ir 
    global genes_ir

    global values_string
    global tissue_samples_string
    global tissue_indiv_string
    global genes_dopa
    global genes_gaba

    global ir_selected

    if selected_tab == "gene-centric":
        if ir_selected and genes:
            genes = genes_ir
            values = values_ir
            tissue_samples = tissue_samples_ir
            tissue_indiv = tissue_indiv_ir
        elif not ir_selected and genes:
            genes = genes_dopa + genes_gaba
            values = values_string
            tissue_samples = tissue_samples_string
            tissue_indiv = tissue_indiv_string
        update_figure_gene_tab(selected_tissues1, selected_tissues2, selected_genes, selected_profile)
    elif selected_tab == "pattern-centric":
        if genes:
            genes = genes_ir
            values = values_ir
            tissue_samples = tissue_samples_ir
            tissue_indiv = tissue_indiv_ir
        update_figure_pattern_tab(selected_alg, selected_max_num_clusters, selected_pattern_tissues, selected_pattern_type, selected_mapper_strategy, minsize_tissues, minsize_indiv, minsize_genes)



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Spatial distribution of neurotransmission markers"



app.layout = html.Div([

    html.Div(
        style={'textAlign': 'center', 'fontFamily': 'Consolas, monospace', 'marginBottom': '20px', 'marginTop': '20px'}, 
        children=[html.H1("Spatial distribution of neurotransmission markers", style={'fontSize': '20px', 'fontWeight': 'bold'})]
    ),

    dcc.Tabs(id='tabs', value='gene-centric', children=[
        dcc.Tab(label='Gene-centric stances', value='gene-centric', style={'fontFamily': 'Consolas, monospace', 'fontSize': '15px', 'fontWeight': 'normal'}, selected_style={'fontFamily': 'Consolas, monospace', 'fontSize': '15px', 'fontWeight': 'bold'}, children=[

            html.Div(id="output-gene-alert"),
            
            dcc.Dropdown([tissue for tissue in tissues], id="tissue1-dropdown", placeholder="Select an input tissue set...", multi=True, style={'fontFamily': 'Consolas, monospace', 'fontSize': '13px'}),

            dcc.Dropdown([tissue for tissue in list(reversed(tissues))], id="tissue2-dropdown", placeholder="Select an output tissue set...", multi=True, style={'fontFamily': 'Consolas, monospace', 'fontSize': '13px'}),

            dcc.Dropdown(["All genes"] + [gene for gene in genes], id="gene-dropdown", placeholder="Select a gene set...", style={'fontFamily': 'Consolas, monospace', 'fontSize': '13px'}, multi=True),

            dcc.Dropdown(
                options=[
                    {'label': 'Sex - Male', 'value': 'sex-male'},
                    {'label': 'Sex - Female', 'value': 'sex-female'},
                    {'label': 'Age - 20-29', 'value': 'age-20-29'},
                    {'label': 'Age - 30-39', 'value': 'age-30-39'},
                    {'label': 'Age - 40-49', 'value': 'age-40-49'},
                    {'label': 'Age - 50-59', 'value': 'age-50-59'},
                    {'label': 'Age - 60-69', 'value': 'age-60-69'},
                    {'label': 'Age - 70-79', 'value': 'age-70-79'},
                    {'label': 'Agonal - Ventilator case', 'value': 'agonal-0'},
                    {'label': 'Agonal - Violent and fast death', 'value': 'agonal-1'},
                    {'label': 'Agonal - Fast death of natural causes', 'value': 'agonal-2'},
                    {'label': 'Agonal - Intermediate death', 'value': 'agonal-3'},
                    {'label': 'Agonal - Slow death', 'value': 'agonal-4'},
                ],
                value=None, multi=True, placeholder="Select a profile set...", id='profile-dropdown', style={'fontFamily': 'Consolas, monospace', 'fontSize': '13px'}
            ),

            dcc.Checklist(
                id='ir-checkbox',
                options=[
                    {'label': ' Use genes from STRING database', 'value': 'checked'}
                ],
                style={'fontFamily': 'Consolas, monospace', 'fontSize': '13px'},
                value=[]
            ),
        
            dcc.Graph(id="graph", style={'fontFamily': 'Consolas, monospace', 'display':'none'}),

            html.Div(
                html.Div(
                    dash_table.DataTable(id="table1"),
                    id='table1-container',
                ),
            ),

            html.Div(
                html.Div(
                    dash_table.DataTable(id="table2"),
                    id='table2-container',
                ),
            ),

            html.Div(id='tablegene-popup'),

            html.Div(id='tabletissue-popup'),

        ]),

        dcc.Tab(label='Pattern-centric stances', value='pattern-centric', style={'fontFamily': 'Consolas, monospace', 'fontSize': '15px', 'fontWeight': 'normal'}, selected_style={'fontFamily': 'Consolas, monospace', 'fontSize': '15px', 'fontWeight': 'bold'}, children=[
            
            html.Div(id="output-pattern-alert"),
            
            dcc.Dropdown(["Biclustering - Spectral Coclustering", "Biclustering - Spectral Biclustering", "Biclustering - BicPAMS", "Triclustering - TriCluster"], id="alg-dropdown", placeholder="Select an algorithm...", style={'fontFamily': 'Consolas, monospace', 'fontSize': '13px'}),

            dcc.Dropdown(["All tissues"] + [tissue for tissue in tissues], id="tissues-pattern-dropdown", placeholder="Select a tissue set...", style={'fontFamily': 'Consolas, monospace', 'fontSize': '13px'}, multi=True),
            
            # sklearn
            html.Div(id="num-clusters-dropdown-container", children=[
                dcc.Dropdown(["No options"], id="num-clusters-dropdown", placeholder="Select a maximum number of clusters...", style={'fontFamily': 'Consolas, monospace', 'fontSize': '13px', 'marginBottom':'40px'})
            ], style={'display':'none'}),

            # bicpams
            html.Div(id="pattern-type-container", children=[
                dcc.Dropdown(["Constant", "Order-preserving"], id="pattern-type-dropdown", placeholder="Select a type of bicluster...", style={'fontFamily': 'Consolas, monospace', 'fontSize': '13px'})
            ], style={'display':'none'}),
            html.Div(id="mapper-strategy-container", children=[
                dcc.Dropdown(["Column Normalization", "Row Normalization", "Overall Normalization", "Width", "Rows Frequency", "Frequency"], id="mapper-strategy-dropdown", placeholder="Select a mapper strategy...", style={'fontFamily': 'Consolas, monospace', 'fontSize': '13px', 'marginBottom':'40px'})
            ], style={'display':'none'}),

            # tricluster
            html.Div(id="min-size-individuals-container", children=[
                dcc.Dropdown(["No options"], id="min-size-individuals-dropdown", placeholder="Select a minimum size for individuals...", style={'fontFamily': 'Consolas, monospace', 'fontSize': '13px'})
            ], style={'display':'none'}),
            html.Div(id="min-size-genes-container", children=[
                dcc.Dropdown([i for i in range(2, 197)], id="min-size-genes-dropdown", placeholder="Select a minimum size for genes...", style={'fontFamily': 'Consolas, monospace', 'fontSize': '13px'})
            ], style={'display':'none'}),
            html.Div(id="min-size-tissues-container", children=[
                dcc.Dropdown(["No options"], id="min-size-tissues-dropdown", placeholder="Select a minimum size for tissues...", style={'fontFamily': 'Consolas, monospace', 'fontSize': '13px', 'marginBottom':'40px'})
            ], style={'display':'none'}),

            html.Div(
                html.Div(
                    dash_table.DataTable(id="table3"),
                    id='table3-container',
                ),
            ),

            html.Div(id='clusters-popup'),

        ]),
    ]),

        html.Div(id='tab-output')
])



if __name__ == '__main__':
    app.run(debug=False)