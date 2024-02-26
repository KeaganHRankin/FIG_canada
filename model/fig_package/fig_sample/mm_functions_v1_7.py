"""
This Python file contains a function that imports and correctly formats the database data
for visualization and data analysis.
"""

# Import Libraries
# Data
import numpy as np
import scipy as sp
import pandas as pd

#Viz
import matplotlib.pyplot as plt
import seaborn as sns


#----------------------------------------------------------------------------------------------------------------------------------------------------
# SUMMARY SHEET V FUNCTIONS


def load_database(path, ss_use_cols, ss_l, non_mm_dfs = False):
    """ 
    Loads in the building summary and summary_sheet_V data.
    path = file path for import.
    ss_use_cols = range of columns on "summary_sheet_V" to be used; ex. B:CQ
    ss_l = length of summary sheet range (aka # of buildings in database)
    non_mm_dfs = whether to return SND and highrise residential dfs. Default false.
    """
    
    print('[INFO] starting.')
    
    # Import Building Summary
    sum_df = pd.read_excel(io = path, 
                           sheet_name = "Building Data", 
                           usecols = 'H:Y',
                           header = 2,
                           index_col = 0
                           )

    # Adjust Names
    sum_df.index.name = 'Building'
    sum_df.rename(columns={'Code.2':'Code','Building Code.1':'Building Code'}, inplace=True)
    
    
    print('[INFO] building summary imported.')
    
    # Import Raw Data
    raw_df = pd.read_excel(io = path,
                           sheet_name = "Summary_Sheet_V",
                           usecols = ss_use_cols,
                           header = 3,
                           )
    
    # Transpose data
    raw_df = raw_df.T

    # Reorganize headers
    head = raw_df.iloc[0]
    raw_df = raw_df[1:]
    raw_df.columns = head

    # Change index to numeric
    indexr = np.arange(1,ss_l+1,1)
    raw_df.index = indexr
    
    print('[INFO] raw data imported.')
    
    # FILTER DATA TO MM AND FORMAT.
    # Filter floors first. Put those above in highrise.
    mm_df = sum_df[sum_df['Number of Floors Above Ground'] <= 4]
    hr_df = sum_df[sum_df['Number of Floors Above Ground'] > 4]
    hr_df = hr_df[~hr_df['Bedroom Qnt'].isnull()]

    # Remove SNDs and place in their own frame
    snd_df = sum_df[sum_df['Building Code'].isin(['SND','SNR'])]
    mm_df = mm_df[~mm_df['Building Code'].isin(['SND','SNR'])]

    # Filter bedrooms
    mm_df = mm_df[~mm_df['Bedroom Qnt'].isnull()]

    
    # Cut raw dataframe into the codes above. Missing Middle, High Rise, Single Family Detached
    mm_raw_df = raw_df.loc[mm_df['Code']]
    hr_raw_df = raw_df.loc[hr_df['Code']]
    snd_raw_df = raw_df.loc[snd_df['Code']]
    
    print('[INFO] data filtered and organized. Returning [Complete].')
    
    # Return the desired dataframes:
    if non_mm_dfs == False:
        return mm_df, mm_raw_df
    
    elif non_mm_dfs == True:
        return mm_df, snd_df, hr_df, mm_raw_df, snd_raw_df, hr_raw_df

    
    
def load_names(filename):
    """
    loads the name conversions from code to uni/masterformat.
    from "Covert.ipynb"  (Gueven et al. 2022)
    filepath = filepath to mapping_material_names.xslx
    """
    
    #Load in the mapper
    mapper = pd.read_excel(filename, header=1, usecols='B:L').replace(r'\n','', regex=True) 
    #building_name_mapper = pd.read_excel('BuildingTypeNames.xlsx')
    
    #Additional categories map as well
    additional_categories_map = {v:k for k,v in {
        'Continuous Footings':'0CF',
        'Foundation Walls':'0FW',
        'Spread Footings':'0SF',
        'Column Piers':'0CP',
        'Columns Supporting Floors':'CSF',
        'Floor Girders and Beams':'FGB',
        'Floor Trusses':'0FT',
        'Floor Joists':'0FJ',
        'Columns Supporting Roofs':'CSR',
        'Roof Girders and Beams':'RGB',
        'Roof Trusses':'0RT',
        'Roof Joists':'0RJ',
        'Parking Bumpers':'0PB',
        'Precast Concrete Stair Treads':'PCS',
        'Roof Curbs':'0RC',
        'Exterior Wall Construction':'EWC',
        'Composite Decking':'CPD',
        'Cast-in-Place concrete':'CIC',
        'Floor Structural Frame':'FSF',
        'Associated Metal Fabrications':'AMF',
        'Floor Construction Supplementary Components':'FCS',
        'Roof Construction Supplementary Components':'RCS',
        'Residential Elevators':'0RE',
        'Vegetated Low-Slope Roofing':'VLR',
        'Swimming Pools':'SWP',
        'Excavation Soil Anchors':'ESA',
        'Floor Trusses':'FTS',
        'Roof Window and Skylight Performance':'RWS',
        'Rainwater Storage Tanks':'RST',
        'Gray Water Tanks':'GWT'}.items()
    }

    additional_categories_map['0FT'] = 'Floor Trusses'
    
    #return values
    return mapper


def load_master_uni(uni_level, master_level, filepath):
    """
    loads in desired level of uni and master format with coded names, which
    can then be used to with a grouper to map raw data to coded names.
    
    uni_level = desired level of uniformat.
    master_level = desired level of masterformat.
    filepath= filepath for dataset.
    
    future improvements -> fully automatic grouping of dataset
    that renames columns based on the dfs loaded above.
    -> there are some materials that are not grouped at higher levels.
    Need to loop back through the levels until all are grouped!
    """
    # Store a dictionary of which columns map to which uni/masterformat levels
    uni_cols = {1:"B:E",
                2:"H:K",
                3:"N:Q",
                4:"T:W",
                5:"T:W",    #5:"Z:AC",
               }
    
    master_cols = {1:"AG:AJ",
                   2:"AM:AP",
                   3:"AS:AV",
                   4:"AY:BB",
                   5:"BE:BH",
                  }

    # Parse based on requested levels in "usecols".
    # Load uni and master mappers. Fix the column titles.
    uni_mapper = pd.read_excel(filepath,
                               sheet_name="UF&MF Data",
                               usecols=uni_cols[uni_level],
                               header=4,
                               index_col=0
                              )
    
    uni_mapper.index.name = 'c'
    uni_mapper.columns = ['category', 'level', 'code']
    
    
    master_mapper = pd.read_excel(filepath,
                                  sheet_name="UF&MF Data",
                                  usecols=master_cols[master_level],
                                  header=4,
                                  index_col=0
                                 )
    
    master_mapper.index.name = 'c'
    master_mapper.columns = ['category', 'level', 'code']
    
    for key in master_cols:
        # Collect data for the desired master level and all of the lower masterformat codes.
        if key == master_level:
            break
        # Loop to get all of the master_format levels
        master_map = pd.read_excel(filepath,
                                  sheet_name="UF&MF Data",
                                  usecols=master_cols[key],
                                  header=4,
                                  index_col=0
                                  )
        
        master_map.index.name = 'c'
        master_map.columns = ['category', 'level', 'code']
    
        master_mapper = pd.concat([master_mapper, master_map])
    
    
    # -----
    for key in uni_cols:
        # Collect data for the desired master level and all of the lower masterformat codes.
        if key == uni_level:
            break
        # Loop to get all of the uni_format levels
        uni_map = pd.read_excel(filepath,
                                sheet_name="UF&MF Data",
                                usecols=uni_cols[key],
                                header=4,
                                index_col=0
                                )

        uni_map.index.name = 'c'
        uni_map.columns = ['category', 'level', 'code']

        uni_mapper = pd.concat([uni_mapper, uni_map])
        #if uni_level == 5:
        # Append the lvl 5 formats 
    # -----
    
    # Drop null rows
    uni_mapper = uni_mapper.dropna()
    master_mapper = master_mapper.dropna()

    
    # Return
    return uni_mapper, master_mapper



def group_by_master_uni(raw_df, sum_df, uni_level, master_level, db_path):
    """
    Groups the database by uni and master format mapper.
    Requires load_master_uni() already imported.
    uni_level and master_level should be the same as in
    the previous function.
    
    raw_df = raw dataframe from mat database.
    sum_df = summary dataframe from which to draw intensity info.
    uni_level = uniformat aggregate level.
    master_level = masterformat aggregate level.
    db_path = path to database.
    
    output: group df w/ intensity normalizers (GFA, Bedroom Qnt, etc.)
    """
    
    # retrieve uni and master map.
    uni_map, master_map = load_master_uni(uni_level, master_level, db_path)
    
    # Hash for grouper and mapper string lengths. {level:str_len}.
    uni_lens = {1:1,
                2:3,
                3:5,
                4:8,
                5:12,
               }
    
    master_lens = {1:2,
                   2:4,
                   3:6,
                   4:8,
                   5:11,
                  }

    # Group based on uniformat level.
    kg_cols = [d for d in raw_df.columns if "kg" in str(d)]
    grouper = lambda x: x.split('_')[1][0:uni_lens[uni_level]]
    uni_group = raw_df[kg_cols].groupby(grouper, axis=1).sum()
    uni_group = uni_group.rename(columns=dict(zip(uni_map["code"], uni_map["category"])))

    # Appending General Data for Visualization
    uni_group = uni_group.join(sum_df[['GFA', 'Unit Qnt', 'Bedroom Qnt', 'Number of Floors Below Ground', 'Number of Floors Above Ground']])
    uni_group = uni_group.fillna(0)#.drop('000', axis=1
    
    
    # Group based on masterformat level.
    kg_cols = [d for d in raw_df.columns if "kg" in str(d)]
    grouper = lambda x: x.split('_')[2][0:master_lens[master_level]]
    master_group = raw_df[kg_cols].groupby(grouper, axis=1).sum()
    master_group = master_group.rename(columns=dict(zip(master_map["code"].str[0:master_lens[master_level]], master_map["category"])))

    # Appending General Data for Visualization
    master_group = master_group.join(sum_df[['GFA', 'Unit Qnt', 'Bedroom Qnt', 'Number of Floors Below Ground', 'Number of Floors Above Ground']])
    master_group = master_group.fillna(0)

    return uni_group, master_group



def import_sum_df(path):
    """
    Fuction imports the summary sheet with some cleaning.
    path = file path
    
    """
    
    
    sum_df = pd.read_excel(io = path, 
                       sheet_name = "Building Data", 
                       usecols = 'H:Y',
                       header = 2,
                       index_col = 0
                       )

    # Adjust Names
    sum_df.index.name = 'Building'
    sum_df.rename(columns={'Code.2':'Code','Building Code.1':'Building Code'}, inplace=True)
    sum_df = sum_df[sum_df['Building Name'].notna()]

    # Updating with detailed GFA.
    sum_df['GFA_commercial_area'] = sum_df['Commercial / Office Area (sq.m.)'].fillna(0)
    sum_df['GFA_aboveground_area'] = sum_df['GFA W/O Underground Floor'].fillna(sum_df['GFA'])
    sum_df['GFA_abovegrade_area_residential'] = sum_df['GFA_aboveground_area'] - sum_df['GFA_commercial_area'] 
    sum_df['GFA_residential'] = sum_df['GFA'] - sum_df['GFA_commercial_area']
    
    return sum_df


# --------------------------------------------------------------------------------------------------------
# ONTOLOGY TEMPLATE FUNCTIONS
def import_ontology_template_residential(path):
    """
    Imports the ontology template for residential buildings, adding variance and building code,
    as well as buildings currently under review.
    path = filepath to excel dataframe.
    """
    
    print('[Info] importing ontology...')
    o_template = pd.read_excel(io = path, 
                               sheet_name = "Ontology template", 
                               usecols = 'B:AQ',
                               header = 1,
                               index_col = 0
                               )

    # Filter less useful columns
    o_template = o_template[['Building Identifier', 'Floor', 'Code L5', 
                             'Code L4', 'CodeL3', 'CodeL2', 'CodeL1',
                             'MF # L1', 'MF # L2', 'MF # L3', 'MF # L4', 'MF # L5',
                             'Quantity 1', 'Uncertainty level', 
                             'GHG Quantity 1 Min', 'GHG Quantity 1 Max', 
                             #'GHG Quantity 1 Mean', 'GHG Quantity 1 St Dev', 
                             'GHG Quantity 1 Most Likely',
                             'CODE']]

    # calculate variance and building code.
    o_template = o_template.dropna(axis=0, subset=['Building Identifier','CODE']) # Clean code by removing empty rows.
    #o_template['GHG Variance'] = o_template['GHG Quantity 1 St Dev'] ** 2
    o_template['Building Key'] = o_template['CODE'].str[:3].apply(lambda x: int(x)) #if type(x) == str else x)

    #TEMPORARILY DROP ULSTER AVENUE
    #o_template = o_template[o_template['Building Key'] != 103]
    
    print('[Info] ontology import complete.')

    return o_template



def o_template_groupby(o_template, group_var, agg_dict, sum_df):
    """A function that groups ontology template by variable, 
       and aggregates based on a passed dictionary. Optional sum_df
       o_template = template to be grouped.
       group_var = variable to be grouped on.
       agg_dict = aggregation dictionary.
       sum_df = optional inclusion of building summary information."""
    
    #groupby
    o_temp_group = o_template.groupby(group_var).agg(agg_dict)
    
    if 'sum_df' in locals():
        o_temp_group['Code'] = sum_df['Building Code']
        o_temp_group['GFA'] = sum_df['GFA']
        o_temp_group['Beds'] = sum_df['Bedroom Qnt']
        o_temp_group['Floors'] = sum_df['Number of Floors Above Ground']
        o_temp_group['Units'] = sum_df['Unit Qnt']
    
    return o_temp_group



def label_mm_ontology(m_g, code_col_name, floors_col_name, units_col_name):
    """ 
    Labels rows in the ontology template based on missing middle and building type.
    m_g = ontology_template type dataframe.
    code_col_name = name of column where the codes are.
    floors_col_name = name of column where the # of floors is listed.
    units_col_name = name of column where the # of units is listed.
    """
    
    m_g['labels_specific'] = np.zeros
    m_g['labels_general'] = np.zeros
    
    conditions = [m_g[code_col_name].isin(['LNW']),
                  (m_g[code_col_name].isin(['LRM','MIX'])) & (m_g[floors_col_name] <= 4),
                  m_g[code_col_name].isin(['TWN','SMR','SMD']),
                  m_g[code_col_name].isin(['SND','SNR']),
                  m_g[floors_col_name] >= 5,
                 ]
    
    conditions_2 = [m_g[code_col_name].isin(['LNW']),
                  (m_g[code_col_name].isin(['LRM','MIX'])) & (m_g[floors_col_name] <= 4) & (m_g[units_col_name] <= 4),
                  m_g[code_col_name].isin(['SMR','SMD']),
                  m_g[code_col_name].isin(['SND','SNR']),
                  (m_g[code_col_name].isin(['TWN'])),
                  (m_g[code_col_name].isin(['LRM','MIX'])) & (m_g[floors_col_name] <= 4) & (m_g[units_col_name] > 4),
                  m_g[floors_col_name] >= 5,
                 ]
    
    
    labels_1 = ['Laneway', 'Low-Rise Multi-Unit', 'Duplex/Rowhouses', 'Single Family', 'Mid High Rise']
    labels_2 = ['Missing Middle', 'Missing Middle', 'Missing Middle', 'Single Family', 'Mid High Rise']
    
    labels_3 = ['Laneway', 'Multiplexes', 'Semi-Detached', 'Single Family', 'Rowhouses', 'Low-Rise Apartments', 'Mid High Rise']
    labels_4 = ['Laneway', 'Small MM', 'Small MM', 'Single Family', 'Large MM', 'Large MM', 'Mid High Rise']
    
    # insert specific labels
    for c, l in zip(conditions, labels_1):
        m_g.loc[c, 'labels_specific'] = l

    # Insert general labels
    for c, l in zip(conditions, labels_2):
        m_g.loc[c, 'labels_general'] = l
        
    # Insert mm_split_labels
    for c, l in zip(conditions_2, labels_3):
        m_g.loc[c, 'mm_split_labels'] = l
        
    # Insert small_large_mm_labels
    for c, l in zip(conditions_2, labels_4):
        m_g.loc[c, 'small_large_mm_labels'] = l
    
    return m_g