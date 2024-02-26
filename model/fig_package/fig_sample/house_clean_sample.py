"""
This class file contains functions that process housing data and 
sample iterations of the road data using methods from MonteCarlo()
class instances. Functions are developed in house_sampling_iter.ipynb
"""

### Imports
import numpy as np
import scipy as sp
import pandas as pd
import geopandas as gpd
from .simulation import MonteCarlo
from .mm_functions_v1_7 import *



class HouseClean:
    """
    Class processes housing data after it has been imported using
    the importer class. It contains methods that do all required
    prep for the HouseSampling class.
    """

    def __init__(self, master='C:/Users/rankin6/'):
        """
        init
        o_path = initial path to ontology template
        """
        self.o_path = master+'OneDrive - University of Toronto/Keagan/6. Dataset/Dataset_2.0__02282023.xlsx'
        # For material reduction sampling
        self.mat_red_path = master+'/OneDrive - University of Toronto/Saxe - Rankin/Project 2. Housing Projections/FIG_Canada/data/proj/material reduction projections.xlsx'
        # Use default cols for net-zero pathway reductions, or use "AD:AK" for no change past 2030
        self.cols = "U:AB"


    def ontology_import_and_clean(self, ontology_path):
        """
        Uses the missing middle paper functions to import the ontology template,
        removes undesired materials and returns with building key.
        """
        # Ontology template material information.
        
        #uni_map, master_map = load_master_uni(4, 4, ontology_path)
        sum_df = import_sum_df(ontology_path)
        # Load the data from the Ontology Template.
        o_template = import_ontology_template_residential(ontology_path)
        # Drop identifier
        o_template = o_template.drop(columns=['Building Identifier'])
        # Filter so that the ontology template only includes residential buildings (with bedrooms).
        o_template = o_template[o_template['Building Key'].isin(sum_df[sum_df['Bedroom Qnt'] > 0].index)]
        # Find building code
        o_template['Building Key'] = o_template['CODE'].str[:3].astype(int)
        o_template['Building Key'].unique().shape

        print(o_template.shape)

        # Plastic Windows MF #3: 08 53 00
        # Flooring MF #3: 09 65 00
        # Pavers MF #4: 09 30 19
        mf_drop3 = ['08 53 00', '09 65 00', '09 30 00', '09 30 19']

        # Drop some Uniformat item
        # Grey water tanks UF CodeL3: D2030.10
        # F is a swimming pool Code L3: F1050.10
        # Parking Bumpers Code L4: G2020.40
        uf_drop_l4 = ['D2030.10', 'F1050.10' ,'G2020.40']

        def filter_rows_by_values(df, col, values):
            return df[~df[col].isin(values)]

        o_template = filter_rows_by_values(o_template, 'MF # L3', mf_drop3)
        o_template = filter_rows_by_values(o_template, 'Code L4', uf_drop_l4)

        print(o_template.shape)

        return (o_template, sum_df)



    def o_temp_label(self, o_template, sum_df):
        """
        Labels o_template with min, max, most likely GHG values.
        Also labels o_template with mm_label information so that
        different housing forms can be seperated when running mc.
        """
        # Get min max most likely cols
        o_template['min_ghg_factor'] = o_template['GHG Quantity 1 Min'] / o_template['Quantity 1']
        o_template['most_likely_ghg_factor'] = o_template['GHG Quantity 1 Most Likely'] / o_template['Quantity 1']
        o_template['max_ghg_factor'] = o_template['GHG Quantity 1 Max'] / o_template['Quantity 1']

        # Label the ontology template with building form and units.
        o_template_labeled = o_template.join(sum_df[['Code','Number of Floors Above Ground','Unit Qnt','Building Code']], on='Building Key')

        # Rename some columns
        o_template_labeled['Units'] = o_template_labeled['Unit Qnt'].fillna(1)
        o_template_labeled = label_mm_ontology(o_template_labeled, 'Building Code', 'Number of Floors Above Ground', 'Units')

        return o_template_labeled



    def o_temp_g_init_midhigh_total(self, o_template_l, ghg_col, fully_q_keys, conversion_func):
        """
        Returns o_temp_g required for monte carlo iters that
        do not sample GHG emissions randomly. ALSO...
        Converts midhigh rise concrete ghg values into estimated total ghg
        using some conversion function (e.g. linear model, piecewise, bayesian)
        o_template = labeled ontology template.
        ghg_col = column containing ghg values that need converted (min max most likely etc.).
        fully_q_keys = list of building keys of fully quantified mid/high rise (unmodified by func).
        conversion_func = function used to total ghg from concrete ghg. Should return a VECTOR
            of estimates of full-building ghg.
        """
        # Init a df for performing function and one where we drop all non_fully quantified mid high and will
        # append the new results + return below.
        o_manipulate = o_template_l
        o_return = o_template_l[(o_template_l['mm_split_labels'] != 'Mid High Rise') | (o_template_l['Building Key'].isin(fully_q_keys))]

        # Isolate concrete cols in non_fully quantified
        o_manipulate = o_manipulate[(o_template_l['mm_split_labels'] == 'Mid High Rise') & 
                                    (o_template_l['MF # L1'] == '03 00 00') & 
                                    (~o_template_l['Building Key'].isin(fully_q_keys))]
        
        # o_template label and aggregate
        aggregation = {
                    ghg_col:'sum',
                    'Units':'first',
                    'Number of Floors Above Ground':'first',
                    'mm_split_labels':'first',
                    'labels_general':'first',
                    }
        
        o_manipulate_g = o_manipulate.groupby("Building Key").agg(aggregation)

        # Apply the conversion function
        o_manipulate_g[ghg_col] = conversion_func(o_manipulate_g, ghg_col)

        # Group the rest of the buildings, append the concrete_only mid high rise back on
        o_return_g = o_return.groupby("Building Key").agg(aggregation)
        o_temp_g = pd.concat([o_return_g, o_manipulate_g]).drop(['Number of Floors Above Ground','conc_ghg_per_floor'], axis=1)

        # Other
        o_temp_g['ghg_per_unit'] = o_temp_g[ghg_col] / o_temp_g['Units']

        return o_temp_g



    def ghgperfloor_conversion_func(self, conc_o_template, ghg_col):
        """
        Estimates total emissions from mid high rise buildings that were not
        fully quantified using concrete/floor linear fitting to non-concrete
        emissions.
        conc_o_template = concrete_only_ontology template of mid high rise
        ghg_col = name of column with ghg values.
        x1, c = params
        """

        # Get column of x
        conc_o_template['conc_ghg_per_floor'] = conc_o_template[ghg_col] / conc_o_template['Number of Floors Above Ground']

        # Model params
        c = 3.392e+05
        x1 = 4.1568

        #apply to each row: this linear model relates concrete per floor to non-concrete emissions.
        estimate = conc_o_template['conc_ghg_per_floor'].apply(lambda x: x*x1 + c)
        #print(estimate)

        total_estimate = estimate + conc_o_template[ghg_col]
        #print(total_estimate)

        return total_estimate
    


    ## -- Aggregate Methods --
    def import_process_house_data(self, ghg_factor_col):
        """
        AGGREGATE METHOD (uses above methods)
        Performs full processing loop on ontology data,
        importing, cleaning, labelling, and interpolating
        midhigh total.
        ghg_factor_col = ghg factor column from ontology template to use (min, max, or most likely)
        """
        # Import and Clean
        imps = self.ontology_import_and_clean(self.o_path)

        # Label
        print('2/3: Labelling house data.')
        o_temp_l = self.o_temp_label(imps[0], imps[1])

        # Interpolate
        print('3/3: Interpolate total ghg for midhigh rise.')
        o_temp_g = self.o_temp_g_init_midhigh_total(o_temp_l, ghg_factor_col, [77,95,112], self.ghgperfloor_conversion_func)

        print('Complete, Returning grouped Housing Data')
        return(o_temp_g)
    


    def import_process_house_data_matreduction(self, ghg_factor_col='GHG Quantity 1 Most Likely', year=2022):
        """
        AGGREGATE METHOD (uses above methods),
        Allows for reductions in house material emissions based on tech
        improvements in the given year.
        Performs full processing loop on ontology data,
        importing, cleaning, labelling, and interpolating
        midhigh total.
        ghg_factor_col = ghg factor column from ontology template to use (min, max, or most likely)
        """
        # Import and Clean
        imps = self.ontology_import_and_clean(self.o_path)

        ## Label
        print('2/3: Labelling house data.')
        o_temp_l = self.o_temp_label(imps[0], imps[1])

        ## Reduce emissions based on year.
        print('2.5/3: Reducing emission factors.')
        # Import info
        mat_red_table = pd.read_excel(self.mat_red_path, sheet_name='material emissions', header=5, usecols=self.cols, index_col=0).iloc[:29].T
        emissions_house_map = pd.read_excel(self.mat_red_path, sheet_name='mat emissions map', header=0, usecols="K:N")#.iloc[:,3:]
        emissions_house_map.columns = ['Minimum', 'Maximum', 'Most likely', 'Map Name']
        emissions_house_map = emissions_house_map.round(5)
        emissions_house_map['stringjoin'] = emissions_house_map['Minimum'].astype(str) + emissions_house_map['Maximum'].astype(str) + emissions_house_map['Most likely'].astype(str)

        # But first need to reduce material factor.
        o_temp_l['stringjoin'] = round(o_temp_l['min_ghg_factor'], 5).astype(str) + round(o_temp_l['max_ghg_factor'], 5).astype(str) + round(o_temp_l['most_likely_ghg_factor'], 5).astype(str)

        # Join mapping name to o_template
        o_temp_t = o_temp_l.join(emissions_house_map[['Map Name','stringjoin']].set_index('stringjoin'), on='stringjoin').drop('stringjoin', axis=1)
        o_temp_t['Map Name'] = o_temp_t['Map Name'].fillna('other')

        # Here is how I can join and adjust ghg quantity 1 most likely when I run the sampling
        o_temp_t = o_temp_t.join(mat_red_table[year], on='Map Name')
        o_temp_t[ghg_factor_col] = o_temp_t[ghg_factor_col] * o_temp_t[year]

        ## Interpolate
        print('3/3: Interpolate total ghg for midhigh rise.')
        o_temp_g = self.o_temp_g_init_midhigh_total(o_temp_t, ghg_factor_col, [77,95,112], self.ghgperfloor_conversion_func)

        ## fix mid/high not going to zero problem (becuase of regression intercept)
        if year == 2050:
            o_temp_g['ghg_per_unit'] = 0

        print('Complete, Returning grouped Housing Data')
        return(o_temp_g)




class HouseSample:
    """
    Class contains methods that perform Monte Carlo sampling of
    houses in Ontario DAs. Different aggregate methods are also
    supplied.
    """

    def __init__(self, master='C:/Users/rankin6/'):
        """init"""

        self.path = master+'OneDrive - University of Toronto/Saxe - Rankin/Project 2. Housing Projections/FIG_Canada/data/houses/'
        self.shapefile_path = self.path + 'lda_000b21a_e.shp'
        self.province_name = 'ontario'

        # Map census data file name with province name
        self.prov_da_file_map = {'newfoundland':'2021_atlantic_census_data_housing.csv',
                                'nova_scotia':'2021_atlantic_census_data_housing.csv',
                                'new_brunswick':'2021_atlantic_census_data_housing.csv',
                                'pei':'2021_atlantic_census_data_housing.csv',
                                'quebec':'2021_quebec_census_data_housing.csv',
                                'ontario':'2021_ontario_census_data_housing.csv',
                                'manitoba':'2021_prairies_census_data_housing.csv',
                                'saskatchewan':'2021_prairies_census_data_housing.csv',
                                'alberta':'2021_prairies_census_data_housing.csv',
                                'british_columbia':'2021_britishcolumbia_census_data_housing.csv',
                                'yukon':'2021_territories_census_data_housing.csv',
                                'nwt':'2021_territories_census_data_housing.csv',
                                'nunavut':'2021_territories_census_data_housing.csv',
                                }
        
        #self.census_data_path = self.path + 'da_census_data_reduced/' + prov_da_file_map[self.province_name]


    ### Meant to be run independtly before the sampling process, unlike with roads, as 
    ### the data can be used independently for plotting as well.  
    def get_prov_da(self, prov_name, shapefile_path, census_data_path, dropna=True):
        """
        given province, return the organized census data for the data.
        Similar to FIG-Ontario importer function that returned ont_da:
        Imports, formats, and merges dissemination area shapefile with relevant census data on housing.
        shape_path = filepath for DA shapefile starting from data folder.
        data_path = filepath for raw census data (see ont_census_da_works) starting from data folder.
        """
        # Import the census shapefile
        print('[I] Reading DA Shapefile...')
        canada_da = gpd.read_file(shapefile_path)

        #province to DAUID first two numbers map
        da_name_map = {'newfoundland':'10',
                    'nova_scotia':'12',
                    'new_brunswick':'13',
                    'pei':'11',
                    'quebec':'24',
                    'ontario':'35',
                    'manitoba':'46',
                    'saskatchewan':'47',
                    'alberta':'48',
                    'british_columbia':'59',
                    'yukon':'60',
                    'nwt':'61',
                    'nunavut':'62',
                    }

        province_da = canada_da[canada_da['DAUID'].str.startswith(da_name_map[prov_name])]
        #toronto_da = canada_da[canada_da['DAUID'].str.startswith('3520')]


        # Import the census data
        print('[I] Importing census data...')
        census_d = pd.read_csv(census_data_path)

        # We add a column that names the different census characteristics/vars (from census documentation)
        char_ids = [1, 4, 41, 42, 43, 44, 45, 46, 47, 48, 49, 57]
        char_n = ['pop_2021','private_dwellings','tot_pd_check_count',
        'single_detached','semi_detached','rowhouse','flat_duplex','lowrise_apartment','mid_high_rise','other','mobile',
        'avg_household_size']

        char_hash = dict(zip(char_ids, char_n))

        census_d['CHAR_NAME'] = census_d['CHARACTERISTIC_ID'].map(char_hash)


        # Pivot census df for merging, change some keys, join on DGUID.
        print('[I] Merging data...')
        census_d_pivot = pd.pivot_table(census_d, index=['DGUID'], columns='CHAR_NAME', values='C1_COUNT_TOTAL')

        prov_da_census = province_da.set_index('DGUID').join(census_d_pivot)


        # Add some columns that are useful: per person normalized houses and population density of DAs
        print('[I] Adding useful columns...')
        prov_da_census['pop_km2'] = prov_da_census['pop_2021']/prov_da_census['LANDAREA']

        prov_da_census['plex_per_person'] = prov_da_census['flat_duplex']/prov_da_census['pop_2021']
        prov_da_census['row_per_person'] = prov_da_census['rowhouse']/prov_da_census['pop_2021']
        prov_da_census['lowrise_per_person'] = prov_da_census['lowrise_apartment']/prov_da_census['pop_2021']
        prov_da_census['midhigh_per_person'] = prov_da_census['mid_high_rise']/prov_da_census['pop_2021']
        prov_da_census['semi_per_person'] = prov_da_census['semi_detached']/prov_da_census['pop_2021']
        prov_da_census['single_per_person'] = prov_da_census['single_detached']/prov_da_census['pop_2021']
        prov_da_census['missing_middle_per_person'] = prov_da_census[['semi_detached', 
                                                                    'rowhouse','lowrise_apartment', 'flat_duplex']].sum(axis=1)/prov_da_census['pop_2021']
        
        if dropna == True:
            prov_da_census = prov_da_census.dropna()


        print('[C] Returning... Complete.')
        return prov_da_census



    # Basic MC Procedure to be looped in another function
    def prep_sample_house_generic(self, houses_clean, prov_da, hc_form, oa_form, x):
        """ 
        Basic x iteration MC sampling procedure for given housing form.
        houses_clean = output from house_processor.import_process_house_data()
        prov_da = given province da file from imp.import_da_ontario() -> imp.import_da_province()
        hc_form = name of form in houses_clean dataframe.
        oa_form = name of form in ont_da dataframe.
        """
        # rename
        # Isolate required columns.
        ont_da_form = prov_da[['DAUID', 'pop_2021', oa_form]].reset_index(drop=True)
        ont_da_form = ont_da_form.astype(int)

        # Convert to long form (one row per house of the particular form)
        ont_da_form_long = pd.DataFrame(ont_da_form.values.repeat(ont_da_form[oa_form], axis=0), 
                                columns=ont_da_form.columns, 
                                dtype=np.int32
                                )

        # We actually dont even need the pop column, can just turn it into the uq id.
        rep_idx = np.arange(0, ont_da_form_long.shape[0])
        ont_da_form_long[oa_form] = rep_idx

        # Duplicate rows using fast np implementation.
        rs = []
        for c in ont_da_form_long.columns:
            r0 = pd.DataFrame(np.repeat(ont_da_form_long[c].to_numpy(), x, axis=0), 
                            columns=[c], dtype=np.int32)
            rs.append(r0)
        r_d = pd.concat(rs, axis=1)

        # Fit housing form to distribution, sample 
        simulator = MonteCarlo()
        fitted = simulator.fit_bounded_distribution(houses_clean[houses_clean['mm_split_labels'] == hc_form]['ghg_per_unit'], low_bound=10)
        ghg = simulator.bounded_distribution(fitted['d_object'], fitted['params'], r_d.shape[0])
        ghg_pp = np.divide(ghg, r_d['pop_2021'])

        # Reformat, add and groupby da+unique iter count to downsize df
        #print(r_d.columns)
        r_d = r_d.drop('pop_2021', axis=1)
        r_d['house_ghg_pp'] = ghg_pp
        r_d['iter'] = r_d.groupby(oa_form).cumcount()

        r_d_g = r_d.groupby(['DAUID','iter']).agg({'house_ghg_pp':np.sum}).reset_index()

        return r_d_g
    

    ### FULL RUN FUNCTIONS
    def house_sample_run_efficient(self, houses_clean, prov_da, iters):
        """
        Apply house_sample_generic optimizing for at runtime by
        vectorizing.
        houses_clean = output from house_processor.import_process_house_data().
        prov_da = given province da file from imp.import_da_ontario() -> imp.import_da_province()
        x = number of iterations.
        """
        # Inits
        print('function benchmarked at 10 iters/25 seconds. Try loop-writing into a file at higher iters to avoid memory issues.')
        houses_clean_forms = ['Single Family', 'Mid High Rise', 'Semi-Detached', 'Rowhouses','Low-Rise Apartments', 'Multiplexes']
        ont_da_forms = ['single_detached', 'mid_high_rise', 'semi_detached', 'rowhouse','lowrise_apartment', 'flat_duplex']

        # Apply generic function to each form
        split_samples = []
        for hc, oa in zip(houses_clean_forms, ont_da_forms):
            print('Sampling form: ', hc)
            form_sample = self.prep_sample_house_generic(houses_clean, prov_da, hc, oa, iters)
            form_sample = form_sample.set_index('DAUID')
            form_sample = form_sample.rename(columns={'house_ghg_pp':'house_ghg_pp'+'_'+oa})
            split_samples.append(form_sample)

        # Concat results. Group by like DA and iter to eliminate NA.
        all_form_samps = pd.concat(split_samples)
        all_form_samps = all_form_samps.groupby(['DAUID', 'iter']).agg(np.sum).reset_index()

        # Sum all ghg together, remake dataframe for returning.
        house_ghg = all_form_samps.iloc[:,2:].sum(axis=1)
        all_form_samps = pd.DataFrame({'DAUID':all_form_samps['DAUID'],
                                    'iter': all_form_samps['iter'],
                                    'house_ghg_pp':house_ghg})

        return all_form_samps
    


    def house_sample_run_efficient_with_import(self, houses_clean, iters):
        """
        Apply house_sample_generic optimizing for at runtime by
        vectorizing. Imports prov_da within the function!
        houses_clean = output from HouseClean.import_process_house_data().
        prov_da = given province da file from imp.import_da_ontario() -> imp.import_da_province()
        x = number of iterations.
        """
        # Inits
        print('function benchmarked at 10 iters/25 seconds. Try loop-writing into a file at higher iters to avoid memory issues.')
        houses_clean_forms = ['Single Family', 'Mid High Rise', 'Semi-Detached', 'Rowhouses','Low-Rise Apartments', 'Multiplexes']
        ont_da_forms = ['single_detached', 'mid_high_rise', 'semi_detached', 'rowhouse','lowrise_apartment', 'flat_duplex']
        prov_da = self.get_prov_da(self.province_name, self.shapefile_path, 
                                   self.path + 'da_census_data_reduced/' + self.prov_da_file_map[self.province_name], dropna=True)

        # Apply generic function to each form
        split_samples = []
        for hc, oa in zip(houses_clean_forms, ont_da_forms):
            print('Sampling form: ', hc)
            form_sample = self.prep_sample_house_generic(houses_clean, prov_da, hc, oa, iters)
            form_sample = form_sample.set_index('DAUID')
            form_sample = form_sample.rename(columns={'house_ghg_pp':'house_ghg_pp'+'_'+oa})
            split_samples.append(form_sample)

        # Concat results. Group by like DA and iter to eliminate NA.
        all_form_samps = pd.concat(split_samples)
        all_form_samps = all_form_samps.groupby(['DAUID', 'iter']).agg(np.sum).reset_index()

        # Sum all ghg together, remake dataframe for returning.
        house_ghg = all_form_samps.iloc[:,2:].sum(axis=1)
        all_form_samps = pd.DataFrame({'DAUID':all_form_samps['DAUID'],
                                    'iter': all_form_samps['iter'],
                                    'house_ghg_pp':house_ghg})

        return all_form_samps