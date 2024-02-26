"""
This file contains classes and methods for
large-scale, combined sampling of houses, roads,
and water infrastructure.
"""
### Python packages
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import geopandas as gpd

### FIG-Canada functions
from .mm_functions_v1_7 import *
from .simulation import MonteCarlo
from .road_clean_sample import *
from .house_clean_sample import *
from .water_clean_sample import *

### Progress bar for jupyter notebook
import tqdm.notebook as tq

### Ouput supression functions
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stderr = old_stderr




class Sampler:
    """Object for sampling infrastructure in various ways."""

    def __init__(self, master='C:/Users/rankin6/'):
        """init to reduce user inputs"""
        self.sampler_province_name = 'ontario'
        self.emission_factor_option = 'GHG Quantity 1 Most Likely'
        self.road_type_map = {'Local / Street': 'Secondary', 
                              'Arterial': 'Primary',
                              'Collector': 'Secondary',
                              }
        
        # For material reduction sampling
        self.mat_red_path = master+'OneDrive - University of Toronto/Saxe - Rankin/Project 2. Housing Projections/FIG_Canada/data/proj/material reduction projections.xlsx'
        # Use default cols for net-zero pathway reductions, or use "AD:AK" for no change past 2030
        self.cols = "U:AB"
        

    def clean_all(self):
        """
        Clean house, road, and water data, return in a
        dictionary passable format for easy use.
        Modify clean all from the basic setup by modifying self variables.
        """
        # Def objects
        house_cleaner = HouseClean()
        road_cleaner = RoadClean()
        road_cleaner.province_name = self.sampler_province_name
        water_cleaner = WaterClean()

        # Run cleaning functions using self inputs
        house_clean = house_cleaner.import_process_house_data(self.emission_factor_option)
        roads_clean = road_cleaner.full_road_clean_map(user_type_map=self.road_type_map)
        water_clean = water_cleaner.import_water()

        # get provincial DA data
        house_sampler = HouseSample()
        house_sampler.province_name = self.sampler_province_name
        prov_da_s = house_sampler.get_prov_da(prov_name=house_sampler.province_name,
                                              shapefile_path=house_sampler.shapefile_path,
                                              census_data_path=house_sampler.path + 'da_census_data_reduced/' + house_sampler.prov_da_file_map[house_sampler.province_name]
                                              )

        # Return
        return {'houses': house_clean, 'roads': roads_clean, 'water': water_clean, 'da_data': prov_da_s}
    


    def clean_all_bic(self, quant=0.25):
        """
        Clean house, road, and water data, return in a
        dictionary passable format for easy use. SAMPLES ONLY BEST-IN-CLASS
        HOUSES FROM HOUSECLEAN ACCORDING TO SOME QUANTILE INPUT
        Modify clean all from the basic setup by modifying self variables.
        quant = quantile cutoff for house sampling (number between 0-1).
        """
        # Def objects
        house_cleaner = HouseClean()
        road_cleaner = RoadClean()
        road_cleaner.province_name = self.sampler_province_name
        water_cleaner = WaterClean()

        # Run cleaning functions using self inputs
        house_clean = house_cleaner.import_process_house_data(self.emission_factor_option)
        roads_clean = road_cleaner.full_road_clean_map(user_type_map=self.road_type_map)
        water_clean = water_cleaner.import_water()

        # get bic house data
        splits = house_clean['mm_split_labels'].unique()
        houses_clean_bic = []
        for s in splits:
            houses_clean_split = house_clean[house_clean['mm_split_labels'] == s]
            bic = houses_clean_split['ghg_per_unit'].quantile(quant)
            print(s)
            print(bic)
            print(houses_clean_split[houses_clean_split['ghg_per_unit'] <= bic].shape)
            houses_clean_bic.append(houses_clean_split[houses_clean_split['ghg_per_unit'] <= bic])

        house_clean_bic = pd.concat(houses_clean_bic)

        # get provincial DA data
        house_sampler = HouseSample()
        house_sampler.province_name = self.sampler_province_name
        prov_da_s = house_sampler.get_prov_da(prov_name=house_sampler.province_name,
                                              shapefile_path=house_sampler.shapefile_path,
                                              census_data_path=house_sampler.path + 'da_census_data_reduced/' + house_sampler.prov_da_file_map[house_sampler.province_name]
                                              )

        # Return
        return {'houses': house_clean_bic, 'roads': roads_clean, 'water': water_clean, 'da_data': prov_da_s}
    


    def clean_all_matred(self, year=2022):
        """
        FOR GIVEN YEAR AND REDUCED MATERIAL EMISSION FACTOR
        Clean house, road, and water data, return in a
        dictionary passable format for easy use.
        Modify clean all from the basic setup by modifying self variables.
        """
        # Def objects
        house_cleaner = HouseClean()
        road_cleaner = RoadClean()
        road_cleaner.province_name = self.sampler_province_name
        water_cleaner = WaterClean()

        # Run cleaning functions using self inputs
        house_clean = house_cleaner.import_process_house_data_matreduction(self.emission_factor_option, year=year)
        roads_clean = road_cleaner.full_road_clean_map(user_type_map=self.road_type_map)
        water_clean = water_cleaner.import_water()

        # reduce water_clean[ef] according to year
        mat_red_table = pd.read_excel(self.mat_red_path, sheet_name='material emissions', header=5, usecols=self.cols, index_col=0).iloc[:29].T
        water_m_map = ['steel','other','concrete','steel','plastic','plastic',
               'steel','other','steel','steel','other','concrete','other']
        water_clean['ef']['map'] = water_m_map
        water_clean['ef'] = water_clean['ef'].join(mat_red_table[year], on='map')
        water_clean['ef']['Minimum'] = water_clean['ef']['Minimum']*water_clean['ef'][year]
        water_clean['ef']['Most Likely'] = water_clean['ef']['Most Likely']*water_clean['ef'][year]
        water_clean['ef']['Maximum'] = water_clean['ef']['Maximum']*water_clean['ef'][year]
        water_clean['ef'] = water_clean['ef'].drop(['map', year], axis=1)

        # get provincial DA data
        house_sampler = HouseSample()
        house_sampler.province_name = self.sampler_province_name
        prov_da_s = house_sampler.get_prov_da(prov_name=house_sampler.province_name,
                                              shapefile_path=house_sampler.shapefile_path,
                                              census_data_path=house_sampler.path + 'da_census_data_reduced/' + house_sampler.prov_da_file_map[house_sampler.province_name]
                                              )

        # Return
        return {'houses': house_clean, 'roads': roads_clean, 'water': water_clean, 'da_data': prov_da_s}
    


    def clean_all_matred_bic(self, year=2022, quant=0.25):
        """
        FOR GIVEN YEAR AND REDUCED MATERIAL EMISSION FACTOR FOR ONLY BEST IN CLASS
        Clean house, road, and water data, return in a
        dictionary passable format for easy use.
        Modify clean all from the basic setup by modifying self variables.
        """
        # Def objects
        house_cleaner = HouseClean()
        road_cleaner = RoadClean()
        road_cleaner.province_name = self.sampler_province_name
        water_cleaner = WaterClean()

        # Run cleaning functions using self inputs
        house_clean = house_cleaner.import_process_house_data_matreduction(self.emission_factor_option, year=year)
        roads_clean = road_cleaner.full_road_clean_map(user_type_map=self.road_type_map)
        water_clean = water_cleaner.import_water()

        # reduce water_clean[ef] according to year
        mat_red_table = pd.read_excel(self.mat_red_path, sheet_name='material emissions', header=5, usecols=self.cols, index_col=0).iloc[:29].T
        water_m_map = ['steel','other','concrete','steel','plastic','plastic',
               'steel','other','steel','steel','other','concrete','other']
        water_clean['ef']['map'] = water_m_map
        water_clean['ef'] = water_clean['ef'].join(mat_red_table[year], on='map')
        water_clean['ef']['Minimum'] = water_clean['ef']['Minimum']*water_clean['ef'][year]
        water_clean['ef']['Most Likely'] = water_clean['ef']['Most Likely']*water_clean['ef'][year]
        water_clean['ef']['Maximum'] = water_clean['ef']['Maximum']*water_clean['ef'][year]
        water_clean['ef'] = water_clean['ef'].drop(['map', year], axis=1)

        # get bic house data
        splits = house_clean['mm_split_labels'].unique()
        houses_clean_bic = []
        for s in splits:
            houses_clean_split = house_clean[house_clean['mm_split_labels'] == s]
            bic = houses_clean_split['ghg_per_unit'].quantile(quant)
            print(s)
            print(bic)
            print(houses_clean_split[houses_clean_split['ghg_per_unit'] <= bic].shape)
            houses_clean_bic.append(houses_clean_split[houses_clean_split['ghg_per_unit'] <= bic])

        house_clean_bic = pd.concat(houses_clean_bic)

        # get provincial DA data
        house_sampler = HouseSample()
        house_sampler.province_name = self.sampler_province_name
        prov_da_s = house_sampler.get_prov_da(prov_name=house_sampler.province_name,
                                              shapefile_path=house_sampler.shapefile_path,
                                              census_data_path=house_sampler.path + 'da_census_data_reduced/' + house_sampler.prov_da_file_map[house_sampler.province_name]
                                              )

        # Return
        return {'houses': house_clean_bic, 'roads': roads_clean, 'water': water_clean, 'da_data': prov_da_s}



    def sample_basic_raw(self, i, **all_clean):
        """
        Sample infrastucture for i iterations.
        Modify clean all from the basic setup by modifying self variables.
        all_clean = dict output from Sampler.clean_all()
        """
        # Def objects    
        house_sampler = HouseSample()
        house_sampler.province_name = self.sampler_province_name
        road_sampler = RoadSample() 
        water_sampler = WaterSample()
        water_regressor = WaterRegression()

        # Sample houses
        house_samps = house_sampler.house_sample_run_efficient(houses_clean=all_clean['houses'], prov_da=all_clean['da_data'], iters=i)

        # Sample roads
        road_samps = road_sampler.road_sample_prep_and_run(roads_cleaned=all_clean['roads'], iters=i)

        # Sample water and regression
        all_water_samples = water_sampler.sample_all_water(iters=i, **all_clean['water'])
        water_samps = water_regressor.water_regression_pipeline(all_water_samples=all_water_samples,
                                                                roads_clean=all_clean['roads'],
                                                                prov_da=all_clean['da_data'])


        return {'house_samples': house_samps, 'road_samples': road_samps, 'water_samples': water_samps}
    


    def sample_basic(self, i, **all_clean):
        """
        Sample infrastucture for i iterations.
        Modify clean all from the basic setup by modifying self variables.
        all_clean = dict output from Sampler.clean_all()
        This full version of the function cleans and combines all of the sample,
        returning them in a neat format with some extra, useful info.
        """
        # Def objects    
        house_sampler = HouseSample()
        house_sampler.province_name = self.sampler_province_name
        road_sampler = RoadSample() 
        water_sampler = WaterSample()
        water_regressor = WaterRegression()

        # Sample houses
        house_samps = house_sampler.house_sample_run_efficient(houses_clean=all_clean['houses'], prov_da=all_clean['da_data'], iters=i)

        # Sample roads
        road_samps = road_sampler.road_sample_prep_and_run(roads_cleaned=all_clean['roads'], iters=i)

        # Sample water and regression
        all_water_samples = water_sampler.sample_all_water(iters=i, **all_clean['water'])
        water_samps = water_regressor.water_regression_pipeline(all_water_samples=all_water_samples,
                                                                roads_clean=all_clean['roads'],
                                                                prov_da=all_clean['da_data'])

        ## REFORMAT
        # Reformat so that individual iters can be extracted
        # Road reformat
        r = road_samps.sort_values('uq').groupby('uq').cumcount()
        r.name = 'iter'
        road_samps = pd.concat([road_samps.sort_values('uq'), r], axis=1)
        road_samps = road_samps.groupby(['iter', 'DAUID']).agg({'ghg':np.sum}).reset_index().rename(columns={'ghg':'road_ghg_pp'})
        road_samps = road_samps.set_index(road_samps['DAUID'].astype('string') + '_' + road_samps['iter'].astype('string'))

        # House reformat
        house_samps = house_samps.set_index(house_samps['DAUID'].astype('string') + '_' + house_samps['iter'].astype('string'))

        # Water reformat
        water_samps.name = 'water_inf_ghg_pp'
        col = ['DAUID','water_inf_ghg_pp']
        rs = []
        for c in col:
            r0 = pd.DataFrame(np.repeat(water_samps.reset_index()[c].to_numpy(), i, axis=0), columns=[c])
            rs.append(r0)
        water_samps = pd.concat(rs, axis=1)
        water_samps['iter'] = water_samps.groupby(water_samps['DAUID']).cumcount()
        water_samps = water_samps.set_index(water_samps['DAUID'].astype('string') + '_' + water_samps['iter'].astype('string'))

        # Join relevant columns, fix road ghg per person so it is actually per person.
        all_samps = house_samps.join([road_samps['road_ghg_pp'], water_samps['water_inf_ghg_pp']]).dropna().sort_values('iter')
        all_samps = all_samps.join(all_clean['da_data'].set_index('DAUID')[['tot_pd_check_count', 'pop_2021']], on='DAUID')
        all_samps['road_ghg_pp'] = all_samps['road_ghg_pp']/all_samps['pop_2021']

        return all_samps
    


    def sample_basic_matred(self, i, year=2022, **all_clean):
        """
        Sample infrastucture for i iterations. ALLOWS FOR MATERIAL REDUCTIONS.
        Modify clean all from the basic setup by modifying self variables.
        all_clean = dict output from Sampler.clean_all()
        This full version of the function cleans and combines all of the sample,
        returning them in a neat format with some extra, useful info.
        """
        # Def objects    
        house_sampler = HouseSample()
        house_sampler.province_name = self.sampler_province_name
        road_sampler = RoadSample() 
        water_sampler = WaterSample()
        water_regressor = WaterRegression()

        # Sample houses
        house_samps = house_sampler.house_sample_run_efficient(houses_clean=all_clean['houses'], prov_da=all_clean['da_data'], iters=i)

        # Sample roads
        # min/max/most likely mat values:
        asphalt_mm = np.array([0.07, 0.098])
        concrete_mmm = np.array([0.060967, 0.0814108, 0.10185])
        granular_mmm = np.array([0.00155, 0.00509, 0.00509])

        # Reduce the inputs to the year.
        mat_red_table = pd.read_excel(self.mat_red_path, sheet_name='material emissions', header=5, usecols=self.cols, index_col=0).iloc[:29].T
        asphalt_mm = asphalt_mm*mat_red_table.loc[['asphalt','concrete','other'], year]['asphalt']
        concrete_mmm = concrete_mmm*mat_red_table.loc[['asphalt','concrete','other'], year]['concrete']
        granular_mmm = granular_mmm*mat_red_table.loc[['asphalt','concrete','other'], year]['other']
        road_samps = road_sampler.road_sample_prep_and_run_choose(roads_cleaned=all_clean['roads'], 
                                                                  iters=i,
                                                                  asphalt_mm=asphalt_mm,
                                                                  concrete_mmm=concrete_mmm,
                                                                  granular_mmm=granular_mmm)
        
        # Adjust catchbasing emission factors
        water_sampler.catchbasin_e_facs['concrete'] = np.array(water_sampler.catchbasin_e_facs['concrete'])*mat_red_table.loc[['concrete','steel','other'], year]['concrete']
        water_sampler.catchbasin_e_facs['reinforcement'] = np.array(water_sampler.catchbasin_e_facs['reinforcement'])*mat_red_table.loc[['concrete','steel','other'], year]['steel']
        water_sampler.catchbasin_e_facs['granular'] = np.array(water_sampler.catchbasin_e_facs['granular'])*mat_red_table.loc[['concrete','steel','other'], year]['other']

        # Sample water and regression
        all_water_samples = water_sampler.sample_all_water(iters=i, **all_clean['water'])
        water_samps = water_regressor.water_regression_pipeline(all_water_samples=all_water_samples,
                                                                roads_clean=all_clean['roads'],
                                                                prov_da=all_clean['da_data'])

        ## REFORMAT
        # Reformat so that individual iters can be extracted
        # Road reformat
        r = road_samps.sort_values('uq').groupby('uq').cumcount()
        r.name = 'iter'
        road_samps = pd.concat([road_samps.sort_values('uq'), r], axis=1)
        road_samps = road_samps.groupby(['iter', 'DAUID']).agg({'ghg':np.sum}).reset_index().rename(columns={'ghg':'road_ghg_pp'})
        road_samps = road_samps.set_index(road_samps['DAUID'].astype('string') + '_' + road_samps['iter'].astype('string'))

        # House reformat
        house_samps = house_samps.set_index(house_samps['DAUID'].astype('string') + '_' + house_samps['iter'].astype('string'))

        # Water reformat
        water_samps.name = 'water_inf_ghg_pp'
        col = ['DAUID','water_inf_ghg_pp']
        rs = []
        for c in col:
            r0 = pd.DataFrame(np.repeat(water_samps.reset_index()[c].to_numpy(), i, axis=0), columns=[c])
            rs.append(r0)
        water_samps = pd.concat(rs, axis=1)
        water_samps['iter'] = water_samps.groupby(water_samps['DAUID']).cumcount()
        water_samps = water_samps.set_index(water_samps['DAUID'].astype('string') + '_' + water_samps['iter'].astype('string'))

        # Join relevant columns, fix road ghg per person so it is actually per person.
        all_samps = house_samps.join([road_samps['road_ghg_pp'], water_samps['water_inf_ghg_pp']]).dropna().sort_values('iter')
        all_samps = all_samps.join(all_clean['da_data'].set_index('DAUID')[['tot_pd_check_count', 'pop_2021']], on='DAUID')
        all_samps['road_ghg_pp'] = all_samps['road_ghg_pp']/all_samps['pop_2021']

        return all_samps
    


    ## --AGGREGATE FUNCTIONS--
    # Build a function that reads single iterations of sample_basic() into a file,
    # Then combine in a function with the clean data
    def sample_and_read_inf(self, x, it, name='inf_mc_samples.csv', head=True, **all_clean):
        """
        Read MC samples to file in loop.
        all_clean = dict output from Sampler.clean_all()
        x = number of loops.
        it = number of iterations per loop.
        name = name of file.
        Generally a larger ratio of iters to x is faster (vectorized),
        but memory will run up with higher iter numbers. Balance these
        to your machine.
        """
        for j in tq.tqdm(range(0,x), position=0, leave=True):
            # Run
            inf_samps = self.sample_basic(i=it, **all_clean)
            # Some formatting
            inf_samps['DAUID'] = inf_samps['DAUID'].astype(int)
            inf_samps = inf_samps.drop(['iter','pop_2021', 'tot_pd_check_count'], axis=1).set_index('DAUID')
            if head == False:
                inf_samps.to_csv(name, mode='a', header=False)
            else:
                inf_samps.to_csv(name, mode='a', header=(j==0))

        # return last sample to confirm func works.
        return inf_samps
    

    # This method is the same as sample_and_read_inf(), but incorporates ability to reduce
    # material emission factors based on the year.
    def sample_and_read_inf_matred(self, x, it, name='inf_mc_samples.csv', head=True, year=2022, **all_clean):
        """
        Read MC samples to file in loop.
        all_clean = dict output from Sampler.clean_all()
        x = number of loops.
        it = number of iterations per loop.
        name = name of file.
        Generally a larger ratio of iters to x is faster (vectorized),
        but memory will run up with higher iter numbers. Balance these
        to your machine.
        """
        for j in tq.tqdm(range(0,x), position=0, leave=True):
            # Run
            inf_samps = self.sample_basic_matred(i=it, year=year, **all_clean)
            # Some formatting
            inf_samps['DAUID'] = inf_samps['DAUID'].astype(int)
            inf_samps = inf_samps.drop(['iter','pop_2021', 'tot_pd_check_count'], axis=1).set_index('DAUID')
            if head == False:
                inf_samps.to_csv(name, mode='a', header=False)
            else:
                inf_samps.to_csv(name, mode='a', header=(j==0))

        # return last sample to confirm func works.
        return inf_samps



    def clean_sample_read(self, x, i, name='inf_mc_samples.csv', head=True):
        """
        Perform full clean -> sample -> read to file pipeline
        """
        # clean
        clean = self.clean_all()
        # sample and read
        self.sample_and_read_inf(x=x, i=i, name=name, head=head, **clean)