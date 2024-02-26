"""
This file contains classes and methods for
using neighbourhood samples to forecast future
Canadian emissions under different growth scenarios.
"""
### Python packages
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import geopandas as gpd
import tqdm.notebook as tq

from .fig_sample.road_clean_sample import *
from .fig_sample.house_clean_sample import *
from .fig_helper.helper import *


class Forecast:

    def __init__(self):
        """
        inits for scenarios and types of forecasts
        """
        self.province_name = 'ontario'
        self.master = 'C:/Users/Keagan Rankin/'


    ### Some functions that process samples into a appropriate format for forecasting.
    def neighbourhood_filter(self, samp, prov_da, sf_lh, mm_lh, mh_lh):
        """
        Filter neighbourhoods such given the
        scenario (BAU, best-in-class etc.).
        Our best tools for this will be the 
        % of each housing type as well as density.
        samp = from sample_all_infrastructure().
        prov_da = province da data (for percentages), with DAUID
        and housing count data.
        other inputs = 2 element list/tuple of low and high
        percentage range to filter DAs by, inclusive. [0.0, 1.0]
        sf -> single family.
        mm -> missing middle.
        mh -> mid high rise.
        """
        # Calculate ont_da percentages.
        prov_da['sf_p'] = (prov_da['single_per_person']*prov_da['pop_2021'])/prov_da['tot_pd_check_count']
        prov_da['mm_p'] = (prov_da['missing_middle_per_person']*prov_da['pop_2021'])/prov_da['tot_pd_check_count']
        prov_da['mh_p'] = (prov_da['midhigh_per_person']*prov_da['pop_2021'])/prov_da['tot_pd_check_count']

        # Filter ont_da DAs by percentages
        prov_da = prov_da[(prov_da['sf_p'] >= sf_lh[0]) & (prov_da['sf_p'] <= sf_lh[1])]
        prov_da = prov_da[(prov_da['mm_p'] >= mm_lh[0]) & (prov_da['mm_p'] <= mm_lh[1])]
        prov_da = prov_da[(prov_da['mh_p'] >= mh_lh[0]) & (prov_da['mh_p'] <= mh_lh[1])]

        f_das = prov_da['DAUID']

        # Filter samples by the DAs meeting the category
        samp_f = samp[samp['DAUID'].isin(f_das)]

        return samp_f
    

    def neighbourhood_filter_quantile(self, samp, prov_da, h_type='sf', q=0.9):
        """
        Filter neighbourhoods, but based on a quantile
        of housing type percentage. e.g. take only neighbourhoods
        in the top 90th percentile of single family percentage
        of a given province.
        samp = Monte Carlo sample.
        prov_da = province da data (for percentages), with DAUID
        and housing count data.
        q = quantile.
        type:
        sf -> single family.
        mm -> missing middle.
        mh -> mid high rise.
        """
        # Calculate ont_da percentages.
        prov_da['sf_p'] = (prov_da['single_per_person']*prov_da['pop_2021'])/prov_da['tot_pd_check_count']
        prov_da['mm_p'] = (prov_da['missing_middle_per_person']*prov_da['pop_2021'])/prov_da['tot_pd_check_count']
        prov_da['mh_p'] = (prov_da['midhigh_per_person']*prov_da['pop_2021'])/prov_da['tot_pd_check_count']

        # Filter prov_da DAs by quantile of housing type percentage
        filter_percentage = prov_da[h_type+'_p'].quantile(q)
        print('quantile percentage: ', filter_percentage)
        prov_da = prov_da[(prov_da[h_type+'_p'] >= filter_percentage)]
        f_das = prov_da['DAUID']

        # Filter samples by the DAs meeting the category
        samp_f = samp[samp['DAUID'].isin(f_das)]

        return samp_f


    def iter_add(self, samp, n_iters=500):
        """
        organize the province samples, add an iter column.
        samp = sample
        n_iters = number of iterations in the sample
        """
        samp = samp.sort_values('DAUID')
        samp['iter'] = np.tile(np.arange(0,n_iters), samp['DAUID'].unique().size)
        return samp


    def join_pop(self, samp, prov_da):
        """add DA population to convert per-capita emissions to totals."""
        prov_da['DAUID'] = prov_da['DAUID'].astype(int)
        samp = samp.join(prov_da[['DAUID','tot_pd_check_count']].set_index('DAUID'), on='DAUID')
        return samp


    def filter_road(self, samp, road_filt_index):
        """filter DAs with excessive road lengths.
        road_filt_index = index of roads that are to remain (from Helper.drop_outliers_iqr())"""
        samp = samp[samp['DAUID'].isin(road_filt_index)]
        return samp



    ### These functions are the building blocks of the overall forecast
    def build_from_samples_generic(self, it, all_samps, starts_y, step_size=5, check_tolerance=1000):
        """
        Use house, road, water samples to simulate
        construction of new neighbourhoods for 'it' iterations
        all_samps = sample with iter added with iter_add().
        starts_y = starts in new neighbourhoods in the given year.
        step_size = number of DAs built per cycle after
        informative initial value.
        check_tolerance = tolerance about starts_y value.
        """
        # For each iteration, sample (aka "build" a neighbourhood),
        # check if we have reached yearly required amount, if not up sample size.
        # if we exceed the yearly amount cut back the sample size
        built_samps = []
        built_count = []
        for j in range(0, it):
            print('building iter: ', j)
            #iter_built_samp = []

            # Informative starting sample size based on mean 272 # homes per DA (E(x) of # of samples given mean DAs).
            # Define a low and high number about the yearly starts that we will stop our sampling at.
            # the most extreme DAs have house numbers ~1000 so a 2000 range tolerance should not bias the sample.
            start = int(starts_y/272)
            size_check = 0
            check_low = starts_y - check_tolerance
            check_high = starts_y + check_tolerance

            halter = False
            while halter == False:
                # perform inital sample with mean start.
                if size_check == 0:
                    iter_built_samp = all_samps[all_samps['iter'] == j].sample(n=start, replace=True)

                # check if start is <= 1 (aka starts_y is lower than avg number of 1 DA)
                # if so, sample just one neighbourhood and continue.
                if start <= 1:
                    iter_built_samp = all_samps[all_samps['iter'] == j].sample(n=1, replace=True)
                    halter = True
                
                # if size_check is greater than our tolerance, drop some number of last rows from the sample.
                elif size_check > check_high:
                    iter_built_samp = iter_built_samp.drop(iter_built_samp.index[-step_size:])

                # if size_check is greater than our tolerance, append new samples.
                elif size_check < check_low:
                    iter_built_samp = pd.concat([iter_built_samp, all_samps[all_samps['iter'] == j].sample(n=step_size, replace=True)])

                # if size_check is in the tolerance, move on.
                else:
                    halter = True
                
                # increase size check
                size_check = iter_built_samp['tot_pd_check_count'].sum()
                #print(size_check)
            
            built_samps.append(iter_built_samp)
            built_count.append(size_check)
        
        # Combine yearly output data.
        return pd.concat(built_samps)
    

    def infill_from_samples_generic(self, it, all_samps, infills_y, step_size=5, check_tolerance=500, k=0.8):
        """
        Infill function performing infill sim.
        it = number of iterations
        infills_y = number of units to infill.
        all_samps = sample with iter added with iter_add().
        step_size = sampling step size.
        check_tolerance = tolerance +- desired # of houses to be infilled that year.
        k = circularity constant.
        """
        # Run build generic function
        built_samps = self.build_from_samples_generic(it=it, all_samps=all_samps, starts_y=infills_y, 
                                                      step_size=step_size, check_tolerance=check_tolerance)
        
        # Modify for infill:
        built_samps[['road_ghg_pp','water_inf_ghg_pp']] = 0
        built_samps['house_ghg_pp'] = built_samps['house_ghg_pp'] * k 

        return built_samps
    


    ## ------------
    # BUSINESS AS USUAL EXPERIMENT

    def build_from_samples_bau(self, it, all_samps, starts_y, prov_da, percent_vector, step_size=5, check_tolerance=1000):
        """
        BUSINESS AS USUAL VARIANT: Checks
        to make sure totals of housing form are within the
        percentage range built in recent years.

        Use house, road, water samples to simulate
        construction of new neighbourhoods.
        it = number of iterations in passed all_samps.
        all_samps = from sample_all_infrastructure() saved files.
        starts_y = number of starts in given year.
        prov_da = da data
        percent_vector = vector of BAU breakdown for SF, MM, HR homes.
        step_size = number of DAs built per cycle after
        informative initial value.
        check_tolerance = tolerance about starts_y value.
        """

        prov_da['missing_middle'] = prov_da['missing_middle_per_person']*prov_da['pop_2021']
        # For each iteration, sample (aka "build" a neighbourhood),
        # check if we have reached yearly required amount, if not up sample size.
        # if we exceed the yearly amount cut back the sample size
        built_samps = []
        #built_count = []
        for j in range(0, it):
            print('building iter: ', j)

            # Informative starting sample size based on mean 272 # homes per DA (E(x) of # of samples given mean DAs).
            # Define a low and high number about the yearly starts that we will stop our sampling at.
            # the most extreme DAs have house numbers ~1000 so a 2000 range tolerance should not bias the sample.
            start = int(starts_y/272)
            size_check_sf = 0
            size_check_mm = 0
            size_check_hr = 0

            # initialize checks for each of the three forms. We want to land in the percentage range
            # of the last ten years of construction.
            check_low_sf = (starts_y*percent_vector[0]) - check_tolerance
            check_high_sf = (starts_y*percent_vector[0]) + check_tolerance
            check_low_mm = (starts_y*percent_vector[1]) - check_tolerance
            check_high_mm = (starts_y*percent_vector[1]) + check_tolerance
            check_low_hr = (starts_y*percent_vector[2]) - check_tolerance
            check_high_hr = (starts_y*percent_vector[2]) + check_tolerance

            # init the iter data frame to sample from here so we can append ont_da info
            # without using a ton of memory
            all_samps_i = all_samps[all_samps['iter'] == j]
            all_samps_i = all_samps_i.join(prov_da[['DAUID','single_detached','missing_middle','mid_high_rise']].set_index('DAUID'), on='DAUID')

            halter = False
            while halter == False:
                # perform inital sample with mean start.
                if (size_check_sf+size_check_mm+size_check_hr) == 0:
                    iter_built_samp = all_samps_i.sample(n=start, replace=True)
                
                # if size_check is greater than our tolerance, drop some number of last rows from the sample.
                elif (size_check_sf > check_high_sf) | (size_check_mm > check_high_mm) | (size_check_hr > check_high_hr):
                    iter_built_samp = iter_built_samp.drop(iter_built_samp.index[-step_size:])#[:step_size]) running with taking first 5 away take much longer.

                # if size_check is greater than our tolerance, append new samples.
                elif (size_check_sf < check_low_sf) | (size_check_mm < check_low_mm) | (size_check_hr < check_low_hr):
                    iter_built_samp = pd.concat([iter_built_samp, all_samps_i.sample(n=step_size, replace=True)])

                # if size_check is in the tolerance, move on.
                else:
                    halter = True
                
                # increase size check
                size_check_sf = iter_built_samp['single_detached'].sum()
                size_check_mm = iter_built_samp['missing_middle'].sum()
                size_check_hr = iter_built_samp['mid_high_rise'].sum()
                #print(size_check)
            
            built_samps.append(iter_built_samp)
            #built_count.append(size_check)
        
        # Combine yearly output data.
        return pd.concat(built_samps)#, built_count
    


    def infill_from_samples_bau(self, it, all_samps, prov_da, percent_vector, infills_y, step_size=5, check_tolerance=500, k=0.8):
        """
        Infill function performing infill sim.
        it = number of iterations
        infills_y = number of units to infill.
        all_samps = sample with iter added with iter_add().
        step_size = sampling step size.
        check_tolerance = tolerance +- desired # of houses to be infilled that year.
        k = circularity constant.
        """
        # Run build generic function
        # print('infill func it:', it)
        built_samps = self.build_from_samples_bau(it=it, all_samps=all_samps, starts_y=infills_y, prov_da=prov_da, 
                                                  percent_vector=percent_vector, step_size=step_size, check_tolerance=check_tolerance)
        
        # Modify for infill:
        built_samps[['road_ghg_pp','water_inf_ghg_pp']] = 0
        built_samps['house_ghg_pp'] = built_samps['house_ghg_pp'] * k 

        return built_samps
    # ------------



    

    ### Aggregate functions: running the overall forecast for some given infill rate and inputs.
    def prep_sample(self, samp, n_iters, prov_da, road_filter_index, sf, mm, hr):
        """preps raw mc samples for forecast."""

        samp = self.iter_add(samp=samp, n_iters=n_iters)
        samp = self.join_pop(samp=samp, prov_da=prov_da)
        samp = self.filter_road(samp=samp, road_filt_index=road_filter_index)
        samp_n = self.neighbourhood_filter(samp=samp, prov_da=prov_da, sf_lh=sf, mm_lh=mm, mh_lh=hr)

        return samp_n
    

    def prep_sample_quantile(self, samp, n_iters, prov_da, road_filter_index, h_type='sf', q=0.9):
        """preps raw mc samples for forecast."""

        samp = self.iter_add(samp=samp, n_iters=n_iters)
        samp = self.join_pop(samp=samp, prov_da=prov_da)
        samp = self.filter_road(samp=samp, road_filt_index=road_filter_index)
        samp_n = self.neighbourhood_filter_quantile(samp=samp, prov_da=prov_da, h_type=h_type, q=q)

        return samp_n
    


    def prep_sample_with_imports(self, samp, sf=[0.0, 1.0], mm=[0.0, 1.0], hr=[0.0, 1.0], iters=500):
        """
        performs sample prep with all required imports, extreme outlier road drop.
        iters = number of MC iterations in the sample file!
        """
        # import mcs samples and prov da info
        #bc_mc = pd.read_csv('C:/Users/Keagan Rankin/OneDrive - University of Toronto/Saxe - Rankin/Project 2. Housing Projections/FIG_Canada/mc_samples/bc/inf_mc_samples_bc.csv')
        hs = HouseSample(master=self.master)
        hs.province_name = self.province_name
        prov_da = hs.get_prov_da(prov_name=self.province_name, shapefile_path=hs.shapefile_path, 
                                    census_data_path=hs.path + 'da_census_data_reduced/' + hs.prov_da_file_map[self.province_name], dropna=True)

        # road clean filter extreme outliers
        rc = RoadClean(master=self.master)
        rc.province_name = self.province_name
        roads_clean = rc.full_road_clean_map()
        road_filt = roads_clean.groupby('DAUID').agg({'LENGTH_GEO':np.sum})
        handy = Helper()
        road_filt = handy.drop_outliers_iqr(road_filt, 'LENGTH_GEO', f=3)
        road_filt.index

        samps_p = self.prep_sample(samp=samp, n_iters=iters, prov_da=prov_da, road_filter_index=road_filt.index, 
                                    sf=sf, mm=mm, hr=hr)
        
        return samps_p
    


    def prep_sample_with_imports_quantile(self, samp, h_type='sf', q=0.9, iters=500):
        """
        performs sample prep with all required imports, extreme outlier road drop.
        iters = number of MC iterations in the sample file!
        """
        # import mcs samples and prov da info
        #bc_mc = pd.read_csv('C:/Users/Keagan Rankin/OneDrive - University of Toronto/Saxe - Rankin/Project 2. Housing Projections/FIG_Canada/mc_samples/bc/inf_mc_samples_bc.csv')
        hs = HouseSample(master=self.master)
        hs.province_name = self.province_name
        prov_da = hs.get_prov_da(prov_name=self.province_name, shapefile_path=hs.shapefile_path, 
                                    census_data_path=hs.path + 'da_census_data_reduced/' + hs.prov_da_file_map[self.province_name], dropna=True)

        # road clean filter extreme outliers
        rc = RoadClean(master=self.master)
        rc.province_name = self.province_name
        roads_clean = rc.full_road_clean_map()
        road_filt = roads_clean.groupby('DAUID').agg({'LENGTH_GEO':np.sum})
        handy = Helper()
        road_filt = handy.drop_outliers_iqr(road_filt, 'LENGTH_GEO', f=3)
        road_filt.index

        samps_p = self.prep_sample_quantile(samp=samp, n_iters=iters, prov_da=prov_da, road_filter_index=road_filt.index, 
                                            h_type=h_type, q=q)
        
        return samps_p



    def matflow_generic(self, samps_p, starts, step=(5,5), it=1, infill_percent=0.35, k=0.8):
        """
        runs n-year construction forecast using samples for a given province.
        samps_p = prepped infrastructure MC samples. 
        Should have tot_pd_count_check joined from ont_da.
        starts = np array of housing starts for each year of
        simulation (eventually convert to uncertain draw).
        ont_da = ont_da dataframe from Import.import_da_ontario()
        step = simulation step size (hyper parameter for simulation speed).
        format tuple, first element = build_from_samples(), 2nd element = infill_from_samples()
        it = number of Monte Carlo iterations (to be sampled each year).
        infill_percent = percent infill in medium-sized municipalities in Ontario.
        k = infill circularity constant.
        """
        yearly_build_data = []

        for y in tq.tqdm(np.arange(0, len(starts))):
            print('year: ', y)

            # New Starts + Infill yearly initialization
            starts_y = starts[y]
            new_built_y = starts_y*(1-infill_percent)
            infill_y = starts_y*(infill_percent)

            # Simulate new construction from MC samples.
            print('building new...')
            if new_built_y > 0:
                x_year_build_samps = self.build_from_samples_generic(it=it, all_samps=samps_p, starts_y=new_built_y, 
                                                                    step_size=step[0], check_tolerance=1000)
                x_year_build_samps['year'] = y
            # Simulation new infill from MC samples.
            print('infilling...')
            x_year_infill_samps = self.infill_from_samples_generic(it=it, all_samps=samps_p, infills_y=infill_y, 
                                                                   step_size=step[1], check_tolerance=1000, k=k)
            x_year_infill_samps['year'] = y

            # Append yearly data to list.
            if new_built_y > 0:
                yearly_build_data.append(x_year_build_samps)
            yearly_build_data.append(x_year_infill_samps)
 
        return yearly_build_data
    


    def matflow_yearly(self, samps_pl, starts, step=(5,5), it=1, infill_percent=0.35, k=0.8):
        """
        runs n-year construction forecast using samples for a given province.
        WITH MC SAMPLES THAT CHANGE EACH YEAR.
        samps_pl = list of prepped infrastructure MC samples for each year of starts. 
        Should have tot_pd_count_check joined from ont_da.
        starts = np array of housing starts for each year of
        simulation (eventually convert to uncertain draw).
        ont_da = ont_da dataframe from Import.import_da_ontario()
        step = simulation step size (hyper parameter for simulation speed).
        format tuple, first element = build_from_samples(), 2nd element = infill_from_samples()
        it = number of Monte Carlo iterations (to be sampled each year).
        infill_percent = percent infill in medium-sized municipalities in Ontario.
        k = infill circularity constant.
        """
        yearly_build_data = []

        for y in tq.tqdm(np.arange(0, len(starts))):
            print('year: ', y)

            # New Starts + Infill yearly initialization
            starts_y = starts[y]
            # if no starts that year, continue
            if starts_y < 1:
                continue
            new_built_y = starts_y*(1-infill_percent)
            infill_y = starts_y*(infill_percent)

            # Simulate new construction from MC samples.
            print('building new...')
            x_year_build_samps = self.build_from_samples_generic(it=it, all_samps=samps_pl[y], starts_y=new_built_y, 
                                                                step_size=step[0], check_tolerance=1000)
            x_year_build_samps['year'] = y
            # Simulation new infill from MC samples.
            print('infilling...')
            x_year_infill_samps = self.infill_from_samples_generic(it=it, all_samps=samps_pl[y], infills_y=infill_y, 
                                                                step_size=step[1], check_tolerance=1000, k=k)
            x_year_infill_samps['year'] = y

            # Append yearly data to list.
            yearly_build_data.append(x_year_build_samps)
            yearly_build_data.append(x_year_infill_samps)
 
        return yearly_build_data
    

    def matflow_bau(self, samps_p, starts, prov_da, percent_vector, step=(5,5), it=1, infill_percent=0.35, k=0.8):
        """
        runs n-year construction forecast using samples for a given province.
        samps_p = prepped infrastructure MC samples. 
        Should have tot_pd_count_check joined from ont_da.
        starts = np array of housing starts for each year of
        simulation (eventually convert to uncertain draw).
        ont_da = ont_da dataframe from Import.import_da_ontario()
        step = simulation step size (hyper parameter for simulation speed).
        format tuple, first element = build_from_samples(), 2nd element = infill_from_samples()
        it = number of Monte Carlo iterations (to be sampled each year).
        infill_percent = percent infill in medium-sized municipalities in Ontario.
        k = infill circularity constant.
        """
        yearly_build_data = []
        # check tolerance harcoded for provinces in BAU
        check_tol = [700, 700, 700, 700, 1000, 1000, 800, 800, 900, 1000]

        for y in tq.tqdm(np.arange(0, len(starts))):
            print('year: ', y)

            # New Starts + Infill yearly initialization
            starts_y = starts[y]
            new_built_y = starts_y*(1-infill_percent)
            infill_y = starts_y*(infill_percent)

            # Simulate new construction from MC samples.
            print('building new...')
            if new_built_y > 0:
                x_year_build_samps = self.build_from_samples_bau(it=it, all_samps=samps_p, starts_y=new_built_y, prov_da=prov_da, percent_vector=percent_vector,
                                                                    step_size=step[0], check_tolerance=check_tol[y])
                x_year_build_samps['year'] = y
            # Simulation new infill from MC samples.
            print('infilling...')
            # print('matflow it:', it)
            x_year_infill_samps = self.infill_from_samples_bau(it=it, all_samps=samps_p, infills_y=infill_y, prov_da=prov_da, percent_vector=percent_vector, 
                                                                   step_size=step[1], check_tolerance=check_tol[y], k=k)
            x_year_infill_samps['year'] = y

            # Append yearly data to list.
            if new_built_y > 0:
                yearly_build_data.append(x_year_build_samps)
            yearly_build_data.append(x_year_infill_samps)
 
        return yearly_build_data