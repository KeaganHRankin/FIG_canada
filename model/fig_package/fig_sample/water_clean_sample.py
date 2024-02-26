"""
This class file contains functions that process water data and 
sample iterations of the road data using methods from MonteCarlo()
class instances. Functions are developed in water_sampling_iter.ipynb
"""

### Imports
import numpy as np
import scipy as sp
import pandas as pd
from .simulation import MonteCarlo
from fig_package.fig_helper.helper import Helper
from sklearn.ensemble import RandomForestRegressor


class WaterClean:
    """
    Object stores methods for importing (and processing in the future)
    pre-processed water data for various DAs in Ontario (see FIG-Ontario processing .ipynb files). 
    It also imports a df of emission factors and densities needed in Monte Carlo sampling.
    """

    def __init__(self, master='C:/Users/rankin6/'):
        """
        init
        path_w = path to water data not including the file name.
        path_o = path to material emission data.
        """
        #self.path =  "C:/Users/Keagan Rankin/OneDrive - University of Toronto/Saxe - Rankin/Project 2. Housing Projections/Data/"
        self.path =  master+"OneDrive - University of Toronto/Saxe - Rankin/Project 2. Housing Projections/FIG_Canada/data/"
        self.path_w = self.path + 'water/'
        self.path_o = self.path + 'other/'



    def import_water(self):
        """
        returns data in one large hashmap.
        """
        watermains = pd.read_csv(self.path_w+'watermains_abs_thickness.csv', index_col=0).reset_index(drop=True)
        sewers = pd.read_csv(self.path_w+'sewers_abs_thickness.csv', index_col=0).reset_index(drop=True)
        stormwaters = pd.read_csv(self.path_w+'stormwaters_abs_thickness.csv', index_col=0).reset_index(drop=True)
        catchbasins = pd.read_csv(self.path_w+'catchbasin.csv', index_col=0).reset_index(drop=True)

        mat_emit_db_pipes = pd.read_csv(self.path_w+'pipe_material_emission_factors.csv', index_col=0).reset_index(drop=True)

        return {'w':watermains, 'sw':sewers, 'st':stormwaters, 'cb':catchbasins, 'ef':mat_emit_db_pipes}
    



class WaterSample:
    """
    Stores methods for sampling watermain, sewer, and stormwater pipes + catchbasins
    for multiple Monte Carlo iters.
    """

    def __init__(self):

        # Catcbasin unqiue emission factors (see development file from FIG-Ontario)
        self.catchbasin_e_facs = {'concrete': [0.069454, 0.095546, 0.118937],
                                  'reinforcement': [0.7783, 1.852755, 1.85276],
                                  'granular': [0.00155, 0.00509, 0.00509]}

    

    def mc_iter_pipe_init(self, x, pipe_df, cols):
        """
        Given x number of monte carlo iterations, duplicates, appends, and returns
        expanded input vectors x times so that vectorize simulation can be performed.
        cols = columns to be duplicated
        """
        # Add an extra unique identifier
        unique = np.arange(pipe_df.shape[0])
        pipe_df['uq'] = unique
        #cols = cols.append('uq')
        p_df = pipe_df[cols]

        # Duplicate rows using faster np implementation.
        rs = []
        for c in cols:
            r0 = pd.DataFrame(np.repeat(p_df[c].to_numpy(), x, axis=0), columns=[c])
            rs.append(r0)

        r_d = pd.concat(rs, axis=1)

        return r_d
    


    def sample_pipe_generic(self, pipe_df, pert_factors, density, all_round=False):
        """
        Monte Carlo sampling to propogate uncertainty in
        water pipe networks. Performs vectorized sampling.
        pipe_df = pipe dataframe of form/col name == cleaned
        pipe data ('adj_len','thickness', 'shape'). Single material.
        pert_factors = low, ml, high mat emission factors for
        pipe material to be fit to pert dist.
        density = material density.
        all_round = simplifies process if all pipes are round.
        default False (all pipes not round).
        """

        simu = MonteCarlo()
        # init areas (mm2) depending on rectangular or circular pipes.
        if all_round == True:
            outer_diam = pipe_df['diameter']
            inner_diam = pipe_df['diameter'] - 2*pipe_df['thickness']
            areas = np.pi*((outer_diam**2)-(inner_diam**2))/4
        
        else:
            circ_df = pipe_df[pipe_df['shape'] == 'CIRC']
            rect_df = pipe_df[pipe_df['shape'] == 'RECT']

            outer_diam = circ_df['height']
            inner_diam = circ_df['height'] - 2*circ_df['thickness']
            areas_circ = np.pi*((outer_diam**2)-(inner_diam**2))/4

            areas_rect = rect_df['height']*rect_df['width'] - ((rect_df['height']-2*rect_df['thickness']) * (rect_df['width']-2*rect_df['thickness']))

            areas = pd.concat([areas_circ, areas_rect])
            pipe_df = pd.concat([circ_df, rect_df])


        # calc volume of segment from adjusted len, convert to m^3 then kg
        volumes = (areas*10**-6) * pipe_df['adj_len']
        masses = volumes * density

        #Sample PERT for GHG values
        pert_samples = simu.pert(a=pert_factors['Minimum'].item(), b=pert_factors['Most Likely'].item(), c=pert_factors['Maximum'].item(), 
                                size=masses.shape)
        
        ghgs = masses * pert_samples

        pipe_df['water_ghg'] = ghgs

        return pipe_df[['DAUID', 'uq', 'unique_id', 'water_ghg']]
    


    def water_sample_run_efficiently(self, pipe_df, pipe_emit_density, all_round=False):
        """
        Runs the pipe sampling efficiently by splitting the
        entire database into smaller dfs with single materials
        so that emit factor multiplications are a vectorize op.
        pipe_df = full pipe df of cleaned form.
        pipe_emit_density = df with min, ml, max, and density cols
        for all mats in pipe_df.
        """
        # split into multiple dfs.
        water_df_split_list = []
        mats_in_frame = []
        for m in pipe_emit_density['Material']:
            water_df_split = pipe_df[pipe_df['material'] == m]

            if water_df_split.empty == False:
                water_df_split_list.append(water_df_split)
                mats_in_frame.append(m)
            
            else: 
                print('mat ', m, ' not in frame.')

        # apply generic function to each split df.
        split_samples = []
        for split_df, m in zip(water_df_split_list, mats_in_frame):
            print('iter: ', m)
            samp = self.sample_pipe_generic(pipe_df=split_df,
                                    pert_factors=pipe_emit_density[pipe_emit_density['Material'] == m],
                                    density=pipe_emit_density[pipe_emit_density['Material'] == m]['Density'].item(),
                                    all_round=all_round)
            
            split_samples.append(samp)

        return pd.concat(split_samples)
    



    ### Catchbasin functions
    def cb_sample_single(self, n):
        """
        Calculate the volume/mass/ghg of materials
        in single catch basins using Ontario standard design and
        random (discrete uniform) sampling of standard heights.
        standard_sizes = list of standard heights.
        n = number of samples.
        """
        # init
        standard_sizes = [1980, 1830, 1520, 1380, 1680]
        sh = np.random.choice(standard_sizes, n)
        out_hole_size = 525

        # volume of concrete.
        v_basin_conc_upper = ((830-600)**2)*(sh-450)
        v_basin_conc_lower = ((830*450)-(np.pi*300**2/2))*830
        v_basin_conc_neg = (np.pi*(200/2)**2*(830-600)) + (np.pi*(out_hole_size/2)**2*(830-600))
        v_basin_conc = (v_basin_conc_upper + v_basin_conc_lower - v_basin_conc_neg) * 10**-9

        # volume of steel reinforcement.
        v_basin_steel_vert = ((185*.715)*2 + (185*(.715+.150))*2) * (sh-100) * 2
        v_basin_steel_bot = ((185*.715) * 715) + (185*(0.715+0.3) * 715)
        v_basin_steel_neg = (185*.715)*2 * ((out_hole_size+200)/2 * np.sqrt(2))  # square circle approximation of d
        v_basin_steel = (v_basin_steel_vert  + v_basin_steel_bot - v_basin_steel_neg) * 10 ** -9

        # volume of aggregate.
        v_basin_gran = ((830+600)**2 * 150) * 10 ** -9
        v_basin_gran = np.full(n, v_basin_gran)


        return (v_basin_conc, v_basin_steel, v_basin_gran)



    def cb_sample_double(self, n):
        """
        Calculate the volume/mass/ghg of materials
        in single catch basins using Ontario standard design and
        random (discrete uniform) sampling of standard heights.
        standard_sizes = list of standard heights.
        n = number of samples.
        """
        # init
        standard_sizes = [1980, 1830, 1680]
        sh = np.random.choice(standard_sizes, n)
        out_hole_size = 525

        # volume of concrete.
        v_basin_conc_upper = ((830-600)*(1680-1450) - (250*115*2))*(sh-450)
        v_basin_conc_lower = ((830*450)-(np.pi*300**2/2))*1680
        v_basin_conc_beam = (230*375*830) + (230*25*600)
        v_basin_conc_neg = (np.pi*(200/2)**2*(830-600)) + (np.pi*(out_hole_size/2)**2*(830-600))
        v_basin_conc = (v_basin_conc_upper + v_basin_conc_lower + v_basin_conc_beam - v_basin_conc_neg) * 10**-9


        # volume of steel reinforcement.
        # 10M = 100mm2, #15M = 200mm2
        v_basin_steel_vert = ((185*.715)*2 + (185*(.715+.150))*2) * (sh-100) * 2
        v_basin_steel_vert = v_basin_steel_vert * 2 - ((185*.715)*2 * (sh-100) * 2)
        v_basin_steel_bot = ((185*.715) * 715) + (185*(0.715+0.3) * 715)
        v_basin_steel_neg = (185*.715)*2 * ((out_hole_size+200)/2 * np.sqrt(2))  # square circle approximation of d
        v_basin_steel_beam = 2*100*(830-100) + 2*200*(830-100) + 3*(100*(275+275+130))
        v_basin_steel = (v_basin_steel_vert  + v_basin_steel_bot + v_basin_steel_beam - v_basin_steel_neg) * 10 ** -9

        # volume of aggregate.
        v_basin_gran = ((1680+600)**2 * 150) * 10 ** -9
        v_basin_gran = np.full(n, v_basin_gran)

        return (v_basin_conc, v_basin_steel, v_basin_gran)
    


    def sample_catchbasin_generic(self, catchbasin_df, sample_f, e_f):
        """
        Sample catchbasins equal to
        number in the dataframe given.
        sample_f = sampling function from above.
        e_f = min, ml, max hash of emission factors for materials with names
        concrete, reinforcement, granular.
        """
        simu = MonteCarlo()
        # Sample according to length
        n = catchbasin_df.shape[0]
        v_samples = sample_f(n)

        # calc mass and ghg
        density = [np.full(n,2400), np.full(n, 7850), np.full(n, 2.4*10**3)]
        conc_f = simu.pert(e_f['concrete'][0], e_f['concrete'][1], e_f['concrete'][2], n)
        steel_f = simu.pert(e_f['reinforcement'][0], e_f['reinforcement'][1], e_f['reinforcement'][2], n)
        gran_f = simu.pert(e_f['granular'][0], e_f['granular'][1], e_f['granular'][2], n)

        ghg = (v_samples[0]*density[0]*conc_f) + (v_samples[1]*density[1]*steel_f) + (v_samples[2]*density[2]*gran_f)


        catchbasin_df['catchbasin_ghg'] = ghg

        return catchbasin_df



    def catchbasin_sample_run_efficient(self, catchbasin_df, e_f):
        """
        Split by double and sinlge catchbasins, perform sampling procedure.
        """
        cb_df_split_list = []
        for b in catchbasin_df['basin_type'].unique():
            cb_df_split = catchbasin_df[catchbasin_df['basin_type'] == b]
            cb_df_split_list.append(cb_df_split)
        
        split_samples = []
        for split_df in cb_df_split_list:
            basin_type = split_df['basin_type'].iloc[0]
            print(basin_type)
            print('iter ', basin_type)
            if basin_type == 'single':
                samp = self.sample_catchbasin_generic(split_df, self.cb_sample_single, e_f)
            else:
                samp = self.sample_catchbasin_generic(split_df, self.cb_sample_double, e_f)

            split_samples.append(samp)

        return pd.concat(split_samples)
    


    ### AGGREGATE FUNCTIONS
    # Function runs all water infrastructure and returns n samples.
    def sample_all_water(self, iters, **water_clean):
                         #watermain_df, sewer_df, storm_df, catchbasin_df, pipe_ef):
        """
        Sample all water infrastructure 
        using the above functions n times.
        iters = number of samples
        water_clean = dictionary output from WaterProcess.import_water() method.
        """

        print('\nSampling watermains')
        cols_r = ['DAUID', 'unique_id', 'uq', 'diameter', 'material', 'adj_len', 'thickness']
        water_s = self.water_sample_run_efficiently(self.mc_iter_pipe_init(iters, water_clean['w'], cols_r), water_clean['ef'], True)

        print('\nSampling sewers')
        cols_nr = ['DAUID', 'unique_id', 'uq', 'height', 'width', 'shape', 'material', 'adj_len', 'thickness']
        sewer_s = self.water_sample_run_efficiently(self.mc_iter_pipe_init(iters, water_clean['sw'], cols_nr), water_clean['ef'], False)

        print('\nSampling stormwaters')
        storm_s = self.water_sample_run_efficiently(self.mc_iter_pipe_init(iters, water_clean['st'], cols_nr), water_clean['ef'], False)


        print('\nSampling catchbasins')
        cb_s = self.catchbasin_sample_run_efficient(self.mc_iter_pipe_init(iters, water_clean['cb'], ['DAUID', 'uq', 'basin_type', 'JOIN_FID']), self.catchbasin_e_facs)

        print('\ncomplete, returning tuple')

        return (water_s, sewer_s, storm_s, cb_s)
    



class WaterRegression:
    """
    This class stores methods required to
    fit a regression on available water 
    infrastructure data to the rest of Ontario/sampling space.
    """

    def __init__(self, master='C:/Users/rankin6/'):
        # Some default vals for regression objects.
        #self.model = RandomForestRegressor(max_features=0.521, max_samples=0.836)
        self.model = RandomForestRegressor(criterion='friedman_mse', max_samples=0.8780763793053599,
                                           min_samples_leaf=4, n_estimators=1000)
        self.features = ['pop_km2', 'secondary_len', 'primary_len', 'primary_lanes', 
                         'midhigh_per_person', 'single_per_person', 
                         #'other_cities', 'rural_counties'
                         ]
        
        # training data
        self.train_path = master+'OneDrive - University of Toronto/Saxe - Rankin/Project 2. Housing Projections/FIG_Canada/data/water/water_ml_regression_train.csv'
        self.train = pd.read_csv(self.train_path, index_col=0)

        # relevant DA population data
        self.pop_path = master+'OneDrive - University of Toronto/Saxe - Rankin/Project 2. Housing Projections/FIG_Canada/data/water/water_ml_regression_pop2021.csv'
        self.population = pd.read_csv(self.pop_path, index_col=0)


    def water_regression_clean(self, all_water_samples, roads_clean, ont_da, aggr, features):
        """
        WARNING: DEPRECIATED, FIG-ONTARIO ONLY FUNCTION. USE water_regression_prep INSTEAD.
        USE THIS FUNCTION WITH ONTARIO DATA TO SAVE TRAINING DATASET.
        This function takes DA data and outputs
        from the water MC sampling and returns them
        in a format that can be passed to the fit_predict
        regression function to predict water ghg from
        DAs without data.
        all_water_samples = output from water_sampler function.
        roads_clean = output from method RoadProcess().import_clean_road_data.
        ont_da = data from class method Importer.import_da_ontario.
        aggr = function for water sample aggregation.
        features = desired features (from self.features).
        returns X_train, y_train, and a prediction set.
        """
        # Group water samples by aggr function
        wm_samples_g = all_water_samples[0].groupby('uq').agg({'DAUID':'first',
                                                        'water_ghg':aggr})
        sw_samples_g = all_water_samples[1].groupby('uq').agg({'DAUID':'first',
                                                            'water_ghg':aggr})
        st_samples_g = all_water_samples[2].groupby('uq').agg({'DAUID':'first',
                                                            'water_ghg':aggr})
        cb_samples_g = all_water_samples[3].groupby('uq').agg({'DAUID':'first',
                                                            'catchbasin_ghg':aggr})
        

        # Append watermain data to ont_da data, add all together into one column
        ont_da['DAUID'] = ont_da['DAUID'].astype('int')
        # Add watermains
        da_water = ont_da.join(wm_samples_g.groupby('DAUID').sum(), on='DAUID', rsuffix='_wm')
        da_water['wm_ghg_pp'] = da_water['water_ghg']/da_water['pop_2021']

        # Add sewers
        da_water = da_water.join(sw_samples_g.groupby('DAUID').sum(), on='DAUID', rsuffix='_sw')
        da_water['sw_ghg_pp'] = da_water['water_ghg_sw']/da_water['pop_2021']

        # Add stormwater
        da_water = da_water.join(st_samples_g.groupby('DAUID').sum(), on='DAUID', rsuffix='_st')
        da_water['st_ghg_pp'] = da_water['water_ghg_st']/da_water['pop_2021']

        # Add catchbasins
        da_water = da_water.join(cb_samples_g.groupby('DAUID').sum(), on='DAUID')
        da_water['cb_ghg_pp'] = da_water['catchbasin_ghg']/da_water['pop_2021']

        da_water_f = da_water.dropna(subset=['water_ghg','water_ghg_sw','water_ghg_st','catchbasin_ghg'], how='all')
        da_water_f['water_inf_ghg_pp'] = da_water_f[['wm_ghg_pp','sw_ghg_pp','st_ghg_pp','cb_ghg_pp']].sum(axis=1)

        # Add geoclass
        handy = Helper()
        da_water_f = handy.geo_class_add(da_water_f, 'DAUID')

        
        # Create road features from roads_clean
        road_features = roads_clean.groupby(['DAUID','ROADCLASS']).agg({'LENGTH_GEO':np.sum,
                                                                    'NBRLANES':np.median}).reset_index()
        road_features = road_features.join(pd.get_dummies(road_features['ROADCLASS'], drop_first=True))

        # DEPRECIATED FROM FIG-ONTARIO
        #road_features = road_features.pivot(index='DAUID', columns='ROADCLASS', values=['LENGTH_GEO', 'NBRLANES']).fillna(0)
        #road_features = road_features.droplevel(0, axis=1)
        #road_features.columns = ['highway_len', 'local_len', 'primary_len', 'secondary_len',
        #                        'highway_lanes', 'local_lanes', 'primary_lanes', 'secondary_lanes']
        
        road_features_lanes = road_features.pivot(index='DAUID', columns='ROADCLASS', values=['NBRLANES']).fillna(0)
        road_features_length = road_features.pivot(index='DAUID', columns='ROADCLASS', values=['LENGTH_GEO',]).fillna(0)

        road_features_length = road_features_length.droplevel(0, axis=1)
        road_features_length.columns = [col_name.lower()+'_len' for col_name in road_features_length.columns]
        road_features_lanes = road_features_lanes.droplevel(0, axis=1)
        road_features_lanes.columns = [col_name.lower()+'_lanes' for col_name in road_features_lanes.columns]

        road_features = pd.concat([road_features_length, road_features_lanes], axis=1)

        # Append road features to the da_water dataframe.
        da_water_r = da_water_f.join(road_features, on='DAUID')
        da_water_r = da_water_r.join(pd.get_dummies(da_water_r['geo_class']))

        # Inverse hyperbolic sine transform on features, split data.
        # DEPRECIATED FROM FIG-ONTARIO
        #featsf = ['pop_km2', 'secondary_len', 'local_len', 'primary_len', 
        #    'local_lanes', 'primary_lanes', 'avg_household_size', 'midhigh_per_person', 
        #    'single_per_person', 'other_cities', 'rural_counties', 'water_inf_ghg_pp']
        featsf = self.features + ['water_inf_ghg_pp']

        da_water_r_sh = np.arcsinh(da_water_r[featsf]).dropna()
        X_t = da_water_r_sh.loc[:, da_water_r_sh.columns != 'water_inf_ghg_pp']
        y_t = da_water_r_sh['water_inf_ghg_pp']

        # Also return complimentary set (DAs without water data).
        da_water['water_inf_ghg_pp'] = da_water[['wm_ghg_pp','sw_ghg_pp','st_ghg_pp','cb_ghg_pp']].sum(axis=1)
        da_water_nof = da_water[da_water['water_inf_ghg_pp'] == 0]
        da_water_nof = da_water_nof.join(road_features, on='DAUID')
        da_water_nof = handy.geo_class_add(da_water_nof, 'DAUID')
        da_water_nof = da_water_nof.join(pd.get_dummies(da_water_nof['geo_class']))
        X_com = np.arcsinh(da_water_nof[features].dropna())

        return X_t[features], y_t, X_com
    

    def water_regression_prep(self, all_water_samples, roads_clean, prov_da, aggr):
        """
        Function takes housing and road data for a particular province
        and water samples. It prepares this data for the machine learning regression:
        water samples -> the training y_set, others become the prediction feature_set.
        all_water_samples = output from WaterSamples.sample_all_water().
        roads_clean = output from RoadClean.full_road_clean_map().
        prov_da = output from HouseSample.get_prov_da().
        aggr = aggregator for MC water samples.
        features = feature subset (usually self.features).
        """
        print('Prepping y train and X predict')
        # FIRST, group and prepare all_water_samples:
        # group water samples by aggr function,
        # then sum together emissions in the DAUID
        wm_samples_g = all_water_samples[0].groupby('uq').agg({'DAUID':'first',
                                                        'water_ghg':aggr}).groupby('DAUID').sum()
        sw_samples_g = all_water_samples[1].groupby('uq').agg({'DAUID':'first',
                                                            'water_ghg':aggr}).groupby('DAUID').sum()
        st_samples_g = all_water_samples[2].groupby('uq').agg({'DAUID':'first',
                                                            'water_ghg':aggr}).groupby('DAUID').sum()
        cb_samples_g = all_water_samples[3].groupby('uq').agg({'DAUID':'first',
                                                            'catchbasin_ghg':aggr}).groupby('DAUID').sum()

        # Combine all water emissions 
        y_ti = pd.concat([wm_samples_g, sw_samples_g, st_samples_g, cb_samples_g], axis=1).sum(axis=1)
        y_ti.name = 'all_water_ghg'
        # Divide by population of the DA
        y_t = self.population.join(y_ti)
        y_t['water_inf_ghg_pp'] = y_t['all_water_ghg']/y_t['pop_2021']
        # hyperbolic sine transform
        y_t = np.arcsinh(y_t['water_inf_ghg_pp'])

        # SECOND, create a prediction feature set from the given province.
        # Create road features from roads_clean
        road_features = roads_clean.groupby(['DAUID','ROADCLASS']).agg({'LENGTH_GEO':np.sum,
                                                                        'NBRLANES':np.median}).reset_index()
        road_features = road_features.join(pd.get_dummies(road_features['ROADCLASS'], drop_first=True))
        
        road_features_lanes = road_features.pivot(index='DAUID', columns='ROADCLASS', values=['NBRLANES']).fillna(0)
        road_features_length = road_features.pivot(index='DAUID', columns='ROADCLASS', values=['LENGTH_GEO',]).fillna(0)

        road_features_length = road_features_length.droplevel(0, axis=1)
        road_features_length.columns = [col_name.lower()+'_len' for col_name in road_features_length.columns]
        road_features_lanes = road_features_lanes.droplevel(0, axis=1)
        road_features_lanes.columns = [col_name.lower()+'_lanes' for col_name in road_features_lanes.columns]

        road_features = pd.concat([road_features_length, road_features_lanes], axis=1)
        
        # Append road features to the provincial DA dataframe.
        prov_da['DAUID'] = prov_da['DAUID'].astype(int)
        X_p_t = prov_da.join(road_features, on='DAUID')
        X_p_t = X_p_t.set_index('DAUID')

        # Inverse hyperbolic sine transform, return prediction feature data.
        X_p = np.arcsinh(X_p_t[self.features]).dropna()

        # return both y_train and X_predict
        return (y_t, X_p)




    def water_fit_predict(self, model, X_t, y_t, X_p):
        """
        Given DA properties and summed water MC results,
        predicts the water infrastructure ghg emissions pp
        of DAs in Ontario without data.
        model = sklearn model with optimized h-params,
        given through object self.model.
        X_train = MC output df known values, arcsin transformed.
        y_train = water inf ghg known values, arcsin transformed.
        X_p = MC output df unknown values, arcsin transformed.
        """
        # Fit
        model.fit(X_t, y_t)
        y_p = model.predict(X_p)

        return np.sinh(y_p)
    


    def water_regression_pipeline(self, all_water_samples, roads_clean, prov_da, aggr=np.median):
        """
        Runs the full water regression pipeline from uncleaned
        inputs to final outputs. Uses default object self values
        as inputs where required.
        all_water_samples = output from WaterSamples.sample_all_water().
        roads_clean = output from RoadClean.full_road_clean_map().
        prov_da = output from HouseSample.get_prov_da().
        aggr = aggregator for MC water samples.
        """
        print('\n [1] Cleaning regression data, filtering features, arsinh transforming. \n')
        prep_inputs = self.water_regression_prep(all_water_samples = all_water_samples, 
                                               roads_clean = roads_clean, 
                                               prov_da = prov_da, 
                                               aggr=aggr)
        
        # Merge training x and y into one dataframe to make sure all works.
        train_inputs = self.train.join(prep_inputs[0])


        print('\n [2] Fitting and predicting data for DA complement')
        y_pred = self.water_fit_predict(self.model,
                                       X_t=train_inputs[self.features], 
                                       y_t=train_inputs['water_inf_ghg_pp'], 
                                       X_p=prep_inputs[1])
        
        y_p = pd.Series(data=y_pred, index=prep_inputs[1].index)
        
        return y_p
    


    def water_regression_pipeline_da(self, all_water_samples, roads_clean, ont_da, aggr):
        """
        DEPRECIATED: TRANSFORM INTO ONTARIO SPECIFIC PROVINCE RETURNER
        Runs the full water regression pipeline from uncleaned
        inputs to final outputs. Uses default object self values
        as inputs where required. Returns results for all ontario
        DAs.
        """
        print('\n [1] Cleaning regression data, filtering features, arsinh transforming. \n')
        reg_inputs = self.water_regression_clean(all_water_samples = all_water_samples, 
                                            roads_clean = roads_clean, 
                                            ont_da = ont_da, 
                                            aggr = aggr, 
                                            features = self.features)

        print('\n [2] Fitting and predicting data for DA complement')
        y_com = self.water_fit_predict(self.model,
                                        reg_inputs[0], reg_inputs[1], reg_inputs[2])
        
        da_unknowns = reg_inputs[2]
        da_unknowns['water_inf_ghg_pp'] = y_com
        da_knowns = reg_inputs[0].join(np.sinh(reg_inputs[1]))
        da_all = ont_da[['DAUID']].join(pd.concat([da_unknowns, da_knowns]))
        da_all
        
        return da_all[['DAUID', 'water_inf_ghg_pp']].dropna()
