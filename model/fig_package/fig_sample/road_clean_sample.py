"""
This class file contains functions that process road data and 
sample iterations of the road data using methods from MonteCarlo()
class instances. Functions are developed in road_sampling_iter.ipynb
"""

### Imports
import numpy as np
import scipy as sp
import pandas as pd
# VVV add dot to this when it is in its own package folder VVV
from .simulation import MonteCarlo


class RoadClean:
    """
    Object stores methods for importing and processing arcgis road data
    into a format that can be taken in by the RoadSample object for
    Monte Carlo sampling/error propogation.
    """

    def __init__(self, master='C:/Users/rankin6/'):
        self.path = master+'OneDrive - University of Toronto/Saxe - Rankin/Project 2. Housing Projections/FIG_Canada/data/roads/da_merged_roads/'
        self.province_name = 'ontario'


    def provincial_road_clean(self, province_name):
        """
        Cleans road data for a given province, e.g. removing duplicates
        and halving the length of roads straddling DAs. Returns cleaned
        PAVED road df for sampling.
        """
        # map passed province to file
        road_file_name_map = {"ontario":"ont_da_road_join.csv",
                            "alberta":"ab_da_road_join.csv",
                            "british_columbia":"bc_da_road_join.csv",
                            "manitoba":"mb_da_road_join.csv",
                            "new_brunswick":"nb_da_road_join.csv",
                            "newfoundland":"nl_da_road_join.csv",
                            "nova_scotia":"ns_da_road_join.csv",
                            "nunavut":"nu_da_road_join.csv",
                            "nwt":"nwt_da_road_join.csv",
                            "pei":"pei_da_road_join.csv",
                            "quebec":"qc_da_road_join.csv",
                            "saskatchewan":"sask_da_road_join.csv",
                            "yukon":"yu_da_road_join.csv"
                            }
        
        # Store raw provincial data:
        province = pd.read_csv(self.path+road_file_name_map[province_name])

        # Preprocess:
        # remove unjoined
        prov_t = province.copy()
        prov_t = prov_t.loc[prov_t['JOIN_FID'] != -1, :]
        # remove unwanted road classes
        prov_t = prov_t.loc[~prov_t['ROADCLASS'].isin(['Winter', 'Unknown', 'Resource / Recreation', np.nan])]
        # remove unpaved roads
        prov_t = prov_t.loc[prov_t['PAVSTATUS']=='Paved']
        # relevant columns
        prov_t = prov_t[['Join_Count','JOIN_FID','TARGET_FID','DAUID','NID','PAVSURF','ROADCLASS','NBRLANES','LENGTH_GEO']]

        # Process straddling roads:
        # split
        straddling = prov_t.copy()
        straddling = straddling[straddling.duplicated('JOIN_FID', keep=False) == True].sort_values('JOIN_FID')
        non_straddling = prov_t.copy()
        non_straddling = non_straddling[~non_straddling.duplicated('JOIN_FID', keep=False) == True].sort_values('JOIN_FID')

        # drop duplicated DAUID, NID, AND LENGTH (same part of road in the same DA is a duplicate)
        straddling = straddling.drop_duplicates(['DAUID','NID','LENGTH_GEO'])

        # check some stuff -> did we drop half the values? How are lengths retained?
        #a = straddling['LENGTH_GEO'].sum()
        #b = straddling.drop_duplicates(['LENGTH_GEO'])['LENGTH_GEO'].sum()
        #print('if = 2, there are 2 unique entries for each straddling road segment: ', a/b)
        #print('if = 2, dropped exactly half of dataframe: ', straddling_merge_test['Join_Count'].sum()/straddling_merge_test_dropped['Join_Count'].sum())

        # half length and rejoin/return
        straddling['LENGTH_GEO'] = straddling['LENGTH_GEO']/2

        return pd.concat([straddling, non_straddling])
    

    def grip_mapper(self, prov_roads_cleaned, user_type_map=None):
        """
        Maps provincial road types to the GRIP dataset typologies used
        in Rousseau. Optional user control of types for sample tuning.

        user_type_map = optional input where road types can be manually
        mapped by the user (for tuning road archetype dimensions).
        """
        # Map to GRIP road types. -> Tune to Disag Data from qualitative,
        # literature-based mapping.
        road_type_map = {'Local / Street': 'Local', 
                    'Arterial': 'Secondary', 
                    'Expressway / Highway': 'Highway', 
                    'Collector': 'Tertiary',
                    'Ramp': 'Secondary', 
                    'Alleyway / Laneway': 'Local', 
                    'Alleyway / Lane': 'Local',
                    'Freeway': 'Highway', 
                    'Local / Strata': 'Local',
                    'Local / Unknown': 'Local', 
                    'Service': 'Secondary', 
                    'Service Lane': 'Secondary',
                    'Rapid Transit': 'Secondary',
                    }
        
        # User control over road type mapping given here.
        if user_type_map is not None:
            print('option: editing road_type_map w/ user_type_map')
            for key, value in user_type_map.items():
                road_type_map[key] = user_type_map[key]
        
        # rename to manually assigned GRIP names.
        prov_roads_cleaned = prov_roads_cleaned.replace({'ROADCLASS': road_type_map})

        return prov_roads_cleaned
    

    def full_road_clean_map(self, user_type_map=None):
        """Full pipeline for road cleaning."""

        print('Cleaning road data...')
        # Clean ARCGIS Data
        p = self.provincial_road_clean(self.province_name)
        # Grip Map
        rcm = self.grip_mapper(prov_roads_cleaned=p,
                               user_type_map=user_type_map)

        print('\nImport-clean complete. Returning.')
        return rcm
    


class RoadSample:
    """
    Object stores methods for monte carlo sampling of roads in provincial DAs.
    uses MonteCarlo class from simulation.py to complete distribution initialize 
    and return iterations. 
    """

    def __init__(self, master='C:/Users/rankin6/'):
        """
        Sample options are initialized.
        pw = path to width csv.
        ptf = path to flexible thickness csv.
        ptr = path to rigid thickness csv.
        """
        self.pw = master+'OneDrive - University of Toronto/Saxe - Rankin/Project 2. Housing Projections/Code/brad road data and code/RoadArch-Python/archetypes/archetypes/lane_widths_adjustments.csv'
        self.ptf = master+'OneDrive - University of Toronto/Saxe - Rankin/Project 2. Housing Projections/Code/brad road data and code/RoadArch-Python/archetypes/archetypes/archetypes_agg.csv'
        self.ptr = master+'OneDrive - University of Toronto/Saxe - Rankin/Project 2. Housing Projections/Code/brad road data and code/RoadArch-Python/archetypes/archetypes/archetypes_rigid_agg.csv'



     ## Prep functions
    def road_sample_import_dists(self, path_width, path_thick_f, path_thick_r):
        """
        Import/return other data needed for Monte carlo sampling.
        path_width = full path to lane_width_adjustments.csv from Rousseau.
        path_thick_f = full path to archetypes_agg (flexible pavement t in mm).
        path_thick_r = full path to rigid_archetypes_agg (rigid pavement t in mm).
        returns -> width, two thickness, and ghg factor dfs with params fitted to a dist.
        """
        # THICKNESSES
        arch_flexible = pd.read_csv(path_thick_f)
        arch_rigid = pd.read_csv(path_thick_r)

        name_map = {'rigid_asphalt_mm_median': 'asphalt_mm_median',
                    'rigid_asphalt_mm_low': 'asphalt_mm_low',
                    'rigid_asphalt_mm_high': 'asphalt_mm_high',
                    'rigid_concrete_mm_median': 'concrete_mm_median',
                    'rigid_concrete_mm_low': 'concrete_mm_low',
                    'rigid_concrete_mm_high': 'concrete_mm_high',
                    'rigid_granular_mm_median': 'granular_mm_median',
                    'rigid_granular_mm_low': 'granular_mm_low',
                    'rigid_granular_mm_high': 'granular_mm_high'}

        arch_flexible = arch_flexible[arch_flexible['country'] == 'CAN'].set_index('GRIP_type_name')
        arch_flexible[['concrete_mm_median', 'concrete_mm_low', 'concrete_mm_high']] = 0
        arch_rigid = arch_rigid[arch_rigid['country'] == 'CAN'].rename(columns=name_map).set_index('GRIP_type_name')

        # Fit uniform.
        upp = ['asphalt_mm_high','concrete_mm_high','granular_mm_median']
        low = ['asphalt_mm_low','concrete_mm_low','granular_mm_low']
        nam = ['asphalt','concrete','granular']
        for u, l, m in zip(upp, low, nam):
            arch_flexible['uniform_'+m] = [sp.stats.uniform.fit([arch_flexible.iloc[x][l],
                                        arch_flexible.iloc[x][u]]) for x in range(0,5)]

            arch_rigid['uniform_'+m] = [sp.stats.uniform.fit([arch_rigid.iloc[x][l],
                                        arch_rigid.iloc[x][u]]) for x in range(0,5)] 
            
        
        # WIDTH
        lanewidths = pd.read_csv(path_width)
        lanewidths = lanewidths[lanewidths['Country'] == 'CAN']

        l = [sp.stats.uniform.fit([lanewidths.iloc[x]['lower_lanewidth'],
                                lanewidths.iloc[x]['upper_lanewidth']]) for x in range(0,5)]

        lanewidths['uniform'] = l
        lanewidths = lanewidths.set_index('GRIP road type')


        # GHG
        # material GHG factor (kgCO2eq/kg) distribution params.
        ghg_f_params = {'asphalt': sp.stats.uniform.fit([0.07, 0.098]), # SI 0.07 - 0.098 Uniform -> FIX TO ROUSSEAU MONTE CARLO (SEE HER CODE)
                    'concrete': (0.060967, 0.10185,
                                3.00022503240956, 
                                2.99977496759044), # format a, c, alpha, beta. Aldrick DB 20MPa Pert 0.060967, 0.0814108, 0.10185
                    'granular': (0.00155, 0.0050905,
                                5.00045197740113, 
                                0.9995480225988698), # SI 3.1.2 or Aldrick DB Pert 0.00155, 0.0050904, 0.00509
                    }
        
        return (lanewidths, arch_flexible, arch_rigid, ghg_f_params)
    


    # Same as above function but allows for choice of ghg params.
    def road_sample_import_dists_choose(self, path_width, path_thick_f, path_thick_r, asphalt_mm, concrete_mmm, granular_mmm):
        """
        Import/return other data needed for Monte carlo sampling.
        path_width = full path to lane_width_adjustments.csv from Rousseau.
        path_thick_f = full path to archetypes_agg (flexible pavement t in mm).
        path_thick_r = full path to rigid_archetypes_agg (rigid pavement t in mm).
        returns -> width, two thickness, and ghg factor dfs with params fitted to a dist.
        asphalt_mm = np.array([asphalt_min, asphalt_max]) emission factors
        concrete_mmm = np.array([concrete_min, concrete_most_likely, concrete_max]) emission factors
        granular_mmm = np.array([granular_min, granular_most_likely, granular_max]) emission factors
        """
        # THICKNESSES
        arch_flexible = pd.read_csv(path_thick_f)
        arch_rigid = pd.read_csv(path_thick_r)

        name_map = {'rigid_asphalt_mm_median': 'asphalt_mm_median',
                    'rigid_asphalt_mm_low': 'asphalt_mm_low',
                    'rigid_asphalt_mm_high': 'asphalt_mm_high',
                    'rigid_concrete_mm_median': 'concrete_mm_median',
                    'rigid_concrete_mm_low': 'concrete_mm_low',
                    'rigid_concrete_mm_high': 'concrete_mm_high',
                    'rigid_granular_mm_median': 'granular_mm_median',
                    'rigid_granular_mm_low': 'granular_mm_low',
                    'rigid_granular_mm_high': 'granular_mm_high'}

        arch_flexible = arch_flexible[arch_flexible['country'] == 'CAN'].set_index('GRIP_type_name')
        arch_flexible[['concrete_mm_median', 'concrete_mm_low', 'concrete_mm_high']] = 0
        arch_rigid = arch_rigid[arch_rigid['country'] == 'CAN'].rename(columns=name_map).set_index('GRIP_type_name')

        # Fit uniform.
        upp = ['asphalt_mm_high','concrete_mm_high','granular_mm_median']
        low = ['asphalt_mm_low','concrete_mm_low','granular_mm_low']
        nam = ['asphalt','concrete','granular']
        for u, l, m in zip(upp, low, nam):
            arch_flexible['uniform_'+m] = [sp.stats.uniform.fit([arch_flexible.iloc[x][l],
                                        arch_flexible.iloc[x][u]]) for x in range(0,5)]

            arch_rigid['uniform_'+m] = [sp.stats.uniform.fit([arch_rigid.iloc[x][l],
                                        arch_rigid.iloc[x][u]]) for x in range(0,5)] 
            
        
        # WIDTH
        lanewidths = pd.read_csv(path_width)
        lanewidths = lanewidths[lanewidths['Country'] == 'CAN']

        l = [sp.stats.uniform.fit([lanewidths.iloc[x]['lower_lanewidth'],
                                lanewidths.iloc[x]['upper_lanewidth']]) for x in range(0,5)]

        lanewidths['uniform'] = l
        lanewidths = lanewidths.set_index('GRIP road type')


        # GHG
        # material GHG factor (kgCO2eq/kg) distribution params.
        s = MonteCarlo()
        ghg_f_params = {'asphalt': sp.stats.uniform.fit(asphalt_mm), # SI 0.07 - 0.098 Uniform -> FIX TO ROUSSEAU MONTE CARLO (SEE HER CODE)
                    'concrete': (concrete_mmm[0], 
                                concrete_mmm[2],
                                s.pert_fit(concrete_mmm[0], concrete_mmm[1], concrete_mmm[2])[0], 
                                s.pert_fit(concrete_mmm[0], concrete_mmm[1], concrete_mmm[2])[1]), # format a, c, alpha, beta. Aldrick DB 20MPa Pert 0.060967, 0.0814108, 0.10185
                    'granular': (granular_mmm[0], 
                                granular_mmm[2],
                                s.pert_fit(granular_mmm[0], granular_mmm[1], granular_mmm[2])[0], 
                                s.pert_fit(granular_mmm[0], granular_mmm[1], granular_mmm[2])[1]), # SI 3.1.2 or Aldrick DB Pert 0.00155, 0.0050904, 0.00509
                    }
        
        return (lanewidths, arch_flexible, arch_rigid, ghg_f_params)



    def mc_iters_flex_init(self, x, road_df_c):
        """
        Given x number of monte carlo iterations, duplicates, appends, and returns
        expanded input vectors x times so that vectorize simulation can be performed.
        also samples and adds flexible vs. non flexible column based on binomial draws
        given 100% of local roads in Ont are flexible and 68% of remaining are flexible.
        road_df_c = cleaned road data frame
        """
        # Add unique identifier
        unique = np.arange(road_df_c.shape[0])
        road_df_c['uq'] = unique
        r_df = road_df_c[['DAUID', 'NID', 'uq', 'LENGTH_GEO', 'ROADCLASS', 'NBRLANES', 'PAVSURF']]
        col = ['DAUID', 'NID', 'uq', 'LENGTH_GEO', 'ROADCLASS', 'NBRLANES', 'PAVSURF']

        # Duplicate rows using faster np implementation.
        rs = []
        for c in col:
            r0 = pd.DataFrame(np.repeat(r_df[c].to_numpy(), x, axis=0), columns=[c])
            rs.append(r0)

        r_d = pd.concat(rs, axis=1)
        # Add iteration counter for later aggregation
        
        # sample flexible. Want to randomly assign non-local roads as either flexible or rigid,
        # keep ~68% ratio. Draw from binomial distribution. Fill local = 1 (flexible).
        flex = np.random.binomial(n=1, p=0.68, size=r_d.shape[0])
        r_d['flexible'] = flex
        r_d['flexible'] = np.where(r_d['ROADCLASS'] == 'Local', 1, r_d['flexible'])
        # WHERE KNOWN: fill using PAVESURF column as mapper
        r_d['flexible'] = np.where(r_d['PAVSURF'] == 'Flexible', 1, r_d['flexible'])
        r_d['flexible'] = np.where(r_d['PAVSURF'] == 'Rigid', 0, r_d['flexible'])

        return r_d[['DAUID', 'NID', 'uq', 'LENGTH_GEO', 'ROADCLASS', 'NBRLANES', 'flexible']]
    


    ## Sampling functions
    def sample_road_generic(self, road_df, cl, flex, w_params, tf_params, tr_params, ghg_params):
        """
        Monte Carlo sampling to propogate error for
        bottom-up quantification of embodied GHG in
        Ontario roads, split by DA, based on Rousseau et al.
        archetype approach. Inputs from cleaned road data that has been
        split by class and pavement flexibility.
        road_df = iter and flex prepped (like) road_df.
        cl = GRIP road class string.
        flex = 0 (flexible) or rigid (1). df input.
        tf_params = df with flexible pavement params.
        tr_params = df with rigid pavement params.
        w_params = df with lane width params.
        ghg_params = dict with ghg factor params.
        """
        # draw thickness (mm), width (m), ghg factor vectors (kgco2eq/kg), 
        # then multiply elementwise for performance.
        # thickness
        def elementwise_thickness(cl, f):
            if f == 1:
                asph = sp.stats.uniform.rvs(tf_params.loc[(cl, 'uniform_asphalt')][0], tf_params.loc[(cl, 'uniform_asphalt')][1], size=road_df.shape[0])
                conc = sp.stats.uniform.rvs(tf_params.loc[(cl, 'uniform_concrete')][0], tf_params.loc[(cl, 'uniform_concrete')][1], size=road_df.shape[0])
                gran = sp.stats.uniform.rvs(tf_params.loc[(cl, 'uniform_granular')][0], tf_params.loc[(cl, 'uniform_granular')][1], size=road_df.shape[0])
            elif f == 0:
                asph = sp.stats.uniform.rvs(tr_params.loc[(cl, 'uniform_asphalt')][0], tr_params.loc[(cl, 'uniform_asphalt')][1], size=road_df.shape[0])
                conc = sp.stats.uniform.rvs(tr_params.loc[(cl, 'uniform_concrete')][0], tr_params.loc[(cl, 'uniform_concrete')][1], size=road_df.shape[0])
                gran = sp.stats.uniform.rvs(tr_params.loc[(cl, 'uniform_granular')][0], tr_params.loc[(cl, 'uniform_granular')][1], size=road_df.shape[0])

            return (asph, conc, gran)
        
        thick = elementwise_thickness(cl, flex)
        asph_t = thick[0]
        conc_t = thick[1]
        gran_t = thick[2]

        # width
        width = sp.stats.uniform.rvs(w_params.loc[(cl, 'uniform')][0], w_params.loc[(cl, 'uniform')][1], size=road_df.shape[0])

        # volume (t*w*lanes*l), mass (*ro), and factor
        s = MonteCarlo()    
        asphalt_m = asph_t * 0.001 * width * road_df['NBRLANES'] * road_df['LENGTH_GEO'] * (2.3*10**3)
        concrete_m = conc_t * 0.001 * width * road_df['NBRLANES'] * road_df['LENGTH_GEO'] * (2.3*10**3)
        granular_m = gran_t * 0.001 * width * road_df['NBRLANES'] * road_df['LENGTH_GEO'] * (2.4*10**3)

        asphalt_gf = sp.stats.uniform.rvs(ghg_params['asphalt'][0], ghg_params['asphalt'][1], size=road_df.shape[0])
        concrete_gf = s.pert_sample(ghg_params['concrete'][0],
                                    ghg_params['concrete'][1],
                                    ghg_params['concrete'][2],
                                    ghg_params['concrete'][3],
                                    size=road_df.shape[0])
        granular_gf = s.pert_sample(ghg_params['granular'][0],
                                    ghg_params['granular'][1],
                                    ghg_params['granular'][2],
                                    ghg_params['granular'][3],
                                    size=road_df.shape[0])

        # results
        road_ghg = (asphalt_m * asphalt_gf) + (concrete_m * concrete_gf) + (granular_m * granular_gf)
        road_ghg.name = 'ghg'

        return pd.concat([road_df['DAUID'], road_df['NID'], road_df['uq'], road_ghg], axis=1)



    def roads_sample_run_efficient(self, road_da, road_dists):
        """
        apply sample_road_generic w/ optimal runtime
        by splitting by road class and flexible vs. not
        so that all vectorizations can be optimized 
        road_df = iter and flex prepped road_df.
        road_dists = tuple output of road_sample_import_dists.
        """

        print('Beginning sampling. Benchmarked at ~50 iters/min. For higher iters, try loop writing into a file to avoid memory issues.')
        # Loop the generic function over combinations of classes and pavement flexibility
        classes = ['Highway', 'Local', 'Primary', 'Secondary', 'Tertiary']
        flexible = [0,1]

        roads_df_split_list = []
        for c in classes:
            for f in flexible:
                road_df_split = road_da[(road_da['ROADCLASS'] == c) & (road_da['flexible'] == f)]

                # Local roads are all flexible - don't append if dataframe is empty
                if road_df_split.empty == False:
                    roads_df_split_list.append(road_df_split)


        # apply generic function to each.
        split_samples = []
        for split_df in roads_df_split_list:
            print(split_df['ROADCLASS'].iloc[0])
            print('iter:', split_df['ROADCLASS'].iloc[0], split_df['flexible'].iloc[0])      
            samp = self.sample_road_generic(road_df = split_df, 
                                            cl = split_df['ROADCLASS'].iloc[0], 
                                            flex = split_df['flexible'].iloc[0],
                                            w_params=road_dists[0],
                                            tf_params=road_dists[1], 
                                            tr_params=road_dists[2],  
                                            ghg_params=road_dists[3])
            
            split_samples.append(samp)

        # concat split and return all results from x Monte Carlo iterations.
        return pd.concat(split_samples)



    ## Aggregate functions for prep and sampling
    def road_sample_prep(self, path_width, path_thick_f, path_thick_r, roads_cleaned, iters):
        """
        Run functions to prep road data for sampling,
        return repeated matrix for sampling.
        path_x = path to distribution info csv files
        roads_cleaned = cleaned roads file run through RoadProcess instance.
        iters = number of Monte Carlo iterations.
        """
        print('1/2 Importing distributions.')
        road_dists = self.road_sample_import_dists(path_width, path_thick_f, path_thick_r)
        print('\n2/2 Prepping iterations and sampling flexible binomial.')
        roads_clean_prepped = self.mc_iters_flex_init(iters, roads_cleaned)

        return road_dists, roads_clean_prepped
    

    def road_sample_prep_and_run(self, roads_cleaned, iters):
        """
        Run functions to prep road data for sampling,
        return repeated matrix for sampling.
        path_x = path to distribution info csv files
        roads_cleaned = cleaned roads file run through RoadProcess instance.
        iters = number of Monte Carlo iterations.
        """

        print('1/3 Importing distributions.')
        road_dists = self.road_sample_import_dists(self.pw, self.ptf, self.ptr)
        print('\n2/3 Prepping iterations and sampling flexible binomial.')
        roads_clean_prepped = self.mc_iters_flex_init(iters, roads_cleaned)
        print('\n3/3 Performing Monte Carlo sampling.')
        road_samples = self.roads_sample_run_efficient(roads_clean_prepped, road_dists)

        print('\n Sampling complete. Returning road samples.')
        return road_samples



    # This aggregate function allows for selection of mat emission factors instead of defaults.
    def road_sample_prep_and_run_choose(self, roads_cleaned, iters, asphalt_mm, concrete_mmm, granular_mmm):
        """
        Run functions to prep road data for sampling,
        return repeated matrix for sampling.
        path_x = path to distribution info csv files
        roads_cleaned = cleaned roads file run through RoadProcess instance.
        iters = number of Monte Carlo iterations.
        asphalt_mm = np.array([asphalt_min, asphalt_max]) emission factors
        concrete_mmm = np.array([concrete_min, concrete_most_likely, concrete_max]) emission factors
        granular_mmm = np.array([granular_min, granular_most_likely, granular_max]) emission factors
        """

        print('1/3 Importing distributions.')
        road_dists = self.road_sample_import_dists_choose(self.pw, self.ptf, self.ptr, asphalt_mm, concrete_mmm, granular_mmm)
        print('\n2/3 Prepping iterations and sampling flexible binomial.')
        roads_clean_prepped = self.mc_iters_flex_init(iters, roads_cleaned)
        print('\n3/3 Performing Monte Carlo sampling.')
        road_samples = self.roads_sample_run_efficient(roads_clean_prepped, road_dists)

        print('\n Sampling complete. Returning road samples.')
        return road_samples