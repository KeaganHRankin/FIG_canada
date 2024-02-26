"""
This class file contains a helper object with
various useful methods that are used during Monte
Carlo sampling and data analysis. They are general
methods that don't fit in other classes.
"""

### Imports
import numpy as np
import scipy as sp
import pandas as pd


class Helper:

    def __init__(self):
        """init"""



    def mc_results_aggregator_roads(self, road_samples, agg_func):
        """
        Aggregates road samples from x Monte Carlo iters.
        agg_func is the aggregator,
        ORN_ID and DAUID grouped by first because they
        are preserved through unique identifier uq in the above functions
        (should be the same in all uq values).
        """
        road_samples_g = road_samples.groupby('uq').agg({'DAUID':'first',
                                                        'OGF_ID':'first',
                                                        'ghg':agg_func
                                                        })
        return road_samples_g



    def mc_results_aggregator_houses(self, house_samples, ont_da, agg_func):
        """
        Aggregates hosue samples from x Monte Carlo iters based on agg_func
        house_samples = output of house samples.
        ont_da = ontario da data.
        """
        struct_dt = ['single_detached', 'mid_high_rise', 'semi_detached', 'rowhouse',
                    'lowrise_apartment', 'flat_duplex']
        ont_da_analysis = ont_da

        for g, s in zip(house_samples, struct_dt):
            # Change name
            ghg_agg = g.groupby(g.index).agg(agg_func)
            ghg_agg.name = 'ghg_{}'.format(s)

            # Append
            ont_house_g = ont_da_analysis.join(ghg_agg, on='DAUID')

        # Aggregate columns
        ont_house_g['house_ghg_pp'] = np.vstack([g.groupby(g.index).agg(agg_func) for g in house_samples]).sum(axis=0)
        ont_house_g['missing_middle_ghg_pp'] = np.vstack([g.groupby(g.index).agg(agg_func) for g in house_samples[2:]]).sum(axis=0)

        return ont_house_g



    def geo_class_add(self, da_df, id_col):
        """
        Adds census geographical classification labels to the a df based on da id.
        da_df = dataframe to add to.
        id_col = column with da id in it.
        """

        toronto = ['20']
        counties = ['01','02','07','09','10','11','12','14','15','22','23','31','32',
                    '34','37','38','39','40','41','42','43','46','47']
        other_cities = ['06','18','19','21','24','26','25','30']
        north_districts = ['48','49','51','52','54','56','57','58','59','60']

        conditions = [(da_df[id_col].astype('string').str[2:4].isin(toronto)),
                    (da_df[id_col].astype('string').str[2:4].isin(counties)),
                    (da_df[id_col].astype('string').str[2:4].isin(other_cities)),
                    (da_df[id_col].astype('string').str[2:4].isin(north_districts)),
                    ]
            
        outputs = ['toronto','rural_counties','other_cities','northern_districts']

        col_map = np.select(conditions, outputs, 'none')

        da_df['geo_class'] = col_map

        return da_df


    def drop_outliers_iqr(self, df, c, ql=0.25, qu=0.75, f=1.5):
        """Drop datapoints in dataframe df outside
        quantile/interquartile range based on col c,
        passed quantiles ql and qu, scale factor f"""
        qlv = df[c].quantile(ql)
        quv = df[c].quantile(qu)
        iqr = quv - qlv

        dfr = df[(df[c] > (qlv - f*iqr))] 
        dfr = dfr[((dfr[c] < (quv + f*iqr)))]

        print('dropped: ', df.shape[0] - dfr.shape[0])
        return dfr