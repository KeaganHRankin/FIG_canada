"""
This object manipulates CMHC and StatsCAN data in order to
return projection housing starts under different growth scenarios
for a given province.
These projected starts are used as inputs to the FIG-Canada model.
"""
import numpy as np
import scipy as sp
import pandas as pd

class ProjectStarts:

    def __init__(self):
        self.path = 'C:/Users/Keagan Rankin/OneDrive - University of Toronto/Saxe - Rankin/Project 2. Housing Projections/FIG_Canada/data/proj/'
        # StatsCAN population projections.
        self.pop_proj = pd.read_csv(self.path+'canada population projections/17100057.csv')
        # CMHC business as usual and sigmoid affordability.
        self.bau_and_afford = pd.read_excel(self.path+'cmhc 2030 projections.xlsx', sheet_name='bau_plus_afford_logistic_raw', index_col=0)
        # 2021 census avg houshold size.
        self.household_size = pd.read_excel(self.path+'cmhc 2030 projections.xlsx', sheet_name='household_size', index_col=0)
        # House to housing stock ratio projections cmhc
        self.stock_pop_r = pd.read_excel(self.path+'cmhc 2030 projections.xlsx', sheet_name='stock_pop_ratio', index_col=0)


    def provincial_starts(self, province):
        """
        Returns projected starts for a given province
        from short term (CMHC) and long term (statsCAN population based)
        data under business-as-usual and high-growth scenarios.
        province (str) = province desired, underscores no capitals.
        exception is prince edward island is 'pei'.

        Territory construction is based solely on population projections
        """
        # store short-term starts.
        p_cols = [province+'_d', province+'_d_add']
        short_term = self.bau_and_afford[p_cols]
        
        # store long-term starts. Extend to 2050 assuming constant rate of change.
        name_map = {'newfoundland':'Newfoundland and Labrador', 
                    'pei':'Prince Edward Island',
                    'nova_scotia':'Nova Scotia', 
                    'new_brunswick':'New Brunswick', 
                    'quebec':'Quebec', 
                    'ontario':'Ontario', 
                    'manitoba':'Manitoba',
                    'saskatchewan':'Saskatchewan', 
                    'alberta':'Alberta', 
                    'british_columbia':'British Columbia',
                    }
        
        # Get relevant province data
        long_prov = self.pop_proj.loc[(self.pop_proj['GEO'] == name_map[province]) & 
                                      (self.pop_proj['Sex'] == 'Both sexes') &
                                      (self.pop_proj['Age group'] == 'All ages') &
                                      (self.pop_proj['REF_DATE'] >= 2022)
                                      ,:]
        
        # Convert to housing starts assuming constant household number avg for province.
        long_prov_houses = long_prov.copy()
        long_prov_houses['house_convert'] = (long_prov_houses['VALUE']*10**3)/self.household_size.loc[province].item()  
        long_prov_houses

        # store each scenario in a column
        long_term = {}
        long_term['year'] = np.arange(2031,2051)
        for i, h in enumerate(long_prov_houses['Projection scenario'].unique()):
            # Get unique scenario
            hr = long_prov_houses[long_prov_houses['Projection scenario'] == h]
            # Differenced yearly housing starts
            hrd = np.diff(hr['house_convert'])[8:]
            # Extend assuming rate of change stays constant next 7 years.
            # Negative new construction is assumed to be zero construction. 
            hrd = hrd[~np.isnan(hrd)]
            hrd = np.append(hrd, (np.arange(1,8) * (hrd[-1] - hrd[-2])) + hrd[-1])
            hrd[hrd < 0] = 0
            # Store
            long_term[h] = hrd

        # convert to dataframe
        long_term = pd.DataFrame.from_dict(long_term)

        # return in dataframe form; one for 2023-2030, one for 2031-2050
        # with all of the population scenarios.
        return(short_term, long_term)


    def territory_starts(self, territory):
        """
        Equivilent to the provincial starts method, but for
        territories and only uses StatsCAN data.
        """
        # store long-term starts. Extend to 2050 assuming constant rate of change.
        name_map = {
                    'yukon':'Yukon',
                    'nwt':'Northwest Territories', 
                    'nunavut':'Nunavut'
                    }
        
        # Get relevant province data
        long_teri = self.pop_proj.loc[(self.pop_proj['GEO'] == name_map[territory]) & 
                                (self.pop_proj['Sex'] == 'Both sexes') &
                                (self.pop_proj['Age group'] == 'All ages') &
                                (self.pop_proj['REF_DATE'] >= 2022)
                                    ,:]
        
        # Convert to housing starts assuming constant household number avg for Canada
        long_teri_houses = long_teri.copy()
        long_teri_houses['house_convert'] = (long_teri_houses['VALUE']*10**3)/2.5
        long_teri_houses

        # store each scenario in a column
        long_term = {}
        long_term['year'] = np.arange(2023,2051)
        for i, h in enumerate(long_teri_houses['Projection scenario'].unique()):
            # Get unique scenario
            hr = long_teri_houses[long_teri_houses['Projection scenario'] == h]
            # Differenced yearly housing starts from 2023
            hrd = np.diff(hr['house_convert'])
            # Extend assuming rate of change stays constant next 7 years.
            # Negative new construction is assumed to be zero construction. 
            hrd = hrd[~np.isnan(hrd)]
            hrd = np.append(hrd, (np.arange(1,8) * (hrd[-1] - hrd[-2])) + hrd[-1])
            hrd[hrd < 0] = 0
            # Store
            long_term[h] = hrd

        # convert to dataframe
        long_term = pd.DataFrame.from_dict(long_term)
        return long_term