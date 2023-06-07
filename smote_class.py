import numpy as np
import pandas as pd

class SMOTENC:
    def __init__(self, nominal_columns, numeric_columns, k=5):
        self.nominal_columns = nominal_columns
        self.numeric_columns = numeric_columns
        self.k = k
        
    
    def nearest_neighbors(self, row, df):
        def distance(row1, row2):
            distance = 0
            for col in self.nominal_columns:
                if row1[col] != row2[col]:
                    distance += 1
            for col in self.numeric_columns:
                distance += (row1[col] - row2[col])**2
            return distance**0.5

        distances = []
        for i in range(len(df)):
            r = df.iloc[i]
            
            d = distance(row, r)
            if d != 0:
                distances.append((d, i))

        distances.sort()
        nearest = [i for d, i in distances[:self.k]]
        return nearest
    
    def smotenc(self, X, y):
        minority_class = y[y == 'Yes']
        majority_class = y[y == 'No']
        minority_samples = X[y == 'Yes']
        
        n_synthetic_samples = len(majority_class) - len(minority_class)
        synthetic_samples = []
        
        

        for i in range(len(minority_samples)):
            nn = self.nearest_neighbors(minority_samples.iloc[i], minority_samples)
            samples_per1 = int(n_synthetic_samples / len(minority_samples))

            for j in range(samples_per1):
                nn_idx = int(np.random.choice(a=nn))
                nn_sample = minority_samples.iloc[nn_idx]
                diff = nn_sample[self.numeric_columns] - minority_samples.iloc[i][self.numeric_columns]
                synthetic_sample_numeric = minority_samples.iloc[i][self.numeric_columns] + np.random.rand() * diff
                synthetic_sample_nominal = nn_sample[self.nominal_columns]
                synthetic_sample = pd.concat([synthetic_sample_numeric, synthetic_sample_nominal], axis=0)
                synthetic_samples.append(synthetic_sample)

        X_resampled = pd.concat([minority_samples, pd.DataFrame(synthetic_samples, columns=X.columns)], axis=0)
        

        return X_resampled