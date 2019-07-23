# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

'''
Prepare a prediction model for profit of 50_startups data.
Do transformations for getting better predictions of profit and
make a table containing R^2 value for each prepared model.
'''

# loading the data
Startups = pd.read_csv("C:/Users/cawasthi/Desktop/Data Science/R ML Code/liner_regression/50_Startups.csv")
Startups.columns
Startups['State'] = Startups['State'].astype("category").cat.codes


#to get the correlation 
Startups.corr()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(Startups)

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
# Preparing model                  
ml1 = smf.ols("Startups['Profit']~Startups['Administration']+Startups['Marketing Spend']+Startups['R&D Spend']",data=Startups).fit() 
ml1.summary()

'''
<class 'statsmodels.iolib.summary.Summary'>
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:     Startups['Profit']   R-squared:                       0.951
Model:                            OLS   Adj. R-squared:                  0.948
Method:                 Least Squares   F-statistic:                     296.0
Date:                Thu, 11 Jul 2019   Prob (F-statistic):           4.53e-30
Time:                        08:56:34   Log-Likelihood:                -525.39
No. Observations:                  50   AIC:                             1059.
Df Residuals:                      46   BIC:                             1066.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
===============================================================================================
                                  coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------------------
Intercept                    5.012e+04   6572.353      7.626      0.000    3.69e+04    6.34e+04
Startups['Administration']     -0.0268      0.051     -0.526      0.602      -0.130       0.076
Startups['Marketing Spend']     0.0272      0.016      1.655      0.105      -0.006       0.060
Startups['R&D Spend']           0.8057      0.045     17.846      0.000       0.715       0.897
==============================================================================
Omnibus:                       14.838   Durbin-Watson:                   1.282
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.442
Skew:                          -0.949   Prob(JB):                     2.21e-05
Kurtosis:                       5.586   Cond. No.                     1.40e+06
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.4e+06. This might indicate that there are
strong multicollinearity or other numerical problems.
'''

#intercept are in significant
np.mean(ml1.resid) #0
np.sqrt(sum(ml1.resid**2)/49) #8945.24

#to check for influential record
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)

#removing the influential records
Startups = Startups.drop(Startups.index[45],axis=0)

# Preparing model  
x = smf.ols("Startups['Administration']~Startups['Profit']+Startups['Marketing Spend']+Startups['R&D Spend']",data=Startups).fit().rsquared               
vif_adm =1/(1-x)
vif_adm

y = smf.ols("Startups['Marketing Spend']~Startups['Profit']+Startups['Administration']+Startups['R&D Spend']",data=Startups).fit().rsquared
vif_Mark_Spend = 1/(1-y)
vif_Mark_Spend

#Startups['Administration'] has more insignificant value for intercept
#vif for vif_Mark_Spend is more

# Added varible plot 
sm.graphics.plot_partregress_grid(ml1)
#AVP of Administration is insignificant 


#new model 
new_ml1 = smf.ols("Startups['Profit']~Startups['Marketing Spend']+Startups['R&D Spend']",data=Startups).fit() 
new_ml1.summary()

'''
                            OLS Regression Results                            
==============================================================================
Dep. Variable:     Startups['Profit']   R-squared:                       0.953
Model:                            OLS   Adj. R-squared:                  0.951
Method:                 Least Squares   F-statistic:                     467.7
Date:                Thu, 11 Jul 2019   Prob (F-statistic):           2.69e-31
Time:                        09:00:17   Log-Likelihood:                -513.45
No. Observations:                  49   AIC:                             1033.
Df Residuals:                      46   BIC:                             1039.
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
===============================================================================================
                                  coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------------------
Intercept                    4.538e+04   2724.707     16.654      0.000    3.99e+04    5.09e+04
Startups['Marketing Spend']     0.0336      0.015      2.218      0.032       0.003       0.064
Startups['R&D Spend']           0.8026      0.040     19.976      0.000       0.722       0.884
==============================================================================
Omnibus:                       14.947   Durbin-Watson:                   1.178
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               20.493
Skew:                          -1.000   Prob(JB):                     3.55e-05
Kurtosis:                       5.457   Cond. No.                     5.56e+05
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 5.56e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
'''

#influential plot
sm.graphics.influence_plot(new_ml1)
Startups = Startups.drop(Startups.index[[45,49]],axis=0)

#new model after removing influential records
new_ml1 = smf.ols("Startups['Profit']~Startups['Marketing Spend']+Startups['R&D Spend']",data=Startups).fit() 
new_ml1.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:     Startups['Profit']   R-squared:                       0.953
Model:                            OLS   Adj. R-squared:                  0.951
Method:                 Least Squares   F-statistic:                     467.7
Date:                Thu, 11 Jul 2019   Prob (F-statistic):           2.69e-31
Time:                        09:01:15   Log-Likelihood:                -513.45
No. Observations:                  49   AIC:                             1033.
Df Residuals:                      46   BIC:                             1039.
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
===============================================================================================
                                  coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------------------
Intercept                    4.538e+04   2724.707     16.654      0.000    3.99e+04    5.09e+04
Startups['Marketing Spend']     0.0336      0.015      2.218      0.032       0.003       0.064
Startups['R&D Spend']           0.8026      0.040     19.976      0.000       0.722       0.884
==============================================================================
Omnibus:                       14.947   Durbin-Watson:                   1.178
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               20.493
Skew:                          -1.000   Prob(JB):                     3.55e-05
Kurtosis:                       5.457   Cond. No.                     5.56e+05
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 5.56e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

#intercept are in significant
np.mean(new_ml1.resid) #0
np.sqrt(sum(new_ml1.resid**2)/49) #8601.16

Profit_pred = new_ml1.predict(Startups)

#AVP for final model
sm.graphics.plot_partregress_grid(new_ml1)


######  Linearity #########
# Observed values VS Fitted values
plt.scatter(Startups['Profit'],Profit_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

# Residuals VS Fitted Values 
plt.scatter(Startups['Profit'],new_ml1.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


########    Normality plot for residuals ######
# histogram
plt.hist(new_ml1.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(new_ml1.resid_pearson, dist="norm", plot=pylab)


############ Homoscedasticity #######

# Residuals VS Fitted Values 
plt.scatter(Profit_pred,new_ml1.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
Startups_train,Startups_test  = train_test_split(Startups,test_size = 0.2) # 20% size


#model for traning
Startups_train_model = smf.ols("Startups_train['Profit'] ~ Startups_train['Marketing Spend']+Startups_train['R&D Spend']",data=Startups_train).fit() 
Startups_train_model.summary()
"""
                               OLS Regression Results                               
====================================================================================
Dep. Variable:     Startups_train['Profit']   R-squared:                       0.953
Model:                                  OLS   Adj. R-squared:                  0.950
Method:                       Least Squares   F-statistic:                     362.3
Date:                      Thu, 11 Jul 2019   Prob (F-statistic):           1.42e-24
Time:                              09:11:31   Log-Likelihood:                -410.19
No. Observations:                        39   AIC:                             826.4
Df Residuals:                            36   BIC:                             831.4
Df Model:                                 2                                         
Covariance Type:                  nonrobust                                         
=====================================================================================================
                                        coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------------------------
Intercept                          4.423e+04   3379.868     13.085      0.000    3.74e+04    5.11e+04
Startups_train['Marketing Spend']     0.0404      0.019      2.181      0.036       0.003       0.078
Startups_train['R&D Spend']           0.8006      0.047     17.015      0.000       0.705       0.896
==============================================================================
Omnibus:                       11.158   Durbin-Watson:                   2.170
Prob(Omnibus):                  0.004   Jarque-Bera (JB):               12.160
Skew:                          -0.917   Prob(JB):                      0.00229
Kurtosis:                       5.030   Cond. No.                     6.17e+05
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 6.17e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

np.mean(Startups_train_model.resid) # 0
np.sqrt(sum(Startups_train_model.resid**2)/49) # 7978


#model for testing
Startups_test_model = smf.ols("Startups_test['Profit'] ~ Startups_test['Marketing Spend']+Startups_test['R&D Spend']",data=Startups_test).fit() 
Startups_test_model.summary()


np.mean(Startups_test_model.resid) # 0
np.sqrt(sum(Startups_test_model.resid**2)/49) # 6918

# Predict sales of Computers
comp_sale = pd.read_csv("C:/Users/cawasthi/Desktop/Data Science/R ML Code/liner_regression/Computer_Data.csv")
comp_sale.columns
comp_sale.drop("Unnamed: 0",axis=1,inplace=True)

comp_sale.isnull().sum() #no NULL values 

comp_sale = pd.get_dummies(comp_sale )

comp_sale.columns
comp_sale.drop(["cd_no","multi_no","premium_no"],axis=1,inplace=True)

comp_sale.corr()
import seaborn as sns
sns.set()
plot=sns.pairplot(comp_sale)
plot.savefig("C:/Users/cawasthi/Desktop/Data Science/R ML Code/output.png")

plt.hist(np.log(comp_sale.hd)) # to normalize
plt.hist(np.log(comp_sale.ram)) # to normalize

comp_sale.hd=np.log(comp_sale.hd)
comp_sale.ram=np.log(comp_sale.ram)


import statsmodels.formula.api as smf # for regression model
         
# Preparing model                  
ml1 = smf.ols("comp_sale['price']~comp_sale['speed']+comp_sale['hd']+comp_sale['ram']+ comp_sale['screen'] + comp_sale['ads'] + comp_sale['trend']+comp_sale['cd_yes']+ comp_sale['multi_yes'] + comp_sale['premium_yes']",data=comp_sale).fit() 
ml1.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:     comp_sale['price']   R-squared:                       0.764
Model:                            OLS   Adj. R-squared:                  0.763
Method:                 Least Squares   F-statistic:                     2243.
Date:                Thu, 11 Jul 2019   Prob (F-statistic):               0.00
Time:                        09:46:10   Log-Likelihood:                -44201.
No. Observations:                6259   AIC:                         8.842e+04
Df Residuals:                    6249   BIC:                         8.849e+04
Df Model:                           9                                         
Covariance Type:            nonrobust                                         
============================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------------------
Intercept                -1561.2019     77.810    -20.064      0.000   -1713.737   -1408.667
comp_sale['speed']           8.4477      0.191     44.251      0.000       8.073       8.822
comp_sale['hd']            352.8750     12.694     27.799      0.000     327.991     377.759
comp_sale['ram']           393.1903     10.670     36.849      0.000     372.273     414.108
comp_sale['screen']        113.5487      4.120     27.558      0.000     105.471     121.626
comp_sale['ads']             0.4505      0.052      8.659      0.000       0.348       0.552
comp_sale['trend']         -50.8735      0.666    -76.429      0.000     -52.178     -49.569
comp_sale['cd_yes']         59.0611      9.820      6.014      0.000      39.811      78.311
comp_sale['multi_yes']      75.5840     11.656      6.485      0.000      52.735      98.433
comp_sale['premium_yes']  -509.5891     12.731    -40.028      0.000    -534.546    -484.633
==============================================================================
Omnibus:                      845.636   Durbin-Watson:                   1.953
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1770.059
Skew:                           0.824   Prob(JB):                         0.00
Kurtosis:                       5.017   Cond. No.                     5.26e+03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 5.26e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

Price_pred=ml1.predict()
#intercept are in significant
np.mean(ml1.resid) #0
np.sqrt(sum(ml1.resid**2)/49) #3191.03

# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(ml1.resid_pearson, dist="norm", plot=pylab)

#to check for influential record
import statsmodels.api as sm
sm.graphics.influence_plot(ml1,size = 10**2)

############ Homoscedasticity #######

# Residuals VS Fitted Values 
plt.scatter(Price_pred,ml1.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
comp_sale_train,comp_sale_test  = train_test_split(comp_sale,test_size = 0.2) # 20% size

# Preparing model for traning data               
comp_sale_model = smf.ols("comp_sale_train['price']~comp_sale_train['speed']+comp_sale_train['hd']+comp_sale_train['ram']+ comp_sale_train['screen'] + comp_sale_train['ads'] + comp_sale_train['trend']+comp_sale_train['cd_yes']+ comp_sale_train['multi_yes'] + comp_sale_train['premium_yes']",data=comp_sale_train).fit() 
comp_sale_model.summary()

"""
                               OLS Regression Results                               
====================================================================================
Dep. Variable:     comp_sale_train['price']   R-squared:                       0.765
Model:                                  OLS   Adj. R-squared:                  0.765
Method:                       Least Squares   F-statistic:                     1809.
Date:                      Thu, 11 Jul 2019   Prob (F-statistic):               0.00
Time:                              10:57:49   Log-Likelihood:                -35331.
No. Observations:                      5007   AIC:                         7.068e+04
Df Residuals:                          4997   BIC:                         7.075e+04
Df Model:                                 9                                         
Covariance Type:                  nonrobust                                         
==================================================================================================
                                     coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------------------------
Intercept                      -1642.7826     85.939    -19.116      0.000   -1811.261   -1474.304
comp_sale_train['speed']           8.5816      0.212     40.439      0.000       8.166       8.998
comp_sale_train['hd']            357.7330     14.004     25.546      0.000     330.280     385.186
comp_sale_train['ram']           386.6582     11.816     32.724      0.000     363.494     409.822
comp_sale_train['screen']        116.2434      4.561     25.486      0.000     107.302     125.185
comp_sale_train['ads']             0.5073      0.058      8.774      0.000       0.394       0.621
comp_sale_train['trend']         -50.4347      0.740    -68.200      0.000     -51.884     -48.985
comp_sale_train['cd_yes']         57.4991     10.878      5.286      0.000      36.173      78.825
comp_sale_train['multi_yes']      78.7745     12.885      6.114      0.000      53.515     104.034
comp_sale_train['premium_yes']  -510.9343     13.938    -36.658      0.000    -538.259    -483.610
==============================================================================
Omnibus:                      674.936   Durbin-Watson:                   1.991
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1426.852
Skew:                           0.818   Prob(JB):                    1.46e-310
Kurtosis:                       5.041   Cond. No.                     5.22e+03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 5.22e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
"""
price = comp_sale_model.predict(comp_sale_train)

np.mean(comp_sale_model.resid) #0
np.sqrt(sum(comp_sale_model.resid**2)/49) #2837.74

# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(comp_sale_model.resid_pearson, dist="norm", plot=pylab)

#to check for influential record
import statsmodels.api as sm
sm.graphics.influence_plot(comp_sale_model,size = 10**2)

############ Homoscedasticity #######

# Residuals VS Fitted Values 
plt.scatter(price,comp_sale_model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

# Preparing model for testing data               
comp_sale_model = smf.ols("comp_sale_test['price']~comp_sale_test['speed']+comp_sale_test['hd']+comp_sale_test['ram']+ comp_sale_test['screen'] + comp_sale_test['ads'] + comp_sale_test['trend']+comp_sale_test['cd_yes']+ comp_sale_test['multi_yes'] + comp_sale_test['premium_yes']",data=comp_sale_test).fit() 
comp_sale_model.summary()

"""
                               OLS Regression Results                              
===================================================================================
Dep. Variable:     comp_sale_test['price']   R-squared:                       0.761
Model:                                 OLS   Adj. R-squared:                  0.759
Method:                      Least Squares   F-statistic:                     438.8
Date:                     Thu, 11 Jul 2019   Prob (F-statistic):               0.00
Time:                             11:06:48   Log-Likelihood:                -8861.5
No. Observations:                     1252   AIC:                         1.774e+04
Df Residuals:                         1242   BIC:                         1.779e+04
Df Model:                                9                                         
Covariance Type:                 nonrobust                                         
=================================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
Intercept                     -1215.1909    182.557     -6.657      0.000   -1573.345    -857.037
comp_sale_test['speed']           7.8466      0.437     17.940      0.000       6.989       8.705
comp_sale_test['hd']            325.0777     30.059     10.815      0.000     266.106     384.050
comp_sale_test['ram']           423.1054     24.864     17.017      0.000     374.326     471.885
comp_sale_test['screen']        103.7915      9.589     10.825      0.000      84.980     122.603
comp_sale_test['ads']             0.2222      0.120      1.857      0.064      -0.013       0.457
comp_sale_test['trend']         -52.3785      1.531    -34.211      0.000     -55.382     -49.375
comp_sale_test['cd_yes']         70.9272     22.832      3.107      0.002      26.134     115.720
comp_sale_test['multi_yes']      56.5376     27.387      2.064      0.039       2.807     110.268
comp_sale_test['premium_yes']  -499.3998     31.196    -16.008      0.000    -560.603    -438.197
==============================================================================
Omnibus:                      168.544   Durbin-Watson:                   2.004
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              323.811
Skew:                           0.826   Prob(JB):                     4.85e-71
Kurtosis:                       4.865   Cond. No.                     5.42e+03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 5.42e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
"""
price = comp_sale_model.predict(comp_sale_test)

np.mean(comp_sale_model.resid) #0
np.sqrt(sum(comp_sale_model.resid**2)/49) #1449.74

# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(comp_sale_model.resid_pearson, dist="norm", plot=pylab)

#to check for influential record
import statsmodels.api as sm
sm.graphics.influence_plot(comp_sale_model,size = 10**2)

############ Homoscedasticity #######

# Residuals VS Fitted Values 
plt.scatter(price,comp_sale_model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")




