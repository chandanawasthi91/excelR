
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

'''
1) Calories_consumed-> predict weight gained using calories consumed
'''
cal_con=pd.read_csv("C:/Users/cawasthi/Desktop/Data Science/R ML Code/liner_regression/calories_consumed.csv")
cal_con.columns
np.corrcoef(cal_con['Weight gained in grams'],cal_con['Calories Consumed'])

plt.hist(cal_con['Calories Consumed'])

#boxplot to check for outliers
plt.boxplot(cal_con['Calories Consumed'])
plt.boxplot(cal_con['Weight gained in grams'])

#scaterplot to check the linera relation
plt.scatter(cal_con['Weight gained in grams'],cal_con['Calories Consumed']);plt.xlabel('Weight gained in grams');plt.ylabel('Calories Consumed')

#removing the outliers
cal_con1 = cal_con.drop(cal_con.index[cal_con['Weight gained in grams'] == 62],axis=0)
cal_con2 = cal_con1.drop(cal_con.index[cal_con['Calories Consumed'] == 1400],axis=0)
cal_con = cal_con2

# cal_con['Weight gained in grams'] ~ cal_con['Calories Consumed'] model1
import statsmodels.formula.api as smf
model=smf.ols("cal_con['Weight gained in grams'] ~ cal_con['Calories Consumed']",data=cal_con).fit()

# For getting coefficients of the varibles used in equation
model.params
model.summary()
'''
<class 'statsmodels.iolib.summary.Summary'>
"""
                                    OLS Regression Results                                   
=============================================================================================
Dep. Variable:     cal_con['Weight gained in grams']   R-squared:                       0.916
Model:                                           OLS   Adj. R-squared:                  0.908
Method:                                Least Squares   F-statistic:                     109.6
Date:                               Wed, 10 Jul 2019   Prob (F-statistic):           1.04e-06
Time:                                       10:39:42   Log-Likelihood:                -71.625
No. Observations:                                 12   AIC:                             147.3
Df Residuals:                                     10   BIC:                             148.2
Df Model:                                          1                                         
Covariance Type:                           nonrobust                                         
================================================================================================
                                   coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------
Intercept                     -675.8455    107.190     -6.305      0.000    -914.679    -437.012
cal_con['Calories Consumed']     0.4387      0.042     10.467      0.000       0.345       0.532
==============================================================================
Omnibus:                        2.808   Durbin-Watson:                   2.444
Prob(Omnibus):                  0.246   Jarque-Bera (JB):                1.352
Skew:                          -0.489   Prob(JB):                        0.509
Kurtosis:                       1.678   Cond. No.                     9.16e+03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 9.16e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
"""
'''

pred = model.predict(cal_con['Calories Consumed'])

error = cal_con['Weight gained in grams'] - pred
np.mean(error) # 0
np.sqrt(sum(error**2)/12) # 94.61

import matplotlib.pyplot as plt
plt.scatter(pred,cal_con['Weight gained in grams'],c="r")
#the plot looks slightly curvi linera so trying curvilinear equation


model=smf.ols("cal_con['Weight gained in grams'] ~ cal_con['Calories Consumed'] +(cal_con['Calories Consumed']*cal_con['Calories Consumed'])",data=cal_con).fit()

model.summary()
'''
"""
                                    OLS Regression Results                                   
=============================================================================================
Dep. Variable:     cal_con['Weight gained in grams']   R-squared:                       0.916
Model:                                           OLS   Adj. R-squared:                  0.908
Method:                                Least Squares   F-statistic:                     109.6
Date:                               Wed, 10 Jul 2019   Prob (F-statistic):           1.04e-06
Time:                                       10:42:27   Log-Likelihood:                -71.625
No. Observations:                                 12   AIC:                             147.3
Df Residuals:                                     10   BIC:                             148.2
Df Model:                                          1                                         
Covariance Type:                           nonrobust                                         
================================================================================================
                                   coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------
Intercept                     -675.8455    107.190     -6.305      0.000    -914.679    -437.012
cal_con['Calories Consumed']     0.4387      0.042     10.467      0.000       0.345       0.532
==============================================================================
Omnibus:                        2.808   Durbin-Watson:                   2.444
Prob(Omnibus):                  0.246   Jarque-Bera (JB):                1.352
Skew:                          -0.489   Prob(JB):                        0.509
Kurtosis:                       1.678   Cond. No.                     9.16e+03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 9.16e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
"""
'''

model.resid
pred = model.predict(cal_con['Calories Consumed'])
np.mean(model.resid) #0
np.sqrt(sum(model.resid**2)/12) #RMSE 94.61

import matplotlib.pyplot as plt
plt.scatter(pred,cal_con['Weight gained in grams'],c="r");

'''
2) Delivery_time -> Predict delivery time using sorting time 
'''
Delivery_time=pd.read_csv("C:/Users/cawasthi/Desktop/Data Science/R ML Code/liner_regression/delivery_time.csv")
Delivery_time.columns

#checking the correlation
Delivery_time.corr()
plt.scatter(np.log(Delivery_time['Delivery Time']),Delivery_time['Sorting Time'])
Delivery_time.describe()
#boxplot to check the outliers
plt.boxplot(Delivery_time['Delivery Time'])
plt.boxplot(Delivery_time['Sorting Time'])

#Delivery_time.drop(index=[18,7],inplace=True)

import statsmodels.formula.api as smf
#model=smf.ols("np.log(Delivery_time['Delivery Time']) ~ Delivery_time['Sorting Time'] + Delivery_time['Sorting Time']*Delivery_time['Sorting Time']",data=Delivery_time).fit()
model = smf.ols("Delivery_time['Delivery Time'] ~ Delivery_time['Sorting Time']",data=Delivery_time).fit()
model.summary()

'''
                                  OLS Regression Results                                  
==========================================================================================
Dep. Variable:     Delivery_time['Delivery Time']   R-squared:                       0.682
Model:                                        OLS   Adj. R-squared:                  0.666
Method:                             Least Squares   F-statistic:                     40.80
Date:                            Wed, 10 Jul 2019   Prob (F-statistic):           3.98e-06
Time:                                    10:44:14   Log-Likelihood:                -51.357
No. Observations:                              21   AIC:                             106.7
Df Residuals:                                  19   BIC:                             108.8
Df Model:                                       1                                         
Covariance Type:                        nonrobust                                         
=================================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
Intercept                         6.5827      1.722      3.823      0.001       2.979      10.186
Delivery_time['Sorting Time']     1.6490      0.258      6.387      0.000       1.109       2.189
==============================================================================
Omnibus:                        3.649   Durbin-Watson:                   1.248
Prob(Omnibus):                  0.161   Jarque-Bera (JB):                2.086
Skew:                           0.750   Prob(JB):                        0.352
Kurtosis:                       3.367   Cond. No.                         18.3
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
'''

Delivery_time_predict = model.predict(Delivery_time['Sorting Time'])
error = Delivery_time_predict -Delivery_time['Delivery Time']

np.mean(model.resid) #0
np.mean(error) #0
np.sqrt(sum(error**2)/12) #3.69


'''
3) Emp_data -> Build a prediction model for Churn_out_rate 
'''
Emp_data = pd.read_csv("C:/Users/cawasthi/Desktop/Data Science/R ML Code/liner_regression/emp_data.csv")
Emp_data.columns
Emp_data.corr()

plt.scatter(Emp_data['Salary_hike'],Emp_data['Churn_out_rate']) #plot loooks curvilinear

model_emp = smf.ols("Emp_data['Churn_out_rate'] ~ Emp_data['Salary_hike'] + I(Emp_data['Salary_hike']*Emp_data['Salary_hike'])",data=Emp_data).fit()
model_emp.summary()

'''
"""
                                OLS Regression Results                                
======================================================================================
Dep. Variable:     Emp_data['Churn_out_rate']   R-squared:                       0.974
Model:                                    OLS   Adj. R-squared:                  0.966
Method:                         Least Squares   F-statistic:                     129.6
Date:                        Wed, 10 Jul 2019   Prob (F-statistic):           2.95e-06
Time:                                10:45:22   Log-Likelihood:                -18.751
No. Observations:                          10   AIC:                             43.50
Df Residuals:                               7   BIC:                             44.41
Df Model:                                   2                                         
Covariance Type:                    nonrobust                                         
========================================================================================================================
                                                           coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------------------------------
Intercept                                             1647.0116    228.059      7.222      0.000    1107.738    2186.285
Emp_data['Salary_hike']                                 -1.7371      0.266     -6.538      0.000      -2.365      -1.109
I(Emp_data['Salary_hike'] * Emp_data['Salary_hike'])     0.0005   7.72e-05      6.158      0.000       0.000       0.001
==============================================================================
Omnibus:                        0.169   Durbin-Watson:                   1.152
Prob(Omnibus):                  0.919   Jarque-Bera (JB):                0.362
Skew:                           0.028   Prob(JB):                        0.835
Kurtosis:                       2.070   Cond. No.                     1.10e+09
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.1e+09. This might indicate that there are
strong multicollinearity or other numerical problems.
"""
'''

pred = model_emp.predict(Emp_data['Salary_hike'])
error = pred - Emp_data['Churn_out_rate']

np.mean(model_emp.resid)#0
np.sqrt(sum(error**2)/12) #1.44

'''
4) Salary_hike -> Build a prediction model for Salary_hike
'''
Salary_hike = pd.read_csv("C:/Users/cawasthi/Desktop/Data Science/R ML Code/liner_regression/Salary_Data.csv")
Salary_hike.columns
Salary_hike.corr()

plt.scatter(Salary_hike['YearsExperience'], Salary_hike['Salary'])

model = smf.ols("Salary_hike['Salary'] ~ Salary_hike['YearsExperience']",data=Salary_hike).fit()
model.summary()

'''
"""
                              OLS Regression Results                             
=================================================================================
Dep. Variable:     Salary_hike['Salary']   R-squared:                       0.957
Model:                               OLS   Adj. R-squared:                  0.955
Method:                    Least Squares   F-statistic:                     622.5
Date:                   Wed, 10 Jul 2019   Prob (F-statistic):           1.14e-20
Time:                           10:46:32   Log-Likelihood:                -301.44
No. Observations:                     30   AIC:                             606.9
Df Residuals:                         28   BIC:                             609.7
Df Model:                              1                                         
Covariance Type:               nonrobust                                         
==================================================================================================
                                     coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------------------------
Intercept                       2.579e+04   2273.053     11.347      0.000    2.11e+04    3.04e+04
Salary_hike['YearsExperience']  9449.9623    378.755     24.950      0.000    8674.119    1.02e+04
==============================================================================
Omnibus:                        2.140   Durbin-Watson:                   1.648
Prob(Omnibus):                  0.343   Jarque-Bera (JB):                1.569
Skew:                           0.363   Prob(JB):                        0.456
Kurtosis:                       2.147   Cond. No.                         13.2
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
'''

pred = model.predict(Salary_hike['YearsExperience'])
error = pred - Salary_hike['Salary']
np.mean(model.resid) #0

np.sqrt(sum(error**2)/12) #8841.79
