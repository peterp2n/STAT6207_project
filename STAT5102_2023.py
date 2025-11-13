# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 08:39:14 2023

@author: kingchaucheung
"""
import warnings
import pandas as pd
import numpy as np
#from IPython.display import Image
from pylab import rcParams
warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 16, 9
#np.set_printoptions(suppress=True)
pd.options.display.float_format = '{:.6f}'.format
# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)

import matplotlib.pyplot as plt
import matplotlib 
#import scipy.stats
from scipy import stats
from pylab import rcParams
#from pandas.plotting import scatter_matrix
import seaborn as sns; sns.set()
#from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table, OLSInfluence
from scipy.stats import t, norm, chi2, f
import itertools
from itertools import chain, combinations
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import make_scorer,mean_squared_error, r2_score, mean_absolute_error
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.compat import lzip
import pylab as py
from itertools import chain, combinations
from scipy.stats import zscore
from numpy import linalg as LA
from random import sample

def ceiling(k,m):
    x = divmod(k,m)
    if x[1]>0:
        d = 1
    else:
        d = 0
    return x[0]+d

def gen_regstring(varlist,Y):
    st = Y+' ~ '
    if len(varlist) > 0:
        st = st+' + '.join(varlist)
    else:
        st = st + '1'
    return st

def plot_scatter(X,Y,d,dd,varlabels):
    m = np.polyfit(X, Y, d)
    for i in range(len(m)):
        if i==0:
            predicted = m[-1]
            Z = np.ones(len(X))
        else:
            Z = Z*X
            predicted = predicted + m[-1-i]*Z
    e = np.abs(Y - predicted)
    labels = list(range(len(Y)))
    lpf = [labels[x] for x in np.argsort(e)]
    erf = [X[x] for x in np.argsort(e)]
    spf = [Y[x] for x in np.argsort(e)]
    ls = np.argsort(X)
    XS = [X[s] for s in ls]
    pS = [predicted[s] for s in ls]
    plt.scatter(X,Y, color='purple')
    plt.plot(XS,pS,'r--')
    for c in range(1,dd):
        plt.annotate(lpf[-c], (erf[-c], spf[-c]))
    plt.xlabel(varlabels[0])
    plt.ylabel(varlabels[1])
    plt.title(f'Plot of {varlabels[0]} vs. {varlabels[1]}')

def plot_mscatter(X,Y,d,dd,xlabels,ylabel):
    h = X.shape[1]
    g = ceiling(h,2)
    plt.figure(figsize=(16, 4*g))
    palette = plt.get_cmap('Set1')
    for k in range(h):
        plt.subplot(g,2,k+1)
        m = np.polyfit(X[:,k], Y, d[k])
        for i in range(len(m)):
            if i==0:
                predicted = m[-1]
                Z = np.ones(len(X[:,k]))
            else:
                Z = Z*X[:,k]
                predicted = predicted + m[-1-i]*Z
        e = np.abs(Y - predicted)
        labels = list(range(len(Y)))
        lpf = [labels[x] for x in np.argsort(e)]
        erf = [X[x,k] for x in np.argsort(e)]
        spf = [Y[x] for x in np.argsort(e)]
        ls = np.argsort(X[:,k])
        XS = [X[s,k] for s in ls]
        pS = [predicted[s] for s in ls]
        plt.scatter(X[:,k],Y, color=palette(k+1))
        plt.plot(XS,pS,'r--')
        for c in range(1,dd):
            plt.annotate(lpf[-c], (erf[-c], spf[-c]))
        plt.xlabel(xlabels[k])
        plt.ylabel(ylabel)
    
def residual_plot(model,dd=5):
    res = OLSInfluence(model).resid_studentized_external.values
    ares = np.abs(res)
    pY = model.fittedvalues.values
    labels = list(range(len(ares)))
    lpf = [labels[x] for x in np.argsort(ares)]
    erf = [pY[x] for x in np.argsort(ares)]
    spf = [res[x] for x in np.argsort(ares)]
    plt.scatter(pY,res,color='darkblue')
    for c in range(1,dd+1):
        plt.annotate(lpf[-c], (erf[-c], spf[-c]))
    plt.axhline(y=0.0, color='red', linestyle='--')
    plt.axhline(y=3.0, color='red', linestyle='--')
    plt.axhline(y=-3.0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Standardized residuals')
    plt.title('Standardized residuals versus Predicted values')

def partial_plots(var,fulllist,depvar,labels,data,u=5):
    var1 = [x for x in fulllist if x != var]
    lm = smf.ols(formula= gen_regstring(var1,depvar), data = data).fit()
    lm1 = smf.ols(formula= gen_regstring(var1,var), data = data).fit()
    eY = lm.resid.values
    eX = lm1.resid.values
    m, b = np.polyfit(eX, eY, 1)
    df = pd.DataFrame({'eX':eX, 'eY':eY})
    md = smf.ols(formula= 'eY ~ eX', data = df).fit()
    err1 = OLSInfluence(md).cooks_distance[0].values
    Z = np.argsort(eX)
    eX2 = [eX[i] for i in Z]
    eY2 = [eY[i] for i in Z]
    pY2 = [md.fittedvalues.values[i] for i in Z]
    lpf1 = [labels[x] for x in np.argsort(np.abs(err1))]
    erf1 = [eX[x] for x in np.argsort(np.abs(err1))]
    spf1 = [eY[x] for x in np.argsort(np.abs(err1))]
    
    md4 = smf.ols(formula= gen_regstring(fulllist,depvar), data = data).fit()
    Xk = data[var].values
    ZX = np.argsort(Xk)
    Xk2 = [Xk[i] for i in ZX]
    R2 = [md4.resid.values[i]+md4.params[var]*Xk[i] for i in ZX]
    df = pd.DataFrame({'Xk2':Xk2, 'R2':R2})
    md5 = smf.ols(formula= 'R2 ~ Xk2', data = df).fit()
    err2 = OLSInfluence(md5).cooks_distance[0].values
    lpf2 = [labels[x] for x in np.argsort(np.abs(err2))]
    erf2 = [Xk2[x] for x in np.argsort(np.abs(err2))]
    spf2 = [R2[x] for x in np.argsort(np.abs(err2))]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.scatter(eX2,eY2,color='red')
    ax1.plot(eX2,pY2,'k--')
    for c in range(1,u+1):
        ax1.annotate(lpf1[-c], (erf1[-c], spf1[-c]))
    ax1.set_xlabel(f'Partial Regression Residuals on {var}')
    ax1.set_ylabel('Partial Regression Residuals on Y')
    ax1.set_title('Partial Regression Plot')
    
    ax2.scatter(Xk2,R2,color='red')
    ax2.plot(Xk2,md5.fittedvalues.values,'k--')
    for c in range(1,u+1):
        ax2.annotate(lpf2[-c], (erf2[-c], spf2[-c]))
    ax2.set_xlabel(f'{var}')
    ax2.set_ylabel('Adjusted Y')
    ax2.set_title('Partial Residuals Plot')

def hplot(varlist,depvar,labels,data,k=5):
    model = smf.ols(formula = gen_regstring(varlist,depvar),data=data).fit()
    H = OLSInfluence(model).hat_matrix_diag
    r = OLSInfluence(model).resid_studentized_external.values
    c = OLSInfluence(model).cooks_distance[0].values
    p = len(varlist)
    n = len(data)
    #c = c*(p+1)
    # custom grid appearance
    sh = sorted(H)
    lr = [labels[x] for x in np.argsort(np.abs(r))]
    er = [x for x in np.argsort(np.abs(r))]
    sr = [r[x] for x in np.argsort(np.abs(r))]
    lh = [labels[x] for x in np.argsort(H)]
    eh = [x for x in np.argsort(H)]
    lc = [labels[x] for x in np.argsort(np.abs(c))]
    ec = [x for x in np.argsort(np.abs(c))]
    sc = [c[x] for x in np.argsort(np.abs(c))]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    
    ax1.scatter(er,sr,color='red')
    ax1.set_title('Standardized Residuals')
    ax1.axhline(y = 0, color = 'g', linestyle = '--')
    ax1.axhline(y = 3, color = 'g', linestyle = '--')
    ax1.axhline(y = -3, color = 'g', linestyle = '--')
    for u in range(1,k+1):
        ax1.annotate(lr[-u], (er[-u], sr[-u]))
        
    ax2.scatter(eh,sh,color='red')
    ax2.axhline(y = 2*(p+1)/n, color = 'g', linestyle = '--')
    ax2.set_title('The leverage values')
    for u in range(1,k+1):
        ax2.annotate(lh[-u], (eh[-u], sh[-u]))
        
    ax3.scatter(ec,sc,color='red')
    ax3.axhline(y = 1, color = 'g', linestyle = '--')
    ax3.set_title('The Cooks Distance')
    for u in range(1,k+1):
        ax3.annotate(lc[-u], (ec[-u], sc[-u]))

def GQ_test(Y,X):
    name = ["F statistic", "p-value"]
    test = sms.het_goldfeldquandt(Y, X)
    return lzip(name, test)

def BP_test(model):
    names = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']

    #fit.model.exog = design matrix
    test = sms.het_breuschpagan(model.resid, model.model.exog)

    return lzip(names, test)

def cal_R2(X,Y):
    n = len(Y)
    p = X.shape[1]
    model = linear_model.LinearRegression(fit_intercept = True)
    model.fit(X,Y)
    return model.score(X,Y)

def cal_adjR2(X,Y):
    n = len(Y)
    p = X.shape[1]
    model = linear_model.LinearRegression(fit_intercept = True)
    model.fit(X,Y)
    R2 = model.score(X,Y)
    return 1 - (1-R2)*(n-1)/(n-p-1)

def cal_SSE(X,Y):
    model = linear_model.LinearRegression(fit_intercept = True)
    model.fit(X,Y)
    return mean_squared_error(Y,model.predict(X)) * len(Y)

def cal_PRESS(varlist,Y,data):
    lm = smf.ols(formula= gen_regstring(varlist,Y), data = data).fit()
    lm_result = OLSInfluence(lm)
    e = lm.resid.values/(1-lm_result.hat_matrix_diag)
    return sum(e*e)/len(e)

def cal_IC(X,Y):
    n = len(Y)
    k = X.shape[1]
    IC = {}
    IC['AIC'] = n*np.log(cal_SSE(X,Y)/n) + 2*(k+1)
    IC['BIC'] = n*np.log(cal_SSE(X,Y)/n) + (k+1)*np.log(n)
    return IC

def cal_MSC(varlist,fulllist,depvar,data):
    X = data[fulllist].values
    p = X.shape[1]
    Y = data[depvar].values
    n = len(Y)
    MSE = cal_SSE(X,Y)/(n-p-1)
    r = len(varlist)
    X1 = data[varlist].values
    output = {}
    output['R2'] = cal_R2(X1,Y)
    output['adjR2'] = cal_adjR2(X1,Y)
    output['Cp'] = cal_SSE(X1,Y)/MSE - n + 2*(r+1)
    output['PRESS'] = cal_PRESS(varlist,depvar,data)
    res = cal_IC(X1,Y)
    output['AIC'] = res['AIC']
    output['BIC'] = res['BIC']
    return output

def gen_combinations(input):
    return sum([list(map(list, combinations(input, i))) for i in range(1,len(input) + 1)], [])

def compare_models(fulllist, depvar, computer_data):
    models = []
    for s in gen_combinations(range(len(fulllist))):
        varlist = [fulllist[x] for x in s]
        fs = cal_MSC(varlist,fulllist,depvar,computer_data)
        models.append((varlist,1-fs['R2'],1-fs['adjR2'], fs['Cp'], fs['PRESS'], fs['AIC'], fs['BIC']))
    df = pd.DataFrame(models)
    df.columns=['model','1-R2','1-AdjR2','Cp','PRESS', 'AIC', 'BIC']
    return df

def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to the show density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram

def gen_interactions(X,de,pname):
    s = gen_combinations(range(len(pname)))
    VX = []
    VarX = []
    for i in range(len(s)):
        if len(s[i]) <= de:
            VX.append(np.prod(X[:,s[i]],axis=1))
            VarX.append(list([pname[x] for x in s[i]]))
    VarX2 = []
    for x in VarX:
        if len(x) > 1:
            VarX2.append('_'.join(x))
        else:
            VarX2.append(x[0])
    VX = pd.DataFrame(np.array(VX).T,columns=VarX2)
    return VX, VarX2

def cal_p(varlist, depvar, data):
    lm = smf.ols(formula= gen_regstring(varlist,depvar), data = data).fit()
    return lm.pvalues[1:]

def forward_selection(fulllist, depvar, data, sle=0.05):
    Not_selected_list = fulllist
    selected_list = []
    report = []
    stop = 0
    stage = 0
    while stop==0:
        pvals = []
        for i in range(len(Not_selected_list)):
            x = Not_selected_list[i]
            s = selected_list+[x]
            pvals.append(cal_p(s,depvar,data).values[-1])
        if min(pvals) <= sle:
            selected_list.append(Not_selected_list[np.argmin(pvals)])
            stage = stage+1
            report.append((Not_selected_list[np.argmin(pvals)], min(pvals), stage))
            Not_selected_list = [x for x in Not_selected_list if x not in selected_list]
        else:
            stop = 1
    return report

def backward_elimination(fulllist, depvar, data, sls=0.05):
    Not_selected_list = []
    selected_list = fulllist
    report = []
    stop = 0
    stage = 0
    while stop==0:
        pvals = cal_p(selected_list,depvar,data).values
        if max(pvals) >= sls:
            s = selected_list[np.argmax(pvals)]
            Not_selected_list.append(s)
            stage = stage+1
            report.append((s,max(pvals),stage))
            selected_list = [x for x in selected_list if x not in Not_selected_list]
        else:
            stop = 1
    output={}
    output['report'] = report
    output['selected'] = selected_list
    return output

def stepwise_regresson(fulllist, depvar, data, sle=0.1, sls=0.1, type = 'forward'):
    if type=='forward':
        Not_selected_list = fulllist
        selected_list = []
        report = []
        stop = 0
        while stop==0:
            pvals = []
            for i in range(len(Not_selected_list)):
                x = Not_selected_list[i]
                s = selected_list+[x]
                pvals.append(cal_p(s,depvar,data).values[-1])
            if min(pvals) <= sle:
                u = Not_selected_list[np.argmin(pvals)]
                selected_list.append(u)
                report.append((u,min(pvals),1))
                Not_selected_list = [x for x in Not_selected_list if x not in selected_list]
                pvals = cal_p(selected_list,depvar,data).values
                if max(pvals) > sls:
                    uu = selected_list[np.argmax(pvals)]
                    Not_selected_list.append(uu)
                    report.append((uu,max(pvals),-1))
                    selected_list = [x for x in selected_list if x not in Not_selected_list]
            else:
                if len(selected_list) > 0:
                    pvals = cal_p(selected_list,depvar,data).values
                    if max(pvals) > sls:
                        uu = selected_list[np.argmax(pvals)]
                        Not_selected_list.append(uu)
                        report.append((uu,max(pvals),-1))
                        selected_list = [x for x in selected_list if x not in Not_selected_list]
                    else:
                        stop=1
                else:
                    stop=1
        output={}
        output['report'] = report
        output['selected'] = selected_list
        return output
    else:
        Not_selected_list = []
        selected_list = fulllist
        report = []
        stop = 0
        while stop==0:
            pvals = cal_p(selected_list,depvar,data).values
            if max(pvals) >= sls:
                u = selected_list[np.argmax(pvals)]
                Not_selected_list.append(u)
                report.append((u,max(pvals),-1))
                selected_list = [x for x in selected_list if x not in Not_selected_list]
                pvals = []
                for i in range(len(Not_selected_list)):
                    x = Not_selected_list[i]
                    s = selected_list+[x]
                    pvals.append(cal_p(s,depvar,data).values[-1])
                if min(pvals) <= sle:
                    u = Not_selected_list[np.argmin(pvals)]
                    selected_list.append(u)
                    report.append((u,min(pvals),1))
                    Not_selected_list = [x for x in Not_selected_list if x not in selected_list]
            else:
                if len(Not_selected_list) > 0:
                    pvals = []
                    for i in range(len(Not_selected_list)):
                        x = Not_selected_list[i]
                        s = selected_list+[x]
                        pvals.append(cal_p(s,depvar,data).values[-1])
                    if min(pvals) <= sle:
                        u = Not_selected_list[np.argmin(pvals)]
                        selected_list.append(u)
                        report.append((u,min(pvals),1))
                        Not_selected_list = [x for x in Not_selected_list if x not in selected_list]
                    else:
                        stop=1
                else:
                    stop = 1
        output={}
        output['report'] = report
        output['selected'] = selected_list
        return output

def stagewise_regression(fulllist,depvar,data):
    data2 = data.copy()
    Y = data2[depvar].values
    data2['e'] = Y
    e = Y
    p = len(fulllist)
    n = data2.shape[0]
    beta = np.zeros(p+1)
    stop = 0
    while stop==0:
        SSES = []
        for j in range(p):
            X = data2[fulllist[j]].values
            X = X.reshape((len(X),1))
            SSES.append(cal_SSE(X,e))
        m = np.argmin(SSES)
        lm = smf.ols(formula= gen_regstring([fulllist[m]],'e'), data = data2).fit()
        a = lm.pvalues.values[1]
        b = lm.params.values
        if a > 0.05:
            stop = 1
        else:
            beta[0] = beta[0] + b[0]
            beta[m+1] = beta[m+1] + b[1]
            e = e - b[0] - b[1]*data[fulllist[m]].values
            data2['e'] = e
    return dict([(x,y) for y,x in zip(beta, ['Intercept']+fulllist)])

def find_box_cox(Z,Y):
    n=len(Y)
    p=Z.shape[1]
    lda = np.pad(np.linspace(-10,10,1000),(0,1),'constant')
    max_logL = -100000000000
    Z = np.column_stack((np.ones(n),Z))
    Z2_inv = np.linalg.inv(Z.T.dot(Z)+0.00001*np.eye(p+1))
    beta = Z2_inv.dot(Z.T)
    H = Z.dot(beta)
    max_r = 0
    for r in lda:
        if r==0:
            Y2 = np.log(Y)
        else:
            Y2 = (np.power(Y,r)-1)/r
        e = Y2 - H.dot(Y2)
        S2 = e.T.dot(e)/n
        logL = -(n/2)*np.log(S2) + (r-1)*sum(np.log(Y))
        if logL > max_logL:
            max_logL = logL
            max_r = r
    if max_r==0:
        Y2 = np.log(Y)
    else:
        Y2 = (np.power(Y,max_r)-1)/max_r
    beta = beta.dot(Y2)
    e = Y2 - H.dot(Y2)
    S2 = e.T.dot(e)/n
    shapiro_test = stats.shapiro(e)
    kstest = stats.kstest((e-np.mean(e))/np.std(e), 'norm')
    output = {}
    output['Y'] = Y2
    output['max_lda'] = max_r
    output['beta'] = beta
    output['e'] = e
    output['S2'] = S2
    output['shapiro_test'] = shapiro_test
    output['kstest'] = kstest
    return output

def cal_vif(pname,df):
    vif_series = pd.Series(
        [variance_inflation_factor(df[pname].values, i) for i in range(len(pname))],
        index = pname,dtype = float)
    output={}
    output['vif'] = vif_series
    output['VIF'] = vif_series.mean()
    return output

def cal_cond_indices(df,pname):
    v, vc = np.linalg.eig(df[pname].corr())
    z = np.argsort(v)
    vm = v[z[-1]]
    output={}
    output['Kappa'] = pd.DataFrame({'col':[pname[x] for x in z],'eigenval':[v[x] for x in z],'cond_index':[np.sqrt(vm/v[x]) for x in z]})
    output['eigenvectors'] = pd.DataFrame([vc[x] for x in z],index=[pname[x] for x in z])
    return output

def plot_predicted(model,test_data,response):
    predictions = model.predict(test_data)
    plt.scatter(response,predictions,lw=1.5,color='red')
    #plt.plot(prediction_summary['actual'],prediction_summary['obs_ci_lower'],'g--')
    #plt.plot(prediction_summary['actual'],prediction_summary['obs_ci_upper'],'g--')
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=1, scalex=False, scaley=False)
    plt.xlim([min(response)*0.95,max(response)*1.05])
    plt.ylim([min(response)*0.95,max(response)*1.05])
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted')
    plt.title('Plot of Actual vs. Predicted Responses')

def PCA(df,pname):
    w,v = LA.eig(df[pname].corr())
    S = df[pname].apply(zscore).dot(v)
    S.columns = [f'P{i}' for i in range(1,len(pname)+1)]
    return S

def find_lasso_beta(X,Y,lda,gamma=0.01):
    if np.ndim(X)==1:
        p=1
    else:
        p = X.shape[1]
    n = len(Y)
    X = np.column_stack((np.ones(n),X))
    Temp = X.T.dot(X) + lda*np.eye(p+1)
    Temp_inv = np.linalg.inv(Temp)
    beta = Temp_inv.dot(X.T.dot(Y))
    stop = 0
    it = 0
    while stop==0:
        it = it+1
        beta0 = beta
        DB = -X.T.dot(Y) + X.T.dot(X.dot(beta)) + lda*np.sign(beta)
        beta = beta - gamma*DB
        if sum(np.abs(beta - beta0)) < 0.1:
            stop = 1
        elif it > 1000:
            stop = 1
    output = {}
    output['beta'] = beta
    e = Y - X.dot(beta)
    output['SSE'] = np.sqrt((1/(n-p-1))*e.T.dot(e))
    output['e'] = e
    return output

def cal_PRESS_lasso(X,Y,lda):
    if np.ndim(X)==1:
        p=1
    else:
        p = X.shape[1]
    n = len(Y)
    m = int(n/10)
    XX = np.column_stack((np.ones(n),X))
    err = []
    for i in sample(range(n),m):
        idx = [j for j in range(n) if j != i]
        Xi = X[idx]
        Yi = Y[idx]
        model = find_lasso_beta(Xi,Yi,lda)
        beta = model['beta']
        pYi = XX[i].T.dot(beta)
        err.append(Y[i]-pYi)
    err2 = [x*x for x in err]
    return np.sqrt(sum(err2)/m)

def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    plt.legend(
        loc="lower left",
        frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

def adj_r2_score(predictors, targets, predictions):
    r2 = r2_score(targets, predictions)
    n = predictors.shape[0]
    k = predictors.shape[1]
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))


# function to compute MAPE
def mape_score(targets, predictions):
    return np.mean(np.abs(targets - predictions) / targets) * 100


# function to compute different metrics to check performance of a regression model
def model_performance_regression(model, predictors, target):
    """
    Function to compute different metrics to check regression model performance

    model: regressor
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    r2 = r2_score(target, pred)  # to compute R-squared
    adjr2 = adj_r2_score(predictors, target, pred)  # to compute adjusted R-squared
    rmse = np.sqrt(mean_squared_error(target, pred))  # to compute RMSE
    mae = mean_absolute_error(target, pred)  # to compute MAE
    mape = mape_score(target, pred)  # to compute MAPE

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "RMSE": rmse,
            "MAE": mae,
            "R-squared": r2,
            "Adj. R-squared": adjr2,
            "MAPE": mape,
        },
        index=[0],
    )

    return df_perf

# RMSE
def rmse(predictions, targets):
    return np.sqrt(((targets - predictions) ** 2).mean())


# MAPE
def mape(predictions, targets):
    return np.mean(np.abs((targets - predictions)) / targets) * 100


# MAE
def mae(predictions, targets):
    return np.mean(np.abs((targets - predictions)))

# To see the feature importance of variables in the final model
def feature_importances(model, feature_names, n=10):
    if isinstance(model,LinearRegression):
        importances = model.coef_
    else:
        importances = model.feature_importances_
    zipped = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    for i, f in enumerate(zipped[:n]):
        print("%d: Feature: %s, %.3f" % (i+1, f[0], f[1]))


# Model Performance on test and train data
def model_pref(olsmodel, x_train, x_test, y_train,y_test):

    # Insample Prediction
    y_pred_train = olsmodel.predict(x_train)
    y_observed_train = y_train

    # Prediction on test data
    y_pred_test = olsmodel.predict(x_test)
    y_observed_test = y_test

    print(
        pd.DataFrame(
            {
                "Data": ["Train", "Test"],
                "RMSE": [
                    rmse(y_pred_train, y_observed_train),
                    rmse(y_pred_test, y_observed_test),
                ],
                "MAE": [
                    mae(y_pred_train, y_observed_train),
                    mae(y_pred_test, y_observed_test),
                ],
                "MAPE": [
                    mape(y_pred_train, y_observed_train),
                    mape(y_pred_test, y_observed_test),
                ],
            }
        )
    )

# Plot the histogram.
def norm_hist(data,bins=10):
    mu, std = norm.fit(data) 
    plt.hist(data, bins=25, density=True, alpha=0.6, color='purple',edgecolor='black', linewidth=1.2)
  
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
  
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
    plt.title(title)
  
    plt.show()

def cal_ridge_beta(lda,X,Y):
    n=len(Y)
    if np.ndim(X)==1:
        p=1
    else:
        p=X.shape[1]
    X = np.column_stack((np.ones(n),X))
    X2_inv = np.linalg.inv(X.T.dot(X)+lda*np.eye(p+1))
    beta = X2_inv.dot(X.T.dot(Y))
    e = Y - X.dot(beta)
    output={}
    output['beta'] = beta
    output['e'] = e
    return output

def cal_GCV(lda,X,Y):
    if np.ndim(X)==1:
        p=1
    else:
        p = X.shape[1]
    n = len(Y)
    X = np.column_stack((np.ones(n),X))
    Temp = X.T.dot(X) + lda*np.eye(p+1)
    Temp_inv = np.linalg.inv(Temp)
    H = X.dot(Temp_inv.dot(X.T))
    k = np.trace(H)
    err = []
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        Xi = X[idx]
        Yi = Y[idx]
        Temp = Xi.T.dot(Xi) + lda*np.eye(p+1)
        Temp_inv = np.linalg.inv(Temp)
        beta = Temp_inv.dot(Xi.T.dot(Yi))
        pYi = X[i].T.dot(beta)
        err.append((Y[i]-pYi)/(1-(k/n)))
    err2 = [x*x for x in err]
    return np.sqrt(sum(err2)/n)