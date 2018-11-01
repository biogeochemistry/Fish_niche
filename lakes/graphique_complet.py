import numpy as np
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.api import add_constant, OLS

def graphique(x,y,xerr,yerr,coeff,r_value,variable_test,slope,intercept,variable_analyzed):
    """
        create a file of a lake initiated with a max_depth and area.
        Assumes to have a cone shaped bathymetry curve
        :param max_depth: maximum depth in metre
        :param area: area in metre^2
        :param outpath: filename where an init file of Mylake will be written
        :type max_depth: int
        :type area: int
        :type outpath: str
        :return: string to be written to an init file of MyLake
    """
    #5-7-2018 MC
    plt.style.use('seaborn-poster')
    lineStart = 0
    if variable_analyzed == 'SECCHI':
        lineEnd = 14
    else:
        lineEnd = 14000
    fig, ax = plt.subplots(figsize=(18.0, 10.0))
    plt.plot ( [lineStart, lineEnd], [lineStart, lineEnd], 'k-', color='b',label="1x+0" )

    (_, caps, _)=plt.errorbar ( x,y, xerr=xerr, yerr=yerr, fmt='o',color="r",markersize=4, capsize=10 )
    for cap in caps:
        cap.set_markeredgewidth ( 0.5 )

    fig.suptitle("")
    fig.tight_layout(pad=2)
    ax.grid(True)
    fig.savefig('filename1.png', dpi=125)

    x = add_constant ( x ) # constant intercept term
    # Model: y ~ x + c
    model = OLS ( y, x )
    fitted = model.fit()
    rsquared = fitted.rsquared
    x_pred = np.linspace(x.min(), x.max(), 50)
    x_pred2 = add_constant ( x_pred )
    y_pred = fitted.predict(x_pred2)

    ax.plot(x_pred, y_pred, '-', color='k', linewidth=2,label="linear regression (%0.3f x + %0.3f)"%(slope,intercept))
    fig.savefig('filename2.png', dpi=125)

    print(fitted.params)     # the estimated parameters for the regression line
    print(fitted.summary())  # summary statistics for the regression

    y_hat = fitted.predict(x) # x is an array from line 12 above
    y_err = y - y_hat
    mean_x = x.T[1].mean()
    n = len(x)
    dof = n - fitted.df_model - 1

    t = stats.t.ppf(1-0.025, df=dof)
    s_err = np.sum(np.power(y_err, 2))
    conf = t * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((x_pred-mean_x),2)/((np.sum(np.power(x_pred,2))) - n*(np.power(mean_x,2))))))
    upper = y_pred + abs(conf)
    lower = y_pred - abs(conf)
    ax.fill_between(x_pred, lower, upper, color='#888888', alpha=0.5,label="Confidence interval")
    fig.savefig('filename3_%s.png'%coeff, dpi=125)

    sdev, lower, upper = wls_prediction_std(fitted, exog=x_pred2, alpha=0.025)
    ax.fill_between(x_pred, lower, upper, color='#888888', alpha=0.2,label="Prediction interval")
    plt.xlabel ( "data_samples" )
    plt.ylabel ( "model_samples" )
    plt.plot ( [], [], color='w', label="R^2 : %s"%rsquared )
    ax.legend (loc='lower right')
    fig.savefig('Mean_regression_%s_%s_%s.png'%(variable_analyzed,variable_test,coeff), dpi=125)
    return fitted.summary()

if __name__ == '__main__':
    i=1