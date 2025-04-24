import numpy as np
from scipy.stats import norm

def d1_d2(S, K, T, r, q, sigma):
    sqrtT = np.sqrt(T)
    d1 = (np.log(S/K) + (r-q+1/2*(sigma**2))*T)/(sigma*sqrtT)
    d2 = d1 - sigma*sqrtT
    return d1, d2
    
def EuropeanOption(S, K, T, r, q, sigma, is_call=True):
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    # put price
    if not is_call:
        P = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
        return P.item()
    # call price
    C = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return C.item()
    
def EuropeanGreeks(S, K, T, r, q, sigma, is_call=True):
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    df_r = np.exp(-r*T)
    df_q = np.exp(-q*T)
    sqrtT = np.sqrt(T)
    norm_pdf_d1 = norm.pdf(d1)
    norm_cdf_d1 =norm.cdf(d1)
    norm_cdf_m_d1 = norm.cdf(-d1)
    norm_cdf_d2 = norm.cdf(d2)
    norm_cdf_m_d2 = norm.cdf(-d2)
    # delta
    if not is_call:
        delta = df_q*norm_cdf_d1 - 1
    else:
        delta = df_q*norm_cdf_d1
    # gamma
    gamma = df_q*norm_pdf_d1/(S*sigma*sqrtT)
    # vega
    vega = S*df_q*norm_pdf_d1*sqrtT
    # theta per year := dV/dT, theta per day := dV/dt
    if not is_call:
        thetaPerYear = - S*sigma*df_q*norm_pdf_d1/(2*sqrtT) + q*S*df_q*norm_cdf_m_d1 - r*K*df_r*norm_cdf_m_d2
        thetaPerDay = - S*sigma*df_q*norm_pdf_d1/(2*sqrtT) - q*S*df_q*norm_cdf_m_d1 + r*K*df_r*norm_cdf_m_d2
    else:
        thetaPerYear = - S*sigma*df_q*norm_pdf_d1/(2*sqrtT) - q*S*df_q*norm_cdf_d1 + r*K*df_r*norm_cdf_d2
        thetaPerDay = - S*sigma*df_q*norm_pdf_d1/(2*sqrtT) + q*S*df_q*norm_cdf_d1 - r*K*df_r*norm_cdf_d2
    # rho
    if not is_call:
        rho = -K*T*df_r*norm_cdf_m_d2
    else:
        rho = K*T*df_r*norm_cdf_d2
    return {'delta':delta.item(),
            'gamma':gamma.item(),
            'vega':vega.item(),
            'theta(per year)':thetaPerYear.item(),
            'theta(per day)':thetaPerDay.item(),
            'rho':rho.item()}
    
