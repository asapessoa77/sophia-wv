# Processamento dos dados de ondas nos domínios do tempo e frequencia
# Henrique Pereira
#
# Referencias
# https://www.sciencedirect.com/topics/earth-and-planetary-sciences/wave-spectrum

# WAFO
# 1 Hm0   = 4*sqrt(m0)                         Significant wave height
# 2 Tm01  = 2*pi*m0/m1                         Mean wave period
# 3 Tm02  = 2*pi*sqrt(m0/m2)                   Mean zero-crossing period
# 4 Tm24  = 2*pi*sqrt(m2/m4)                   Mean period between maxima
# 5 Tm_10 = 2*pi*m_1/m0                        Energy period
# 6 Tp    = 2*pi/{w | max(S(w))}               Peak period
# 7 Ss    = 2*pi*Hm0/(g*Tm02^2)                Significant wave steepness
# 8 Sp    = 2*pi*Hm0/(g*Tp^2)                  Average wave steepness
# 9 Ka    = abs(int S(w)*exp(i*w*Tm02) dw ) /m0  Groupiness parameter
# 10 Rs    = (S(0.092)+S(0.12)+S(0.15)/(3*max(S(w)))
#                                              Quality control parameter
# 11 Tp1   = 2*pi*int S(w)^4 dw                Peak Period
#           ------------------                 (robust estimate for Tp)
#           int w*S(w)^4 dw

# 12 alpha = m2/sqrt(m0*m4)                          Irregularity factor
# 13 eps2  = sqrt(m0*m2/m1^2-1)                      Narrowness factor
# 14 eps4  = sqrt(1-m2^2/(m0*m4))=sqrt(1-alpha^2)    Broadness factor
# 15 Qp    = (2/m0^2)int_0^inf w*S(w)^2 dw           Peakedness factor


import numpy as np
from matplotlib import mlab
import pandas as pd
from scipy.stats import norm, kurtosis, skew
from scipy.signal import welch, csd, hilbert
# import wafo
# import wafo.data as wd
# import wafo.misc as wm
# import wafo.objects as wo
# import wafo.spectrum.models as wsm


def envelope_hilbert(x, Fs):
    """
    Compute the analytic signal, using the Hilbert transform.
    The Hilbert transformed signal can be obtained from np.imag(hilbert(x)),
    and the original signal from np.real(hilbert(x)).

    The amplitude envelope is given by magnitude of the analytic signal.
    The instantaneous frequency can be obtained by differentiating the instantaneous
    phase in respect to time. The instantaneous phase corresponds to the phase
    angle of the analytic signal.
    """
    analytic_signal = hilbert(x)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) /
                              (2.0*np.pi) * Fs)
    instantaneous_frequency = np.concatenate((instantaneous_frequency[:1],
                                              instantaneous_frequency))
    return amplitude_envelope

def time_domain(t, s1):

    pp = {}
    tt = {}
    # parametros de onda no dominio do tempo
    eta = s1 - np.mean(s1)

    # calcula posicoes do eixo x do cruzamento
    # do zero ascendente
    pos_x = []
    for i in range(len(eta)-1):
        if ((eta[i] < 0) & (eta[i+1] > 0)):
            x1 = t[i]
            x2 = t[i+1]
            y1 = eta[i]
            y2 = eta[i+1]
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
            x = -b / a
            pos_x.append([int(i), x])
    pos_x = np.array(pos_x)

    #calcula ondas individuas
    Cr, Ca, H, T = [], [], [], []
    for i in range(len(pos_x)-1):
        onda = eta[int(pos_x[i, 0]):int(pos_x[i+1, 0]+1)]
        cr = np.max(onda)
        Cr.append(cr)
        ca = np.min(onda)
        Ca.append(ca)
        H.append(cr + np.abs(ca))
        T.append(pos_x[i+1, 1] - pos_x[i, 1])


    # #criando os vetores H(altura),Cr(crista),Ca(cavado),T (periodo)
    # Cr, Ca, H, T = [], [], [], []

    # #acha os indices que cruzam o zero
    # z = np.where(np.diff(np.sign(eta)))[0]

    # #zeros ascendentes e descendentes
    # zas=z[0::2]
    # zde=z[1::2]

    # #calcula ondas individuas
    # for i in range(len(zas)-1):
    #     onda = eta[zas[i]:(zas[i+1])+1]
    #     cr = np.max(onda)
    #     Cr.append(cr)
    #     ca = np.min(onda)
    #     Ca.append(ca)
    #     H.append(cr + np.abs(ca))
    #     T.append(t[zas[i+1]] - t[zas[i]])
    # # H = np.array(H)
    # # T = np.array(T)

    alt_per = np.array([H, T]).T

    # matriz de altura e periodo em ordem
    # crescente de altuas
    alt_per_sort = alt_per[np.argsort(alt_per[:,0]), :]

    #coloca as alturas em ordem crescente
    Hss = alt_per_sort[:,0] # altura
    Tss = alt_per_sort[:,1] # periodo relacionado a altura

    #calcula a altura e periodo significativo (Hs e Ts)
    div = int(len(alt_per) / 3.0)
    pp['hs'] = np.mean(Hss[-div:])
    pp['ts'] = np.mean(Tss[-div:])

    #calcula a altura das 1/10 maiores (H 1/10)
    div1 = int(len(Hss) / 10.0)
    pp['h10'] = np.mean(Hss[-div1:]) #altura da media das um decimo maiores

    #altura maxima
    pp['hmax'] = np.max(H)    

    #periodo medio
    pp['tz'] = np.mean(T)

    #calcula periodo associado a altura maxima
    ind = np.where(alt_per[:,0] == pp['hmax'])[0][0]
    pp['thmax'] = T[ind]

    # relacao de Hmax por Hs
    pp['hmaxhs'] = pp['hmax'] / pp['hs']

    pp = pd.DataFrame(pp, index=[0])
    tt = pd.DataFrame()

    # dicionario com os parametros das ondas individuais
    tt.index.name = 'n'
    tt['H'] = H
    tt['T'] = T

    return pp, tt


def freq_domain(t, s1, s2=[], s3=[], Fs=None, NFFT=None):

    NOVERLAP = int(NFFT/2.0)

    # espectro de welch
    f, c11 = welch(x=s1, fs=Fs, window='hann', nperseg=NFFT,
                          noverlap=NOVERLAP, nfft=NFFT, detrend='constant',
                          return_onesided=True, scaling='density', axis=-1, average='mean')

    if len(s2) > 0:

        f, c22 = welch(x=s2, fs=Fs, window='hann', nperseg=NFFT,
                              noverlap=NOVERLAP, nfft=NFFT, detrend='constant',
                              return_onesided=True, scaling='density', axis=-1, average='mean')

        f, c33 = welch(x=s3, fs=Fs, window='hann', nperseg=NFFT,
                              noverlap=NOVERLAP, nfft=NFFT, detrend='constant',
                              return_onesided=True, scaling='density', axis=-1, average='mean')

        # espectros cruzados
        f, P12 = csd(x=s1, y=s2, fs=Fs, window='hann', nperseg=NFFT,
                            noverlap=NOVERLAP, nfft=NFFT, detrend='constant',
                            return_onesided=True, scaling='density', axis=-1, average='mean')
        c12, q12 = np.real(P12), np.imag(P12)

        f, P13 = csd(x=s1, y=s3, fs=Fs, window='hann', nperseg=NFFT,
                            noverlap=NOVERLAP, nfft=NFFT, detrend='constant',
                            return_onesided=True, scaling='density', axis=-1, average='mean')
        c13, q13 = np.real(P13), np.imag(P13)

        f, P23 = csd(x=s2, y=s3, fs=Fs, window='hann', nperseg=NFFT,
                            noverlap=NOVERLAP, nfft=NFFT, detrend='constant',
                            return_onesided=True, scaling='density', axis=-1, average='mean')
        c23, q23 = np.real(P23), np.imag(P23)

        # parametros em dataframe
        cc = {'f': f, 'c11': c11, 'c22': c22, 'c33': c33, 'P12': P12, 'c12': c12, 'q12': q12,
              'P13': P13, 'c13': c13, 'q13': q13, 'P23': P23, 'c23': c23, 'q23': q23}
        cc = pd.DataFrame(cc)
        cc.set_index('f', inplace=True)

        # remove a primeira coluna de media
        cc = cc.iloc[1:,:]

        # calcula direcao de onda - Long (1980))
        cc['a1'] = cc.q12 / (cc.c11 * (cc.c22 + cc.c33)) ** 0.5
        cc['b1'] = cc.q13 / (cc.c11 * (cc.c22 + cc.c33)) ** 0.5
        cc['a2'] = (cc.c22 - cc.c33) / (cc.c22 + cc.c33)
        cc['b2'] = 2 * cc.c23 / (cc.c22 + cc.c33)
        cc['theta'] = np.rad2deg(np.arctan2(cc.b1, cc.a1))
        cc.loc[cc.theta>360, 'theta'] = cc.loc[cc.theta>360, 'theta'] - 360
        cc.loc[cc.theta<0, 'theta'] = cc.loc[cc.theta<0, 'theta'] + 360

        # calculo do espalhamento angular (ref: manual waverider, pg. 36)
        m1_spr = np.sqrt( (cc.a1**2) + (cc.b1**2) )
        cc['spr'] = np.rad2deg(np.sqrt(2.0 * (1 - m1_spr)))

    else:
        cc = {'f': f, 'c11': c11}
        cc = pd.DataFrame(cc)
        cc.set_index('f', inplace=True)

    # dataframe com os parametros
    pp = {}

    # intervalo do vetor de frequencia
    dff = np.diff(cc.index)[0]

    # calcula momentos e parametros espectrais com o WAFO

    # calculo do espectro
    # S = wsm.SpecData1D(data=cc.c11.values,
    #                 args=cc.index.values,
    #                 type='freq',
    #                 freqtype='f',
    #                 tr=None,
    #                 h=np.inf,
    #                 v=None,
    #                 norm=False,
    #                 date=None)

    # # # parametros de onda
    # [ch, R, txt] = S.characteristic(fact=np.arange(0, 15), T=len(s1), g=9.81)

    # # # momentos espectrais
    # mom, mom_txt = S.moment(nr=4, even=False, j=0)

    # pp = pd.concat([pd.DataFrame(ch, index=[i+'_wafo' for i in txt]),
    #                 pd.DataFrame(mom, index=[i+'_wafo' for i in mom_txt])],
    #                 axis=0).T
    

    # # calculo do periodo de pico
    # pp['tp2_wafo'] = 1/cc.index[cc.c11 == cc.c11.max()][0]

    # calcula os momentos espectrais
    # pp['m-1'] = np.sum(cc.index*-1 * cc.c11) * dff
    pp['m0'] = np.sum(cc.index**0 * cc.c11) * dff
    pp['m1'] = np.sum(cc.index**1 * cc.c11) * dff
    pp['m2'] = np.sum(cc.index**2 * cc.c11) * dff
    pp['m3'] = np.sum(cc.index**3 * cc.c11) * dff
    pp['m4'] = np.sum(cc.index**4 * cc.c11) * dff

    # utiliza os momentos espectrais do WAFO
    # pp['m0'] = pp['m0_wafo']
    # pp['m1'] = pp['m0t_wafo']
    # pp['m2'] = pp['m0tt_wafo']
    # pp['m3'] = pp['m0ttt_wafo']
    # pp['m4'] = pp['m0tttt_wafo']

    #calculo da altura significativa
    pp['hm0'] = 4.01 * np.sqrt(pp['m0'])

    # frequencia de pico
    pp['fp'] = cc.index[cc.c11 == cc.c11.max()].values[0]

    # periodo de pico
    pp['tp'] = 1 / pp['fp']

    if len(s2) > 0:
        # direcao de pico
        pp['dp'] = cc.theta[cc.c11 == cc.c11.max()].values[0]

        # spread de pico
        pp['sprp'] = cc.spr[cc.c11 == cc.c11.max()].values[0]

        # acha direcao para frequencias especificas
        # absolute_differences = np.abs(cc.index - 0.6)
        # index = np.argmin(absolute_differences)
        # pp['dir_f06hz'] = cc.theta.values[index]

    # mean wave period (Tz) also denoted Tm02
    pp['tm02'] = np.sqrt(pp['m0']/pp['m2'])

    # mean period
    pp['tm'] = pp['m0'] / pp['m1']

    # # mean spectral frequency
    # mean_spec_freq = m1 / m0

    # # mean spectral period
    # mean_spec_period = 1.0 / mean_spec_freq

    # # average zero-up-crossing frequency
    # mean_zup_freq = np.sqrt(m2 / m0)

    # # average zero-up-crossing period
    # mean_zup_period = 1.0 / mean_zup_freq

    # spectral bandwidth (Cartwright & Longuet-Higgins, 1956)
    # pp['e'] = np.sqrt(1 - (pp['m2'] ** 2 / (pp['m0'] * pp['m2'])))

    # spectral bandwidth (Longuet-Higgins, 1975)
    pp['v'] = np.sqrt((pp['m0'] * pp['m2']) / (pp['m1'] ** 2) - 1)

    # Peakdness parameter (Goda, 1970)
    pp['Qp'] = (2.0 / pp['m0']**2) * sum(cc.index * cc.c11**2) * dff

    # calcula BFI
    pp['L0'] = 1.56 * pp['tp'] ** 2.0
    pp['k0'] = 2 * np.pi / pp['L0']
    pp['BFI'] = pp['k0'] * pp['m0'] ** (1.0/2.0) * pp['Qp'] * np.sqrt(2 * np.pi)

    pp = pd.DataFrame(pp, index=[0])

    return pp, cc

def prob_domain(s1, tt):

    pp = {}
    pp['kurt_n'] = kurtosis(s1)
    pp['skew_n'] = skew(s1)
    pp['kurt_H'] = kurtosis(tt['H'])
    pp['skew_H'] = skew(tt['H'])
    pp['kurt_T'] = kurtosis(tt['T'])
    pp['skew_T'] = skew(tt['T'])

    pp = pd.DataFrame(pp, index=[0])
    return pp


def wave_group(s1, ppt, tt, ppf, cc, Fs, NFFT):

    NOVERLAP = int(NFFT/2.0)

    pp = {}

    analytic_signal = hilbert(s1)
    envelope_hilbert = np.abs(analytic_signal)
    # instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    # instantaneous_frequency = (np.diff(instantaneous_phase) /
    #                           (2.0*np.pi) * Fs)
    # instantaneous_frequency = np.concatenate((instantaneous_frequency[:1],
    #                                           instantaneous_frequency))

    # calcula o envelope de hilbert
    # serie_hilbert = envelope_hilbert(s1, Fs)

    # f, c11 = welch(x=s1, fs=Fs, window='hann', nperseg=NFFT,
    #                noverlap=NOVERLAP, nfft=NFFT, detrend='constant',
    #                return_onesided=True, scaling='density', axis=-1,
    #                average='mean')

    f_hilbert, c11_hilbert = welch(x=envelope_hilbert, fs=Fs, window='hann', nperseg=NFFT,
                                   noverlap=NOVERLAP, nfft=NFFT, detrend='constant',
                                   return_onesided=True, scaling='density', axis=-1,
                                   average='mean')

    # pp['m0'] = np.sum(f**0 * c11) * (f[3] - f[2])
    pp['m0_hilbert'] = np.sum(f_hilbert**0 * c11_hilbert) * (f_hilbert[3] - f_hilbert[2])

    # calcula parametro de agrupamento (GF)
    # pp['GF_hilbert'] = np.sqrt(pp['m0_hilbert']) / pp['m0']
    # pp['GF_hilbert'] = pp['m0_hilbert'] / pp['m0']
    # pp['GF_hilbert'] = np.sqrt(pp['m0']) / pp['m0_hilbert']
    # pp['GF_hilbert'] = np.sqrt(pp['m0_hilbert']) / pp['m0']
    pp['GF_hilbert'] = pp['m0_hilbert'] / ppf.m0
    # pp['GF_hilbert1'] = np.std(serie_hilbert) / np.mean(serie_hilbert)

    tt = pd.concat([tt['H'], tt['T']], axis=1)    
    tt['H1'] = tt['H'].shift(periods=1)
    tt['T1'] = tt['T'].shift(periods=1)
    pp['corr_H'] = tt[['H','H1']].corr().iloc[0,1]
    pp['corr_T'] = tt[['T','T1']].corr().iloc[0,1]

    # calcula o numero de runs
    # - numero de alturas individuais acima do hs dividido
    # pelo numero de ondas individuais no registro
    pp['nruns'] = len(tt.loc[tt.H > ppt.hs.values[0]]) / len(tt)

    # parametro de grupo do wafo - Ka
    omega = 2 * np.pi * cc.index.values
    integral = np.trapz(cc.c11.values * np.exp(1j * omega * ppf.tm02.values), omega)
    Ka = np.abs(integral) / ppf.m0
    pp['Ka'] = Ka.values[0]

    pp = pd.DataFrame(pp, index=[0])

    return pp, envelope_hilbert






















def jonswap(f, Hm0, Tp, gamma=3.3, sigma_low=0.07, sigma_high=0.09, g=9.81):
    """
    Jonswap spectrum
    """
    # Pierson-Moskowitz
    alpha = 1. / (.23 + .03 * gamma - .185 / (1.9 + gamma)) / 16.
    E_pm = alpha * Hm0**2 * Tp**-4 * f**-5 * np.exp(-1.25 * (Tp * f)**-4)

    # JONSWAP
    sigma = np.ones(f.shape) * sigma_low
    sigma[f > 1./Tp] = sigma_high
    E_js = E_pm * gamma**np.exp(-0.5 * (Tp * f - 1)**2. / sigma**2.)
    return E_pm, E_js

def interp_spectra(p):
    """
    interpolacao para um vetor de frequencia fixo
    """
    freq_interp = np.linspace(0, 0.5, 50)
    spec1d_interp = []
    meandir_interp = []
    spread_interp = []
    for i in range(len(p['spec1d'])):
        spec1d_interp.append(np.interp(freq_interp, p['freq'].iloc[i,:], p['spec1d'].iloc[i,:]))
        meandir_interp.append(np.interp(freq_interp, p['freq'].iloc[i,:], p['meandir'].iloc[i,:]))
        spread_interp.append(np.interp(freq_interp, p['freq'].iloc[i,:], p['spread'].iloc[i,:]))
    spec1d_interp = pd.DataFrame(spec1d_interp, index=p['spec1d'].index)
    meandir_interp = pd.DataFrame(meandir_interp, index=p['meandir'].index)
    spread_interp = pd.DataFrame(spread_interp, index=p['spread'].index)
    return freq_interp, spec1d_interp, meandir_interp, spread_interp

def calc_spec2d_matrix(f, s, d, spr):
    """
    Calcula matriz do espectro 2d
    f - frequency (vec)
    s - 1d spectrum (vec)
    d - direction (vec)
    spr - spreading (vec) 
    """
    thetas = np.linspace(0, 360, 360)
    d2 = np.zeros((len(thetas), len(f)))
    for i in range(len(f)):
        dtheta = norm.pdf(thetas, d[i], spr[i]/2.35)
        d2[:,i] = s[i] * dtheta
    return d2, thetas, dtheta

# def proc_longuet_higgins(ll, NFFT, Fs):
#     """
#     Processamento dos dados com longuet-higgins
#     """
#     # pth = '/home/hp/Dropbox/doutorado/dados/HNE_rio_grande_200912/'
#     # listap = np.sort(glob(pth + '*.HNE'))

#     # gl = 16
#     # NFFT = int(1321 / (gl / 2))
#     # Fs = 1.28

#     param = []
#     for l in ll:
#         print ('LH -- {}'.format(l))
#         datestr = l.split('/')[-1].split('.')[0]
#         date = pd.to_datetime(datestr)
#         w1 = pd.read_csv(l, skiprows=12, sep='\s+', names=['time', 'n1', 'n3', 'n2'], index_col='time')
#         wavet = waveproc.ondat(w1.index.values, w1.n1.values, h=np.inf)
#         wavef = waveproc.ondaf(w1.n1.values, w1.n2.values, w1.n3.values, h=np.inf, nfft=NFFT, fs=Fs)
#         param.append([date, wavef['hm0'], wavef['tp'], wavef['dp']])

#         fig = plot_spec1d_dire1(wavef)
#         fig.savefig('/home/hp/Documents/daat/spec1d_dire1_{}.png'.format(datestr))
#         plt.close('all')

#     df = pd.DataFrame(param, columns=['date', 'hm0', 'tp', 'dp'])
#     df.set_index('date', inplace=True)
#     df['ws'] = df.Wspd
#     df['wd'] = df.Wdir
#     return df


# def espec1(x, nfft, fs):
#     """
#     Calculo do espectro 1d
#     """
#     s, f = mlab.psd(x=x, NFFT=int(nfft), Fs=fs, detrend=mlab.detrend_mean,
#                   window=mlab.window_hanning, noverlap=nfft/2)

#     #degrees of freedom
#     dof = len(x) / nfft * 2
    
#     #confidence interval 95%
#     ici = s * dof / 26.12
#     ics = s * dof / 5.63
    
#     # matriz de saida
#     aa = np.array([f,s,ici,ics]).T
#     return aa

# def espec2(x, y, nfft, fs):
#     """    
#     Calcula o espectro cruzado entre duas series reais
    
#     Dados de entrada: x = serie real 1 (potencia de 2)
#                       y = serie real 2 (potencia de 2)
#                       nfft - Numero de pontos utilizado para o calculo da FFT
#                       fs - frequencia de amostragem
    
#     Dados de saida: [aa2] - col 0: vetor de frequencia
#                             col 1: amplitude do espectro cruzado
#                             col 2: co-espectro
#                             col 3: quad-espectro
#                             col 4: espectro de fase
#                             col 5: espectro de coerencia
#                             col 6: intervalo de confianca inferior do espectro cruzado
#                             col 7: intervalo de confianca superior do espectro cruzado
#                             col 8: intervalo de confianca da coerencia
    
#     Infos:    detrend - mean
#               window - hanning
#               noverlap - 50%    
#     """

#     #cross-spectral density - welch method (complex valued)
#     s, f = mlab.csd(x, y, NFFT=nfft, Fs=fs, detrend=mlab.detrend_mean,
#                     window=mlab.window_hanning, noverlap=nfft/2)

#     # graus de liberdade    
#     dof = len(x) / nfft * 2

#     #co e quad espectro (real e imag) - verificar com parente
#     co = np.real(s)
#     qd = np.imag(s)
    
#     #phase (angle function)
#     ph = np.angle(s, deg=True)
    
#     #ecoherence between x and y (0-1)
#     coer = mlab.cohere(x, y, NFFT=nfft, Fs=fs, detrend=mlab.detrend_mean, window=mlab.window_hanning, noverlap=nfft/2)[0]
#     # coer = coer[0][1:]
    
#     #intervalo de confianca para a amplitude do espectro cruzado - 95%
#     ici = s * dof /26.12
#     ics = s * dof /5.63
    
#     #intervalo de confianca para coerencia
#     icc = np.zeros(len(s))
#     icc[:] = 1 - (0.05 ** (1 / (14 / 2.0 - 1)))
    
#     # matriz de saida
#     aa = np.array([f, s ,co, qd, ph, coer, ici, ics, icc]).T
#     return aa

# def numeronda(h, df, reg):
#     '''
#     Calcula o numero de onda (k)
    
#     Dados de entrada: h - profundidade
#                         deltaf - vetor de frequencia
#                         reg - comprimento do vetor de frequencia
    
#     Dados de saida: k - vetor numero de onda
#     '''

#     #gravidade
#     g = 9.8

#     #vetor numero de onda a ser criado
#     k = []

#     #k anterior
#     kant = 0.001

#     #k posterior
#     kpos = 0.0011

#     for j in range(reg):
#         sigma = (2 * np.pi * df[j] ) ** 2
#         while abs(kpos - kant) > 0.001:
#             kant = kpos
#             dfk = g * kant * h * (1 / np.cosh(kant*h)) ** 2 + g + np.tanh(kant * h)
#             fk = g * kant * np.tanh(kant * h) - sigma
#             kpos = kant - fk / dfk
#         kant = kpos - 0.002
#         k.append(kpos)
#     return k

# def ondaf(eta, etax, etay, h, nfft, fs):
#     """
#     Calcula parametros de onda no dominio da frequencia
    
#     Dados de entrada: eta - vetor de elevacao
#                       etax - vetor de deslocamento em x
#                       etay - vetor de deslocamento em y
#                       h - profundidade
#                       nfft - Numero de pontos utilizado para o calculo da FFT
#                       fs - frequencia de amostragem
#                       az - azimute da boia
    
#     Dados de saida: pondaf = [hm0 tp dp]
#     """

#     #espectro simples
#     sn = espec1(eta,nfft,fs)
#     snx = espec1(etax,nfft,fs)
#     sny = espec1(etay,nfft,fs)

#     #espectros cruzados
#     snn = espec2(eta,eta,nfft,fs)
#     snnx = espec2(eta,etax,nfft,fs)
#     snny = espec2(eta,etay,nfft,fs)
#     snxny = espec2(etax,etay,nfft,fs)
#     snxnx = espec2(etax,etax,nfft,fs)
#     snyny = espec2(etay,etay,nfft,fs)

#     #vetor de frequencia
#     f = sn[:,0]

#     #deltaf
#     df = f[1] - f[0]

#     #calculo do numero de onda
#     k = np.array(numeronda(h,f,len(f)))

#     #calculo dos coeficientes de fourier - NDBC 96_01 e Steele (1992)
#     c = snx[:,1] + sny[:,1]
#     cc = np.sqrt(sn[:,1] * c)

#     a1 = snnx[:,3] / cc
#     b1 = snny[:,3] / cc

#     a2 = (snx[:,1] - sny[:,1]) / c
#     b2 = 2 * snxny[:,2] / c

#     #calcula direcao de onda

#     #mean direction
#     dire1 = np.array([np.angle(np.complex(a1[i], b1[i]), deg=True) for i in range(len(a1))])

#     #principal direction
#     dire2 = 0.5 * np.array([np.angle(np.complex(a2[i], b2[i]),deg=True) for i in range(len(a2))])
    
#     #condicao para valores maiores que 360 e menores que 0
#     dire1[np.where(dire1 < 0)] = dire1[np.where(dire1 < 0)] + 360
#     dire1[np.where(dire1 > 360)] = dire1[np.where(dire1 > 360)] - 360
#     dire2[np.where(dire2 < 0)] = dire2[np.where(dire2 < 0)] + 360
#     dire2[np.where(dire2 > 360)] = dire2[np.where(dire2 > 360)] - 360

#     #acha o indice da frequencia de pico
#     ind = np.where(sn[:,1] == np.max(sn[:,1]))[0]

#     #periodo de pico
#     tp = (1. / f[ind])[0]

#     # calcula os momentos espectrais
#     m0 = np.sum(f**0 * sn[:,1]) * df
#     m1 = np.sum(f**1 * sn[:,1]) * df
#     m2 = np.sum(f**2 * sn[:,1]) * df
#     m3 = np.sum(f**3 * sn[:,1]) * df
#     m4 = np.sum(f**4 * sn[:,1]) * df
#     moments = np.array([m0, m1, m2, m3, m4])

#     #calculo da altura significativa
#     hm0 = 4.01 * np.sqrt(m0)

#     #direcao do periodo de pico
#     dp = dire1[ind][0]

#     # mean spectral frequency
#     mean_spec_freq = m1 / m0

#     # mean spectral period
#     mean_spec_period = 1.0 / mean_spec_freq

#     # average zero-up-crossing frequency
#     mean_zup_freq = np.sqrt(m2 / m0)

#     # average zero-up-crossing period
#     mean_zup_period = 1.0 / mean_zup_freq

#     # spectral bandwidth (Cartwright & Longuet-Higgins, 1956)
#     e = np.sqrt(1 - (m2 ** 2 / (m0 * m2)))

#     # spectral bandwidth (Longuet-Higgins, 1975)
#     v = np.sqrt((m0 * m2) / (m1 ** 2) - 1)

#     #Espalhamento direcional

#     #formula do sigma1 do livro Tucker&Pitt(2001) "Waves in Ocean Engineering" pags 196-198
#     c1 = np.sqrt(a1 ** 2 + b1 ** 2)
#     c2 = np.sqrt(a2 ** 2 + b2 ** 2)
    
#     s1 = c1 / (1 - c1)
#     s2 = (1 + 3 * c2 + np.sqrt(1 + 14 * c2 + c2 ** 2)) / (2 * (1 - c2))
    
#     sigma1 = np.sqrt(2 - 2 * c1) * 180 / np.pi
#     sigma2 = np.sqrt((1 - c2) / 2) * 180 / np.pi

#     # acha o espalhamento angular da frequencia de pico
#     sigma1p = np.real(sigma1[ind])[0]
#     sigma2p = np.real(sigma2[ind])[0]

#     waveout = {'hm0': hm0,
#                'tp': tp,
#                'dp': dp,
#                'f': f,
#                'sn': sn[:,1],
#                'dire1': dire1, 
#                'mean_spec_freq': mean_spec_freq,
#                'mean_spec_period': mean_spec_period,
#                'mean_zup_freq': mean_zup_freq,
#                'mean_zup_period': mean_zup_period,
#                'e': e,
#                'v': v,
#                'sigma1p': sigma1p,
#                'sigma2p': sigma2p,
#                'moments': moments,
#                'a1': a1,
#                'b1': b1,
#                'a2': a2,
#                'b2': b2,
#                }
#     return waveout

# def ondap(hm0, tp, dp, sn, dire1):
#     '''
#     Programa para calcular parametros
#     de onda nas particoes de sea e swell
    
#     desenvolvido para 32 gl
    
#     divide o espectro em 2 partes: 
#     parte 1 - 8.33 a 50 seg
#     parte 2 - 1.56 a 7.14 seg
    
#     calcula o periodo de pico de cada particao, e despreza o
#     pico 2 (menos energetico) se a energia for inferior a 15% da
#     energia do pico 1 (mais energetico)
#     '''
#     tp_swell = (1. / f[ind_swell])[0]
#     tp_sea = (1. / f[ind_sea])[0]

#     ii_swell = np.arange(0,18)
#     ii_sea = np.arange(18, len(sn))

#     # indice da frequencia de pico do swell
#     ind_swell = np.where(sn[ii_swell,1] == np.max(sn[ii_swell,1]))[0]

#     # indice da frequencia de pico do sea
#     ind_sea = np.where(sn[ii_sea,1] == np.max(sn[ii_sea,1]))[0] + ii_sea[0]

#     m0_swell = np.sum(f[ii_swell]**0 * sn[ii_swell,1]) * df
#     m0_sea = np.sum(f[ii_sea]**0 * sn[ii_sea,1]) * df

#     hm0_swell = 4.01 * np.sqrt(m0_swell)
#     hm0_sea = 4.01 * np.sqrt(m0_sea)

#     dp_swell = dire1[ind_swell][0]
#     dp_sea = dire1[ind_sea][0]

#     #vetor de frequencia e energia
#     f,s = sn[:,[0,1]].T

#     df = f[4] - f[3]

#     # seleciona os picos espectrais - considera somente 2 picos
#     g1=np.diff(s)
#     g1=np.sign(g1)
#     g1=np.diff(g1)
#     g1=np.concatenate(([0],g1))
#     g2=np.where(g1==-2)[0]
#     picos=1 # a principio e unimodal
#     l=np.size(g2)

#     # inicializar considerando ser unimodal
#     hm02 = np.nan #9999
#     tp2 = np.nan #9999
#     dp2 = np.nan #9999
#     hm01 = hm0
#     tp1 = tp 
#     dp1 = dp 

#     # se tiver picos
#     if l > 1: #verificando espacamento entre picos (espacamento maior que 4 df)
#         fr=np.argsort(s[g2])[::-1] #frequencia decrescente
#         er=np.sort(s[g2])[::-1] # energia decrescente

#         if (f[g2[fr[1]]]-f[g2[fr[0]]]) > 4*(f[1]-f[0]) and (er[1]/er[0] >= 0.15): #adota criterio de 4*deltaf
#             picos=2
        
#         # calcular o Hs dos picos pegando a cava e dividindo em pico 1 e pico 2
#         if picos == 2:
#             n1=g2[0] #pico mais energetico
#             n2=g2[1] #pico menos energetico
#             nc=np.where(g1[n1:n2]==2)[0] #indice da cava

#             #particao do swell e sea
#             swell = np.arange(n1+nc+1)
#             sea = np.arange(n1+nc+1,len(s))
#             #maxima energia do swell
#             esw = max(s[swell])
#             #maxima energia do sea
#             ese = max(s[sea])
#             #indice do pico do swell
#             isw = np.where(s==esw)[0][0]
#             #indice do pico do sea
#             ise = np.where(s==ese)[0][0]
#             #altura sig. do swell
#             hm0sw = 4.01 * np.sqrt(sum(s[swell]) * df)
#             #altura sig. do sea
#             hm0se = 4.01 * np.sqrt(sum(s[sea]) * df)
#             #periodo de pico do swell
#             tpsw = 1./f[isw]
#             #periodo de pico do sea
#             tpse = 1./f[ise]
#             #direcao do swell
#             dpsw = dire1[isw]
#             #direcao do sea
#             dpse = dire1[ise]

#             #deixa o pico 1 como swell e pico 2 como sea
#             en1 = esw ; en2 = ese
#             hm01 = hm0sw ; hm02 = hm0se
#             tp1 = tpsw ; tp2 = tpse
#             dp1 = dpsw ; dp2 = dpse

#     if tp2 == tp1:
#         tp2 = np.nan
#         hm01 = np.nan
#         dp2 = np.nan
        
#             #seleciona pico 1 como mais energetico
#             # e pico 2 com o menos energetico
#             # if esw > ese:
#             #   en1 = esw ; en2 = ese
#             #   hm01 = hm0sw ; hm02 = hm0se
#             #   tp1 = tpsw ; tp2 = tpse
#             #   dp1 = dpsw ; dp2 = dpse
#             # else:
#             #   en1 = ese ; en2 = esw
#             #   hm01 = hm0se ; hm02 = hm0sw
#             #   tp1 = tpse ; tp2 = tpsw
#             #   dp1 = dpse ; dp2 = dpsw

#     # pondaf1 = np.array([hm01, tp1, dp1, hm02, tp2, dp2])

#     return hm01, tp1, dp1, hm02, tp2, dp2 #pondaf1

# def spread_kuik1988(en, a1, b1):
#     '''
#     Programa para calcular o espalhamento angular
#     Kuik et al 1988
#     Entrada:
#     en - espectro 1D
#     a1 - coef de fourier de 1 ordem
#     b1 - conef de fourir de 1 ordem 
#     Saida:
#     spr - valor do espalhamento angular para cada
#     frequencia
#     '''
#     #esplhamento com vetor complexo - radianos?
#     sprc = (2 * (1 - ( (a1**2 + b1**2) / (en**2) ) **0.5) **0.5)
#     #soma a parte real e imag e coloca em graus
#     spr = (np.real(sprc) + np.imag(sprc)) * 180 / np.pi
#     #parece que aparece na parte real onde tem energia no espectro
#     sprr = np.real(sprc)
#     #diminui 360 nos maiores que 360
#     # spr[np.where(spr>360)[0]] = spr[np.where(spr>360)[0]] - 360
#     #coloca zeros nos valores muito altos
#     # spr[np.where(spr>360)[0]] = 0
#     return sprc, spr, sprr



# def waveproc(t, s1, s2=[], s3=[], Fs=None, NFFT=None, tempo=True, freq=True, prob=True, grupos=True):
#     """
#     Processamento de dados de ondas nos domínios
#     do tempo e frequência
#     inputs:
#     t - serie de tempo
#     s1 - serie de heave
#     s2 - serie de deslocamento horizontal 1
#     s3 - serie de deslocamento horizontal 2
#     Fs - frequencia de amostragem (Hz)
#     NFFT - tamanho do segmento para o cálculo da FFT
#     outpus:
#     cc - dataframe com os espectros
#     pp - dataframe com parametros de onda
#     tt - dicionario com os parametros das ondas individuais
#     """

#     # numero de overlap p welch. se for igual a zero, fica com
#     # o espectro de bartlett
#     # NOVERLAP = 0
#     NOVERLAP = int(NFFT/2.0)

#     # espectro de welch
#     f, c11 = welch(x=s1, fs=Fs, window='hann', nperseg=NFFT,
#                           noverlap=NOVERLAP, nfft=NFFT, detrend='constant',
#                           return_onesided=True, scaling='density', axis=-1, average='mean')

#     if len(s2) > 0:

#         f, c22 = welch(x=s2, fs=Fs, window='hann', nperseg=NFFT,
#                               noverlap=NOVERLAP, nfft=NFFT, detrend='constant',
#                               return_onesided=True, scaling='density', axis=-1, average='mean')

#         f, c33 = welch(x=s3, fs=Fs, window='hann', nperseg=NFFT,
#                               noverlap=NOVERLAP, nfft=NFFT, detrend='constant',
#                               return_onesided=True, scaling='density', axis=-1, average='mean')

#         # espectros cruzados
#         f, P12 = csd(x=s1, y=s2, fs=Fs, window='hann', nperseg=NFFT,
#                             noverlap=NOVERLAP, nfft=NFFT, detrend='constant',
#                             return_onesided=True, scaling='density', axis=-1, average='mean')
#         c12, q12 = np.real(P12), np.imag(P12)

#         f, P13 = csd(x=s1, y=s3, fs=Fs, window='hann', nperseg=NFFT,
#                             noverlap=NOVERLAP, nfft=NFFT, detrend='constant',
#                             return_onesided=True, scaling='density', axis=-1, average='mean')
#         c13, q13 = np.real(P13), np.imag(P13)

#         f, P23 = csd(x=s2, y=s3, fs=Fs, window='hann', nperseg=NFFT,
#                             noverlap=NOVERLAP, nfft=NFFT, detrend='constant',
#                             return_onesided=True, scaling='density', axis=-1, average='mean')
#         c23, q23 = np.real(P23), np.imag(P23)

#         # parametros em dataframe
#         cc = {'f': f, 'c11': c11, 'c22': c22, 'c33': c33, 'P12': P12, 'c12': c12, 'q12': q12,
#               'P13': P13, 'c13': c13, 'q13': q13, 'P23': P23, 'c23': c23, 'q23': q23}
#         cc = pd.DataFrame(cc)
#         cc.set_index('f', inplace=True)

#         # remove a primeira coluna de media
#         cc = cc.iloc[1:,:]

#         # calcula direcao de onda - Long (1980))
#         cc['a1'] = cc.q12 / (cc.c11 * (cc.c22 + cc.c33)) ** 0.5
#         cc['b1'] = cc.q13 / (cc.c11 * (cc.c22 + cc.c33)) ** 0.5
#         cc['a2'] = (cc.c22 - cc.c33) / (cc.c22 + cc.c33)
#         cc['b2'] = 2 * cc.c23 / (cc.c22 + cc.c33)
#         cc['theta'] = np.rad2deg(np.arctan2(cc.b1, cc.a1))
#         cc.loc[cc.theta>360, 'theta'] = cc.loc[cc.theta>360, 'theta'] - 360
#         cc.loc[cc.theta<0, 'theta'] = cc.loc[cc.theta<0, 'theta'] + 360

#         # calculo do espalhamento angular (ref: manual waverider, pg. 36)
#         m1_spr = np.sqrt( (cc.a1**2) + (cc.b1**2) )
#         cc['spr'] = np.rad2deg(np.sqrt(2.0 * (1 - m1_spr)))

#     else:
#         cc = {'f': f, 'c11': c11}
#         cc = pd.DataFrame(cc)
#         cc.set_index('f', inplace=True)

#     # dataframe com os parametros
#     pp = {}

#     # intervalo do vetor de frequencia
#     dff = np.diff(cc.index)[0]

#     # calcula momentos e parametros espectrais com o WAFO

#     # calculo do espectro
#     # S = wsm.SpecData1D(data=cc.c11.values,
#     #                 args=cc.index.values,
#     #                 type='freq',
#     #                 freqtype='f',
#     #                 tr=None,
#     #                 h=np.inf,
#     #                 v=None,
#     #                 norm=False,
#     #                 date=None)

#     # # # parametros de onda
#     # [ch, R, txt] = S.characteristic(fact=np.arange(0, 15), T=len(s1), g=9.81)

#     # # # momentos espectrais
#     # mom, mom_txt = S.moment(nr=4, even=False, j=0)

#     # pp = pd.concat([pd.DataFrame(ch, index=[i+'_wafo' for i in txt]),
#     #                 pd.DataFrame(mom, index=[i+'_wafo' for i in mom_txt])],
#     #                 axis=0).T
    

#     # # calculo do periodo de pico
#     # pp['tp2_wafo'] = 1/cc.index[cc.c11 == cc.c11.max()][0]

#     # calcula os momentos espectrais
#     # pp['m-1'] = np.sum(cc.index*-1 * cc.c11) * dff
#     pp['m0'] = np.sum(cc.index**0 * cc.c11) * dff
#     pp['m1'] = np.sum(cc.index**1 * cc.c11) * dff
#     pp['m2'] = np.sum(cc.index**2 * cc.c11) * dff
#     pp['m3'] = np.sum(cc.index**3 * cc.c11) * dff
#     pp['m4'] = np.sum(cc.index**4 * cc.c11) * dff

#     # utiliza os momentos espectrais do WAFO
#     # pp['m0'] = pp['m0_wafo']
#     # pp['m1'] = pp['m0t_wafo']
#     # pp['m2'] = pp['m0tt_wafo']
#     # pp['m3'] = pp['m0ttt_wafo']
#     # pp['m4'] = pp['m0tttt_wafo']

#     #calculo da altura significativa
#     pp['hm0'] = 4.01 * np.sqrt(pp['m0'])

#     # frequencia de pico
#     pp['fp'] = cc.index[cc.c11 == cc.c11.max()].values[0]

#     # periodo de pico
#     pp['tp'] = 1 / pp['fp']

#     if len(s2) > 0:
#         # direcao de pico
#         pp['dp'] = cc.theta[cc.c11 == cc.c11.max()].values[0]

#         # spread de pico
#         pp['sprp'] = cc.spr[cc.c11 == cc.c11.max()].values[0]

#         # acha direcao para frequencias especificas
#         # absolute_differences = np.abs(cc.index - 0.6)
#         # index = np.argmin(absolute_differences)
#         # pp['dir_f06hz'] = cc.theta.values[index]

#     # mean wave period (Tz) also denoted Tm02
#     pp['tm02'] = np.sqrt(pp['m0']/pp['m2'])

#     # mean period
#     pp['tm'] = pp['m0'] / pp['m1']

#     # # mean spectral frequency
#     # mean_spec_freq = m1 / m0

#     # # mean spectral period
#     # mean_spec_period = 1.0 / mean_spec_freq

#     # # average zero-up-crossing frequency
#     # mean_zup_freq = np.sqrt(m2 / m0)

#     # # average zero-up-crossing period
#     # mean_zup_period = 1.0 / mean_zup_freq

#     # spectral bandwidth (Cartwright & Longuet-Higgins, 1956)
#     # pp['e'] = np.sqrt(1 - (pp['m2'] ** 2 / (pp['m0'] * pp['m2'])))

#     # spectral bandwidth (Longuet-Higgins, 1975)
#     pp['v'] = np.sqrt((pp['m0'] * pp['m2']) / (pp['m1'] ** 2) - 1)

#     # Peakdness parameter (Goda, 1970)
#     pp['Qp'] = (2.0 / pp['m0']**2) * sum(cc.index * cc.c11**2) * dff

#     # calcula BFI
#     pp['L0'] = 1.56 * pp['tp'] ** 2.0
#     pp['k0'] = 2 * np.pi / pp['L0']
#     pp['BFI'] = pp['k0'] * pp['m0'] ** (1.0/2.0) * pp['Qp'] * np.sqrt(2 * np.pi)                        

#     if grupos == True:
#         # calcula o envelope de hilbert
#         serie_hilbert = envelope_hilbert(s1, Fs)

#         f_hilbert, c11_hilbert = welch(x=serie_hilbert, fs=Fs, window='hann', nperseg=NFFT,
#                             noverlap=NOVERLAP, nfft=NFFT, detrend='constant',
#                             return_onesided=True, scaling='density', axis=-1, average='mean')

#         pp['m0_hilbert'] = np.sum(f_hilbert**0 * c11_hilbert) * (f_hilbert[3] - f_hilbert[2])

#         # calcula parametro de agrupamento (GF)
#         # pp['GF_hilbert'] = np.sqrt(pp['m0_hilbert']) / pp['m0']
#         # pp['GF_hilbert'] = pp['m0_hilbert'] / pp['m0']
#         # pp['GF_hilbert'] = np.sqrt(pp['m0']) / pp['m0_hilbert']
#         # pp['GF_hilbert'] = np.sqrt(pp['m0_hilbert']) / pp['m0']
#         pp['GF_hilbert'] = pp['m0_hilbert'] / pp['m0']
#         # pp['GF_hilbert1'] = np.std(serie_hilbert) / np.mean(serie_hilbert)

#     # retorna para dicionario
#     # pp = pp.to_dict('index')[0]


#     # Beijamin Fair Index
#     # L = 1.56 * pp['tp'] ** 2.0
#     # k0 = 2 * np.pi / L
#     # pp['BFI'] = k0 * pp['m0'] ** (1.0/2) * pp['Qp'] * np.sqrt(2 * np.pi)

#     #Espalhamento direcional
#     #formula do sigma1 do livro Tucker&Pitt(2001) "Waves in Ocean Engineering" pags 196-198
#     # c1 = np.sqrt(a1 ** 2 + b1 ** 2)
#     # c2 = np.sqrt(a2 ** 2 + b2 ** 2)    
#     # s1 = c1 / (1 - c1)
#     # s2 = (1 + 3 * c2 + np.sqrt(1 + 14 * c2 + c2 ** 2)) / (2 * (1 - c2))
#     # sigma1 = np.sqrt(2 - 2 * c1) * 180 / np.pi
#     # sigma2 = np.sqrt((1 - c2) / 2) * 180 / np.pi

#     # # acha o espalhamento angular da frequencia de pico
#     # sigma1p = np.real(sigma1[ind])[0]
#     # sigma2p = np.real(sigma2[ind])[0]
    
#     if tempo == True:
#         # parametros de onda no dominio do tempo
#         eta = s1 - np.mean(s1)

#         # calcula posicoes do eixo x do cruzamento
#         # do zero ascendente
#         pos_x = []
#         for i in range(len(eta)-1):
#             if ((eta[i] < 0) & (eta[i+1] > 0)):
#                 x1 = t[i]
#                 x2 = t[i+1]
#                 y1 = eta[i]
#                 y2 = eta[i+1]
#                 a = (y2 - y1) / (x2 - x1)
#                 b = y1 - a * x1
#                 x = -b / a
#                 pos_x.append([int(i), x])
#         pos_x = np.array(pos_x)

#         #calcula ondas individuas
#         Cr, Ca, H, T = [], [], [], []
#         for i in range(len(pos_x)-1):
#             onda = eta[int(pos_x[i, 0]):int(pos_x[i+1, 0]+1)]
#             cr = np.max(onda)
#             Cr.append(cr)
#             ca = np.min(onda)
#             Ca.append(ca)
#             H.append(cr + np.abs(ca))
#             T.append(pos_x[i+1, 1] - pos_x[i, 1])


#         # #criando os vetores H(altura),Cr(crista),Ca(cavado),T (periodo)
#         # Cr, Ca, H, T = [], [], [], []

#         # #acha os indices que cruzam o zero
#         # z = np.where(np.diff(np.sign(eta)))[0]

#         # #zeros ascendentes e descendentes
#         # zas=z[0::2]
#         # zde=z[1::2]

#         # #calcula ondas individuas
#         # for i in range(len(zas)-1):
#         #     onda = eta[zas[i]:(zas[i+1])+1]
#         #     cr = np.max(onda)
#         #     Cr.append(cr)
#         #     ca = np.min(onda)
#         #     Ca.append(ca)
#         #     H.append(cr + np.abs(ca))
#         #     T.append(t[zas[i+1]] - t[zas[i]])
#         # # H = np.array(H)
#         # # T = np.array(T)

#         alt_per = np.array([H, T]).T

#         # matriz de altura e periodo em ordem
#         # crescente de altuas
#         alt_per_sort = alt_per[np.argsort(alt_per[:,0]), :]

#         #coloca as alturas em ordem crescente
#         Hss = alt_per_sort[:,0] # altura
#         Tss = alt_per_sort[:,1] # periodo relacionado a altura

#         #calcula a altura e periodo significativo (Hs e Ts)
#         div = int(len(alt_per) / 3.0)
#         pp['hs'] = np.mean(Hss[-div:])
#         pp['ts'] = np.mean(Tss[-div:])

#         #calcula a altura das 1/10 maiores (H 1/10)
#         div1 = int(len(Hss) / 10.0)
#         pp['h10'] = np.mean(Hss[-div1:]) #altura da media das um decimo maiores

#         #altura maxima
#         pp['hmax'] = np.max(H)    

#         #periodo medio
#         pp['tz'] = np.mean(T)

#         #calcula periodo associado a altura maxima
#         ind = np.where(alt_per[:,0] == pp['hmax'])[0][0]
#         pp['thmax'] = T[ind]

#         # relacao de Hmax por Hs
#         pp['hmaxhs'] = pp['hmax'] / pp['hs']

#     # cria dataframe com os parametros
#     pp = pd.DataFrame(pp, index=[0])
#     tt = pd.DataFrame()

#     if tempo == True:
#         # dicionario com os parametros das ondas individuais
#         tt.index.name = 'n'
#         tt['H'] = H
#         tt['T'] = T

#     if prob == True:
#         # calcula assimetria e curtose para a serie de elevacao,
#         # alturas e periodos
#         pp['kurt_n'] = kurtosis(s1)
#         pp['skew_n'] = skew(s1)
#         pp['kurt_H'] = kurtosis(tt['H'])
#         pp['skew_H'] = skew(tt['H'])
#         pp['kurt_T'] = kurtosis(tt['T'])
#         pp['skew_T'] = skew(tt['T'])

#     if grupos == True:
#         # calculo do fator de correlacao entre alturas e periodos
#         # individuais
#         tt['H1'] = tt['H'].shift(periods=1)
#         tt['T1'] = tt['T'].shift(periods=1)
#         pp['corr_H'] = tt[['H','H1']].corr().iloc[0,1]
#         pp['corr_T'] = tt[['T','T1']].corr().iloc[0,1]

#         # calcula o numero de runs
#         # - numero de alturas individuais acima do hs dividido
#         # pelo numero de ondas individuais no registro
#         pp['nruns'] = len(tt.loc[tt.H > pp.hs.values[0]]) / len(tt)

#     return cc, pp, tt
