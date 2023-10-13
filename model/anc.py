import numpy as np
def ANC(omega,mean_g,sigma_0):
    mean_omega = mean_g^2
    m = mean_omega.size[0]
    beta = np.zeros(m)
    s = np.zeros(m)
    sigma = np.zeros(m)
    new_omega = 0
    for j in range(m):
        #local clip
        beta[j] = mean_omega[j]/np.sum(mean_omega)
        s[j] = beta[j]* mean_omega[j]
        new_omega = np.min(np.abs(omega[j]), np.abs(s[j]))

        # add adaptive noise
        sigma[j] = sigma_0*beta[j]*np.sqrt(m)*mean_omega[j]
        new_omega += np.random.normal(0, sigma[j], 1)

    return new_omega
