import numpy as np
from rating_models import IterativeOLR, IterativeMargin, IterativePoisson
from scipy.special import expit as logistic
from numpy.testing import assert_allclose


def log_lik_elo(r1, r2, res):
    r_diff = r1 - r2
    return res * np.log(logistic(-r_diff)) + (1 - res) * np.log((1 - logistic(-r_diff)))


def grad_log_lik_elo_exact(r1, r2, res):
    r_diff = r1 - r2
    p1 = logistic(-r_diff)
    grad = p1 - res
    return -grad, grad


def grad_log_lik_elo_approx(r1, r2, res, eps=1e-8):
    dr1 = (log_lik_elo(r1, r2 + eps, res) - log_lik_elo(r1, r2 - eps, res)) / (2 * eps)
    dr2 = (log_lik_elo(r1 + eps, r2, res) - log_lik_elo(r1 - eps, r2, res)) / (2 * eps)
    return dr1, dr2


def compute_prob(r1, r2, c, h):
    d = r1 - r2 + h
    return logistic(-c + d), logistic(c + d) - logistic(-c + d), 1 - logistic(c + d)


def log_lik_olr(r1, r2, res, c, h):
    prob = compute_prob(r1, r2, c, h)
    return np.log(prob[res])


def grad_log_lik_olr_exact(r1, r2, res, c, h):
    prob = compute_prob(r1, r2, c, h)
    grad = IterativeOLR.get_update(res, prob)
    return -grad, grad


def grad_log_lik_olr_approx(r1, r2, res, c, h, eps=1e-8):
    dr1 = (log_lik_olr(r1, r2 + eps, res, c=c, h=h) - log_lik_olr(r1, r2 - eps, res, c=c, h=h)) / (2 * eps)
    dr2 = (log_lik_olr(r1 + eps, r2, res, c=c, h=h) - log_lik_olr(r1 - eps, r2, res, c=c, h=h)) / (2 * eps)
    return dr1, dr2


def log_lik_pois1(r1, r2, c, h, g1, g2):
    xb1 = c + r1 - r2 + h
    xb2 = c + r2 - r1
    return g1 * xb1 - np.exp(xb1) + g2 * xb2 - np.exp(xb2)


def grad_log_lik_pois1_exact(r1, r2, c, h, g1, g2):
    mu1 = np.exp(c + r1 - r2 + h)
    mu2 = np.exp(c + r2 - r1)
    margin_diff = IterativeMargin.get_update(g1, g2, mu1, mu2)
    return -margin_diff, margin_diff


def grad_log_lik_pois1_approx(r1, r2, c, h, g1, g2, eps=1e-8):
    dr1 = (log_lik_pois1(r1, r2 + eps, c, h, g1, g2) - log_lik_pois1(r1, r2 - eps, c, h, g1, g2)) / (2 * eps)
    dr2 = (log_lik_pois1(r1 + eps, r2, c, h, g1, g2) - log_lik_pois1(r1 - eps, r2, c, h, g1, g2)) / (2 * eps)
    return dr1, dr2


def log_lik_pois2(a1, a2, d1, d2, c, h, g1, g2):
    xb1 = c + a1 - d2 + h
    xb2 = c + a2 - d1
    return g1 * xb1 - np.exp(xb1) + g2 * xb2 - np.exp(xb2)


def grad_log_lik_pois2_exact(a1, a2, d1, d2, c, h, g1, g2):
    mu1 = np.exp(c + a1 - d2 + h)
    mu2 = np.exp(c + a2 - d1)
    update1, update2 = IterativePoisson.get_update(g1, g2, mu1, mu2)
    return update1, -update2, update2, -update1


def grad_log_lik_pois2_approx(a1, a2, d1, d2, c, h, g1, g2, eps=1e-8):
    da1 = (log_lik_pois2(a1 + eps, a2, d1, d2, c, h, g1, g2) - log_lik_pois2(a1 - eps, a2, d1, d2, c, h, g1, g2)) / (2 * eps)
    dd1 = (log_lik_pois2(a1, a2, d1 + eps, d2, c, h, g1, g2) - log_lik_pois2(a1, a2, d1 - eps, d2, c, h, g1, g2)) / (2 * eps)
    da2 = (log_lik_pois2(a1, a2 + eps, d1, d2, c, h, g1, g2) - log_lik_pois2(a1, a2 - eps, d1, d2, c, h, g1, g2)) / (2 * eps)
    dd2 = (log_lik_pois2(a1, a2, d1, d2 + eps, c, h, g1, g2) - log_lik_pois2(a1, a2, d1, d2 - eps, c, h, g1, g2)) / (2 * eps)
    return da1, dd1, da2, dd2


def test_elo():
    r1, r2 = np.random.normal(size=2)
    for res in (0, 0.5, 1):
        grad1 = grad_log_lik_elo_exact(r1, r2, res)
        grad2 = grad_log_lik_elo_approx(r1, r2, res, eps=1e-8)
        assert_allclose(grad1, grad2, rtol=1e-7, atol=1e-6)


def test_olr():
    h, c = 0.3, 0.6
    r1, r2 = np.random.normal(size=2)
    for res in (0, 1, 2):
        grad1 = grad_log_lik_olr_exact(r1, r2, res=res, c=c, h=h)
        grad2 = grad_log_lik_olr_approx(r1, r2, res=res, c=c, h=h, eps=1e-8)
        assert_allclose(grad1, grad2, rtol=1e-7, atol=1e-7)


def test_poisson1():
    h, c = 0.3, 0.02
    g1, g2 = np.random.poisson(lam=2.7, size=2)
    r1, r2 = np.random.normal(scale=0.25, size=2)
    grad1 = grad_log_lik_pois1_exact(r1, r2, c, h, g1, g2)
    grad2 = grad_log_lik_pois1_approx(r1, r2, c, h, g1, g2)
    assert_allclose(grad1, grad2, rtol=1e-7, atol=1e-7)


def test_poisson2():
    h, c = 0.3, 0.02
    g1, g2 = np.random.poisson(lam=2.7, size=2)
    a1, a2, d1, d2 = np.random.normal(scale=0.25, size=4)
    grad1 = grad_log_lik_pois2_exact(a1, a2, d1, d2, c, h, g1, g2)
    grad2 = grad_log_lik_pois2_approx(a1, a2, d1, d2, c, h, g1, g2, eps=1e-8)
    assert_allclose(grad1, grad2, rtol=1e-7, atol=1e-7)
