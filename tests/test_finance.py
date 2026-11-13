"""Unit tests for the official money-saved metric and its breakdown.

The hand-worked example (interest 20 %):

    #  actual  pred  loan     outcome                     money effect vs approve-all
    1  default  1    10,000   caught defaulter           +10,000
    2  default  0    20,000   missed defaulter (approved)       0  (exposure 20,000, not subtracted)
    3  repaid   1    50,000   good customer rejected     -10,000  (0.20 x 50,000)
    4  repaid   0    30,000   good customer approved           0
    5  default  1     5,000   caught defaulter            +5,000
                                                          -------
                                       money saved =       5,000
"""
import numpy as np
import pytest

from quickloan import finance as F

Y = np.array([1, 1, 0, 0, 1])
P = np.array([1, 0, 1, 0, 1])
L = np.array([10_000, 20_000, 50_000, 30_000, 5_000], dtype=float)


def test_hand_worked_example():
    assert F.money_saved(Y, P, L, 0.20) == pytest.approx(5_000)


def test_caught_defaulters_contribute_full_principal():
    assert F.money_saved([1, 1], [1, 1], [7_000, 3_000]) == pytest.approx(10_000)


def test_wrongly_rejected_cost_only_interest():
    assert F.money_saved([0], [1], [40_000], 0.20) == pytest.approx(-8_000)
    assert F.money_saved([0], [1], [40_000], 0.05) == pytest.approx(-2_000)


def test_missed_defaulters_and_correct_approvals_are_neutral():
    assert F.money_saved([1, 0], [0, 0], [99_000, 12_000]) == 0


def test_approve_everyone_is_zero():
    assert F.money_saved(Y, np.zeros_like(Y), L) == 0
    assert F.baselines(Y, L)["approve_all"] == 0


def test_reject_everyone():
    expected = L[Y == 1].sum() - 0.2 * L[Y == 0].sum()            # 35,000 - 16,000 = 19,000
    assert F.money_saved(Y, np.ones_like(Y), L) == pytest.approx(expected)
    assert F.baselines(Y, L)["reject_all"] == pytest.approx(19_000)


def test_negative_result_is_preserved():
    assert F.money_saved([0, 0, 1], [1, 1, 0], [100_000, 100_000, 1_000], 0.5) == pytest.approx(-100_000)


def test_breakdown_identity_and_no_double_count():
    b = F.breakdown(Y, P, L, 0.20)
    assert b["loans_saved"] == 15_000
    assert b["interest_lost"] == pytest.approx(10_000)
    assert b["net_money_saved"] == pytest.approx(5_000)
    assert b["missed_default_principal"] == 20_000
    assert (b["n_tp"], b["n_fp"], b["n_fn"], b["n_tn"]) == (2, 1, 1, 1)
    # missed exposure is reported separately and NOT subtracted from net
    assert b["net_money_saved"] == pytest.approx(b["loans_saved"] - b["interest_lost"])


@pytest.mark.parametrize("interest", np.round(np.arange(0.05, 0.501, 0.01), 2))
def test_every_slider_value_recalculates(interest):
    assert F.money_saved(Y, P, L, interest) == pytest.approx(15_000 - interest * 50_000)


def test_money_saved_equals_pnl_difference():
    for i in (0.05, 0.2, 0.5):
        diff = F.portfolio_pnl(Y, P, L, i) - F.portfolio_pnl(Y, np.zeros_like(Y), L, i)
        assert F.money_saved(Y, P, L, i) == pytest.approx(diff)


def test_accepts_pandas_and_lists():
    import pandas as pd
    assert F.money_saved(pd.Series(Y), list(P), pd.Series(L)) == pytest.approx(5_000)


def test_best_threshold_picks_max():
    scores = np.array([0.9, 0.3, 0.6, 0.1, 0.8])
    t, v = F.best_threshold(Y, scores, L, 0.20)
    assert v == pytest.approx(max(F.money_saved(Y, (scores >= g).astype(int), L) for g in np.arange(0.05, 0.951, 0.01)))
