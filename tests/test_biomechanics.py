"""Katman 3 — Statik biyomekanik hesaplama motoru testleri.

Biyomekanik motor yalnızca matematiksel hesap yapar; güvenilirlik kararı vermez.
Bu testler hesaplama doğruluğunu ve sınır koşullarını doğrular.
"""

import numpy as np
import pytest

from physioload.geometry import (
    SegmentMass,
    approximate_com_2d,
    bilateral_asymmetry_degrees,
    force_from_mass_2d,
    joint_angle_degrees,
    monte_carlo_angle_uncertainty,
    segment_orientation_degrees,
)


# ---------------------------------------------------------------------------
# Eklem açısı
# ---------------------------------------------------------------------------

def test_right_angle_knee_geometry():
    """Dik açılı diz geometrisi 90 derece vermelidir."""
    angle = joint_angle_degrees((0, 0), (1, 0), (1, 1))
    assert angle == pytest.approx(90.0)


def test_straight_angle_is_180():
    """Düz geometri 180 derece (tam açı) vermelidir."""
    angle = joint_angle_degrees((0, 0), (1, 0), (2, 0))
    assert angle == pytest.approx(180.0)


def test_zero_length_segment_raises():
    """Sıfır uzunluklu segment için hata fırlatılmalıdır."""
    with pytest.raises(ValueError, match="zero-length"):
        joint_angle_degrees((0, 0), (0, 0), (1, 1))


# ---------------------------------------------------------------------------
# Segment yönelimi
# ---------------------------------------------------------------------------

def test_horizontal_segment_is_zero_degrees():
    orient = segment_orientation_degrees((0, 0), (1, 0))
    assert orient == pytest.approx(0.0)


def test_vertical_segment_is_90_degrees():
    orient = segment_orientation_degrees((0, 0), (0, 1))
    assert orient == pytest.approx(90.0)


def test_coincident_points_raise():
    with pytest.raises(ValueError, match="coincident"):
        segment_orientation_degrees((5, 5), (5, 5))


# ---------------------------------------------------------------------------
# Bilateral asimetri
# ---------------------------------------------------------------------------

def test_asymmetry_zero_for_equal_angles():
    assert bilateral_asymmetry_degrees(45.0, 45.0) == pytest.approx(0.0)


def test_asymmetry_absolute_difference():
    assert bilateral_asymmetry_degrees(60.0, 45.0) == pytest.approx(15.0)
    assert bilateral_asymmetry_degrees(45.0, 60.0) == pytest.approx(15.0)


def test_asymmetry_rejects_nan():
    with pytest.raises(ValueError, match="finite"):
        bilateral_asymmetry_degrees(float("nan"), 45.0)


# ---------------------------------------------------------------------------
# Yaklaşık 2B CoM
# ---------------------------------------------------------------------------

def test_com_single_segment_midpoint():
    """Tek segment, varsayılan CoM fraksiyonu 0.5 → orta nokta."""
    result = approximate_com_2d([SegmentMass((0, 0), (2, 0), 1.0, 0.5)])
    assert np.allclose(result, [1.0, 0.0])


def test_com_mass_weighted_average():
    """İki eşit segmentin CoM'u kütle ağırlıklı ortalamadadır."""
    result = approximate_com_2d([
        SegmentMass((0, 0), (2, 0), 0.5, 0.5),
        SegmentMass((0, 2), (2, 2), 0.5, 0.5),
    ])
    assert np.allclose(result, [1.0, 1.0])


def test_com_no_segments_raises():
    with pytest.raises(ValueError):
        approximate_com_2d([])


def test_com_invalid_mass_fraction_raises():
    with pytest.raises(ValueError, match="mass fraction"):
        approximate_com_2d([SegmentMass((0, 0), (1, 0), -0.5, 0.5)])


# ---------------------------------------------------------------------------
# Yerçekimi kuvveti
# ---------------------------------------------------------------------------

def test_force_from_mass_downward():
    """Yerçekimi kuvveti yalnızca Y (aşağı) bileşeni içermelidir."""
    force = force_from_mass_2d(10.0)
    assert force[0] == pytest.approx(0.0)
    assert force[1] == pytest.approx(10.0 * 9.80665)


def test_force_from_mass_rejects_nonpositive():
    with pytest.raises(ValueError):
        force_from_mass_2d(0.0)
    with pytest.raises(ValueError):
        force_from_mass_2d(-5.0)


# ---------------------------------------------------------------------------
# Monte Carlo açı belirsizliği
# ---------------------------------------------------------------------------

def test_monte_carlo_is_deterministic():
    """Aynı tohumla çalıştırılan Monte Carlo aynı sonucu vermelidir."""
    points = {"shoulder": (0, 0), "elbow": (1, 0), "wrist": (1, 1)}
    r1 = monte_carlo_angle_uncertainty(points, ("shoulder", "elbow", "wrist"),
                                       landmark_sigma_px=0.05, samples=200, seed=7)
    r2 = monte_carlo_angle_uncertainty(points, ("shoulder", "elbow", "wrist"),
                                       landmark_sigma_px=0.05, samples=200, seed=7)
    assert r1["mean_deg"] == pytest.approx(r2["mean_deg"])
    assert r1["std_deg"] == pytest.approx(r2["std_deg"])


def test_monte_carlo_std_increases_with_sigma():
    """Büyük bozunum sigma'sı daha yüksek standart sapma üretmelidir."""
    points = {"hip": (0, 0), "knee": (1, 0), "ankle": (1, 1)}
    low = monte_carlo_angle_uncertainty(points, ("hip", "knee", "ankle"),
                                        landmark_sigma_px=0.01, samples=300, seed=0)
    high = monte_carlo_angle_uncertainty(points, ("hip", "knee", "ankle"),
                                         landmark_sigma_px=5.0, samples=300, seed=0)
    assert high["std_deg"] > low["std_deg"]


def test_monte_carlo_missing_landmark_raises():
    points = {"hip": (0, 0), "knee": (1, 0)}
    with pytest.raises(KeyError, match="ankle"):
        monte_carlo_angle_uncertainty(points, ("hip", "knee", "ankle"),
                                       landmark_sigma_px=0.1, samples=100, seed=0)

