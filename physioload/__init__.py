"""PhysioLoad AI — Ölçüm Duyarlı Seçici Biyomekanik Analiz Mimarisi.

Sistem yedi katmanlı araştırma akışını uygular:
  1. Veri ve görüntü yönetimi
  2. Poz kestirimi ve ortak landmark biçimi
  3. Statik biyomekanik hesaplama motoru
  4. Hata yayılımı ve Ölçüm Duyarlılık Profili
  5. Fiziksel geçerlilik denetimi
  6. Ölçüm Bazlı Kalite Kapısı  (özgün çekirdek)
  7. Açıklanabilir biyomekanik raporlama

Referans eklem verileri poz modelinin girdisi değildir; yalnızca hata analizi
ve A0-A4 değerlendirme hattında kullanılır.
"""

from .geometry import (
    SegmentMass,
    approximate_com_2d,
    bilateral_asymmetry_degrees,
    external_moment_2d,
    force_from_mass_2d,
    joint_angle_degrees,
    monte_carlo_angle_uncertainty,
    segment_orientation_degrees,
)
from .physical_validity import (
    PhysicalValidityContext,
    PhysicalValidityResult,
    assess_physical_validity,
)
from .quality import (
    MeasurementStatus,
    OutputDecision,
    QualityDecision,
    QualityFeatures,
    QualityGate,
    fit_calibrator,
)
from .sensitivity import (
    LandmarkImpact,
    error_impact_map,
    measurement_sensitivity_profile,
)
from .data import (
    SampleRecord,
    assert_no_group_leakage,
    dataset_fingerprint,
    group_split,
    write_manifest,
)
from .metrics import (
    classification_auc,
    mae,
    pck,
    risk_coverage,
    rmse,
)

__all__ = [
    # geometry / biomechanics
    "SegmentMass",
    "approximate_com_2d",
    "bilateral_asymmetry_degrees",
    "external_moment_2d",
    "force_from_mass_2d",
    "joint_angle_degrees",
    "monte_carlo_angle_uncertainty",
    "segment_orientation_degrees",
    # physical validity
    "PhysicalValidityContext",
    "PhysicalValidityResult",
    "assess_physical_validity",
    # quality gate
    "MeasurementStatus",
    "OutputDecision",
    "QualityDecision",
    "QualityFeatures",
    "QualityGate",
    "fit_calibrator",
    # sensitivity / error propagation
    "LandmarkImpact",
    "measurement_sensitivity_profile",
    "error_impact_map",
    # data management
    "SampleRecord",
    "assert_no_group_leakage",
    "dataset_fingerprint",
    "group_split",
    "write_manifest",
    # evaluation metrics
    "classification_auc",
    "mae",
    "pck",
    "risk_coverage",
    "rmse",
]

__version__ = "0.1.0"

