"""Configuration for category inference thresholds.

Centralizes and parameterizes category confidence thresholds used during metadata enrichment.
Allows runtime tuning via CLI flags without modifying source code.
"""

from dataclasses import dataclass

from loguru import logger


@dataclass
class CategoryThresholdConfig:
    """Configuration for category inference behavior."""

    strict_threshold: float = 0.75
    """Minimum confidence required to assign a category (vs. marking as unassigned/unknown)"""

    allow_unassigned_categories: bool = False
    """If True, entities with confidence < strict_threshold get category=null; else fallback to inferred category"""

    def validate(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.strict_threshold <= 1.0:
            msg = f"strict_threshold must be between 0.0 and 1.0, got {self.strict_threshold}"
            raise ValueError(msg)

    def __repr__(self) -> str:
        return (
            f"CategoryThresholdConfig("
            f"strict_threshold={self.strict_threshold}, "
            f"allow_unassigned_categories={self.allow_unassigned_categories})"
        )


_DEFAULT_CONFIG = CategoryThresholdConfig()


def get_default_config() -> CategoryThresholdConfig:
    """Get the default category threshold configuration."""
    return CategoryThresholdConfig(
        strict_threshold=0.75,
        allow_unassigned_categories=False,
    )


def create_config(
    strict_threshold: float | None = None,
    allow_unassigned_categories: bool | None = None,
) -> CategoryThresholdConfig:
    """Create a category threshold configuration.

    Args:
        strict_threshold: Override default 0.75 threshold
        allow_unassigned_categories: Override default False

    Returns:
        CategoryThresholdConfig instance (validated)
    """
    config = get_default_config()

    if strict_threshold is not None:
        config.strict_threshold = strict_threshold
    if allow_unassigned_categories is not None:
        config.allow_unassigned_categories = allow_unassigned_categories

    config.validate()
    logger.info(f"Category threshold config: {config}")
    return config


def apply_threshold(confidence: float, config: CategoryThresholdConfig) -> bool:
    """Check if a confidence score passes the configured threshold.

    Args:
        confidence: Confidence score (0.0 - 1.0)
        config: CategoryThresholdConfig instance

    Returns:
        True if confidence meets or exceeds threshold, False otherwise
    """
    return confidence >= config.strict_threshold
