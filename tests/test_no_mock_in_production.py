import asyncio

from fastapi import HTTPException

from main import get_regional_segmentation, model_store


def test_no_mock_in_production():
    """The API should return 503 instead of fabricating regional data."""
    original = model_store.regional_df
    model_store.regional_df = None
    try:
        try:
            asyncio.run(get_regional_segmentation())
        except HTTPException as exc:
            assert exc.status_code == 503
            assert "regional_clusters.csv" in str(exc.detail)
        else:
            raise AssertionError("Expected regional segmentation to fail without real data.")
    finally:
        model_store.regional_df = original
