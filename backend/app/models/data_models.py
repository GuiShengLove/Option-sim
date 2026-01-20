"""
Data API Models
================
Pydantic models for data API endpoints.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel


class AssetSummary(BaseModel):
    """Summary information for an option contract."""
    id: str
    type: Literal['call', 'put']
    strike: float
    expiry: str
    close: float
    change: float
    change_percent: float
    iv: float
    volume: int


class CandleData(BaseModel):
    """Single candlestick data point."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int] = None
    avg_iv: Optional[float] = None  # Average Implied Volatility for the day


class AssetListResponse(BaseModel):
    """Response for asset list endpoint."""
    date: str
    count: int
    assets: List[AssetSummary]


class CandleResponse(BaseModel):
    """Response for candle data endpoint."""
    asset_id: str
    symbol: str
    candles: List[CandleData]


class IVSurfacePoint(BaseModel):
    """Single point on IV surface."""
    strike: float
    dte: int
    iv: float


class IVSurfaceResponse(BaseModel):
    """Response for IV surface endpoint."""
    date: str
    strikes: List[float]
    dtes: List[int]
    iv_matrix: List[List[float]]  # [dte_index][strike_index]
    data_quality: Optional[dict] = None  # Data quality metrics


class DateListResponse(BaseModel):
    """Response for available dates endpoint."""
    count: int
    start_date: str
    end_date: str
    dates: List[str]
