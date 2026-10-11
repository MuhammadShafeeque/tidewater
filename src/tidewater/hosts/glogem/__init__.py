"""GloGEM host: ocean factor tables on GloGEM's model years, and IDL code that reads them.

GloGEM is IDL and cannot import Python. The tables are written here; the IDL
side (``IDL_DIR``) reads them in GloGEM's calving model and also holds the
laws of ``tidewater/spec/SPEC.md`` in IDL.
"""

from .tables import (IDL_DIR, LABEL, SOUTH_REGIONS, forcing_span, start_month,
                     start_months, write_ocean_factor_tables, write_tables)

__all__ = ["IDL_DIR", "LABEL", "SOUTH_REGIONS", "forcing_span", "start_month",
           "start_months", "write_ocean_factor_tables", "write_tables"]
