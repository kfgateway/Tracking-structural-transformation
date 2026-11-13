# Tracking the structural transformation of street greenery under urban renewal

A street-level study of **form, function, and ecosystem services**. This
repository contains the full, reproducible analysis pipeline for a multi-epoch
(1995 / 2005 / 2015) study of how New York City's street-tree population changed
in structure, diversity, and carbon ecosystem services, and whether that change
is attributable to urban renewal.

> Prepared for double-anonymised review: the code contains no author-identifying
> information.

## What it does

```
raw censuses ─▶ canonical panel ─▶ structural indicators ─┐
                                 ├─ carbon ecosystem svcs ─┼─▶ change table ─▶ DID ─▶ figures
            renewal polygons ────▶ treatment table ───────┘        + spatial robustness
```

| Stage | Module | Output |
|------|--------|--------|
| Harmonise censuses | `stgreen.data.census` | `nyc_tree_census_panel.parquet` |
| Structural form | `stgreen.metrics.structure` | `structure_panel_{unit}.parquet` |
| Ecosystem services | `stgreen.metrics.ecoservices` | `ecoservices_panel_{unit}.parquet` |
| Renewal treatment | `stgreen.data.renewal` | `renewal_treatment_{unit}.parquet` |
| Change tracking | `stgreen.change.change_metrics` | `analysis_panel`, `change_table` |
| Causal estimation | `stgreen.analysis.did` / `matching` | DID tables |
| Spatial robustness | `stgreen.analysis.spatial` | Moran's I, spatial DID |
| Figures | `stgreen.viz.figures` | `outputs/figures/*` |

## Quick start

```
make env     # pip install -e ".[dev]"
make all     # full pipeline (uses a synthetic renewal fixture by default)
make test    # unit tests
```

With the real NYC rezoning layer:

```
python -m stgreen.pipeline --renewal data/raw/nyzma.geojson \
    --date-col EffectiveDate --id-col ProjectID
```

See `data/README.md` for data sources, licenses, and documented caveats.

## Methods in brief

- **Form**: abundance, basal area (from SI diameter), DBH size-class/maturity
  distribution, species richness and Shannon/Simpson/Pielou diversity, condition.
- **Ecosystem services**: carbon storage via Jenkins et al. (2003) allometry with
  Nowak & Crane (2002) urban adjustments (0.8 open-grown factor, 0.26 root-shoot,
  0.5 carbon fraction); optional growth-based sequestration.
- **Causal design**: difference-in-differences (treated = units renewed between
  pre and post census) with propensity-score matching, a parallel-trends
  falsification test on the pre-period, and spatial-lag/-error robustness.
