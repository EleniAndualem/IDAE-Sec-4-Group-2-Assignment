# Data Understanding Report

## Dataset Overview
- **Rows:** 20,640
- **Columns:** 10
- **Duplicate rows:** 0

## Columns

| Column | Data Type | Missing Values |
|--------|-----------|----------------|
| longitude | float64 | 0 |
| latitude | float64 | 0 |
| housing_median_age | float64 | 0 |
| total_rooms | float64 | 0 |
| total_bedrooms | float64 | 207 |
| population | float64 | 0 |
| households | float64 | 0 |
| median_income | float64 | 0 |
| median_house_value | float64 | 0 |
| ocean_proximity | str | 0 |

## Numeric Statistics

| Stat | longitude | latitude | housing_median_age | total_rooms | total_bedrooms | population | households | median_income | median_house_value |
|------|---|---|---|---|---|---|---|---|---|
| count | 20,640.0000 | 20,640.0000 | 20,640.0000 | 20,640.0000 | 20,433.0000 | 20,640.0000 | 20,640.0000 | 20,640.0000 | 20,640.0000 |
| mean | -119.5697 | 35.6319 | 28.6395 | 2,635.7631 | 537.8706 | 1,425.4767 | 499.5397 | 3.8707 | 206,855.8169 |
| std | 2.0035 | 2.1360 | 12.5856 | 2,181.6153 | 421.3851 | 1,132.4621 | 382.3298 | 1.8998 | 115,395.6159 |
| min | -124.3500 | 32.5400 | 1.0000 | 2.0000 | 1.0000 | 3.0000 | 1.0000 | 0.4999 | 14,999.0000 |
| 25% | -121.8000 | 33.9300 | 18.0000 | 1,447.7500 | 296.0000 | 787.0000 | 280.0000 | 2.5634 | 119,600.0000 |
| 50% | -118.4900 | 34.2600 | 29.0000 | 2,127.0000 | 435.0000 | 1,166.0000 | 409.0000 | 3.5348 | 179,700.0000 |
| 75% | -118.0100 | 37.7100 | 37.0000 | 3,148.0000 | 647.0000 | 1,725.0000 | 605.0000 | 4.7432 | 264,725.0000 |
| max | -114.3100 | 41.9500 | 52.0000 | 39,320.0000 | 6,445.0000 | 35,682.0000 | 6,082.0000 | 15.0001 | 500,001.0000 |

## Categorical Statistics

### ocean_proximity

| Category | Count |
|----------|-------|
| <1H OCEAN | 9,136 |
| INLAND | 6,551 |
| NEAR OCEAN | 2,658 |
| NEAR BAY | 2,290 |
| ISLAND | 5 |
