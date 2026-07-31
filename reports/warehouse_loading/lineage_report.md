# Data Lineage Report

| Stage | Table/File | Row Count | Notes |
|---|---|---|---|
| Raw | hotel_bookings.csv | 119390 | - |
| Cleaned | hotel_bookings_clean.parquet | 87395 | after dedupe + negative-adr drop |
| Warehouse Files | dim_date | 806 | local parquet, always written |
| Warehouse Files | dim_hotel | 2 | local parquet, always written |
| Warehouse Files | dim_room_type | 4 | local parquet, always written |
| Warehouse Files | dim_guest | 5672 | local parquet, always written |
| Warehouse Files | fact_booking | 87395 | local parquet, always written |
| Database | (all tables) | — | not loaded to DB (--write-db not passed) |
