Paired comparison on 600 common scenarios; metric `route_success`.

| Condition | Method | Run | Rate (95% CI) | Paired diff vs reference | McNemar p |
|---|---|---|---|---|---|
| displace | waypoint (dev-comparison) | `20261008T184255Z-00e829` | 59.5% (52.6–66.1), n=200 | reference |  |
| displace | route (dev-comparison) | `20261008T184834Z-432abb` | 87.0% (81.6–91.0), n=200 | +27.5 pts (62 won, 7 lost) | 4.1e-12 |
| none | waypoint (dev-comparison) | `20261008T184255Z-00e829` | 93.0% (88.6–95.8), n=200 | reference |  |
| none | route (dev-comparison) | `20261008T184834Z-432abb` | 98.0% (95.0–99.2), n=200 | +5.0 pts (14 won, 4 lost) | 0.0309 |
| shove | waypoint (dev-comparison) | `20261008T184255Z-00e829` | 82.5% (76.6–87.1), n=200 | reference |  |
| shove | route (dev-comparison) | `20261008T184834Z-432abb` | 95.5% (91.7–97.6), n=200 | +13.0 pts (32 won, 6 lost) | 2.43e-05 |
| all | waypoint (dev-comparison) | `20261008T184255Z-00e829` | 78.3% (74.9–81.4), n=600 | reference |  |
| all | route (dev-comparison) | `20261008T184834Z-432abb` | 93.5% (91.2–95.2), n=600 | +15.2 pts (108 won, 17 lost) | 2.22e-17 |
