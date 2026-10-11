Paired comparison on 600 common scenarios; metric `route_success`.

| Condition | Method | Run | Rate (95% CI) | Paired diff vs reference | McNemar p |
|---|---|---|---|---|---|
| displace | waypoint (dev-comparison) | `20261008T184255Z-00e829` | 59.5% (52.6–66.1), n=200 | reference |  |
| displace | route (dev-comparison-2) | `20261008T194619Z-34a1cd` | 87.0% (81.6–91.0), n=200 | +27.5 pts (62 won, 7 lost) | 4.1e-12 |
| displace | mpc (dev-comparison-2) | `20261008T195210Z-5780e2` | 90.0% (85.1–93.4), n=200 | +30.5 pts (67 won, 6 lost) | 3.95e-14 |
| none | waypoint (dev-comparison) | `20261008T184255Z-00e829` | 93.0% (88.6–95.8), n=200 | reference |  |
| none | route (dev-comparison-2) | `20261008T194619Z-34a1cd` | 98.0% (95.0–99.2), n=200 | +5.0 pts (14 won, 4 lost) | 0.0309 |
| none | mpc (dev-comparison-2) | `20261008T195210Z-5780e2` | 94.5% (90.4–96.9), n=200 | +1.5 pts (13 won, 10 lost) | 0.678 |
| shove | waypoint (dev-comparison) | `20261008T184255Z-00e829` | 82.5% (76.6–87.1), n=200 | reference |  |
| shove | route (dev-comparison-2) | `20261008T194619Z-34a1cd` | 95.5% (91.7–97.6), n=200 | +13.0 pts (32 won, 6 lost) | 2.43e-05 |
| shove | mpc (dev-comparison-2) | `20261008T195210Z-5780e2` | 94.5% (90.4–96.9), n=200 | +12.0 pts (32 won, 8 lost) | 0.000182 |
| all | waypoint (dev-comparison) | `20261008T184255Z-00e829` | 78.3% (74.9–81.4), n=600 | reference |  |
| all | route (dev-comparison-2) | `20261008T194619Z-34a1cd` | 93.5% (91.2–95.2), n=600 | +15.2 pts (108 won, 17 lost) | 2.22e-17 |
| all | mpc (dev-comparison-2) | `20261008T195210Z-5780e2` | 93.0% (90.7–94.8), n=600 | +14.7 pts (112 won, 24 lost) | 8.68e-15 |
