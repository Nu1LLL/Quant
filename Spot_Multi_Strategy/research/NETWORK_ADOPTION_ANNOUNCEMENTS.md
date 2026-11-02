# Network-adoption study: official futures-delisting knowledge times

This is a source-time manifest for the separately committed `cde1897`
preregistration, not a strategy result. These are USDⓈ-M perpetual-contract
notices, not spot-only delistings. Each symbol stays in the original seven-name
cohort and becomes ineligible after its first public contract-delisting notice;
subsequent relisting does not reintroduce it. The simulator must use the
strictly later tradable 4-hour open for a held-contract exit.

| Contract | First notice (UTC) | Announced stop/settlement (UTC) | Official source |
| --- | --- | --- | --- |
| `CVCUSDT` | 2022-11-28 03:30 | 2022-11-29 09:00 | [Binance contract notice](https://www.binance.com/en-IN/support/announcement/detail/3188e26f14a1464b91487abad452c04f) |
| `RENUSDT` | 2024-11-26 08:30 | 2024-12-03 09:00 | [Binance REN notice](https://www.binance.com/en/support/announcement/detail/131b75b63a71490393034f36de70c92c) |
| `OMGUSDT` | 2024-12-06 03:00 | originally 2024-12-16 09:00, [later postponed](https://www.binance.com/en/support/announcement/detail/1e43c128806544018d3cc537a8add3f6) | [Binance contract notice](https://www.binance.com/en-PH/support/announcement/detail/27eac260521b4f40ab331e1a345887a7) |

Binance's [2024-06-03 OMG spot-delisting notice](https://www.binance.com/en/support/announcement/detail/f65faeeefe07417a8b6dd2900ba8da7e)
explicitly said `OMGUSDT` perpetual trading would continue, so it is **not**
used as the futures notice. The CVC contract [relaunched in May 2025](https://www.binance.com/pt/support/announcement/detail/bbe829426ad844b9b9dac62f878d1e67),
but the frozen no-reentry rule still excludes it after November 2022.

No post-2020 market prices, funding rates, on-chain signal scores, or strategy
results were read to create this notice manifest. Missing archive months for
other contracts, conflicting notice timestamps or an unknown trading halt must
be resolved with contemporaneous official sources or fail the data gate; the
manifest is not a license to infer a delisting from future archive absence.
