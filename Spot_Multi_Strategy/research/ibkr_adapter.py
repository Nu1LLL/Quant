"""Optional read-only IBKR paper-account and historical-bar adapter.

`ibapi` is imported only when a live probe is requested. Importing the research
package therefore does not require TWS or the optional broker dependency.
No order method is exposed by this module.
"""

import math
import threading


def audit_bars(bars):
    if not bars:
        return {"passed": False, "reason": "no bars", "count": 0}
    times = [int(bar["date"]) for bar in bars]
    invalid = 0
    for bar in bars:
        open_, high, low, close = (
            float(bar[key]) for key in ("open", "high", "low", "close")
        )
        valid = (
            all(math.isfinite(value) and value > 0 for value in (open_, high, low, close))
            and low <= min(open_, close) <= max(open_, close) <= high
        )
        invalid += not valid
    ordered = all(later > earlier for earlier, later in zip(times, times[1:]))
    return {
        "passed": invalid == 0 and ordered,
        "count": len(bars),
        "invalid_ohlc": invalid,
        "strictly_increasing": ordered,
        "gaps_over_60s": sum(
            later - earlier > 60 for earlier, later in zip(times, times[1:])
        ),
        "start_epoch": min(times),
        "end_epoch": max(times),
        "limitation": "gaps require session-calendar review",
    }


def probe_default_markets(host="127.0.0.1", port=7497, client_id=92241):
    """Read one day of AAPL and EUR/USD minute bars from a DU paper account."""
    try:
        from ibapi.client import EClient
        from ibapi.contract import Contract
        from ibapi.wrapper import EWrapper
    except ImportError as exc:
        raise RuntimeError("install the optional ibapi package to use IBKR probes") from exc

    class Probe(EWrapper, EClient):
        def __init__(self):
            EClient.__init__(self, self)
            self.ready = threading.Event()
            self.accounts_ready = threading.Event()
            self.paper = False
            self.bars = {}
            self.done = {}
            self.errors = []

        def nextValidId(self, orderId):
            self.ready.set()

        def managedAccounts(self, accountsList):
            accounts = [value for value in accountsList.split(",") if value]
            self.paper = bool(accounts) and all(value.startswith("DU") for value in accounts)
            self.accounts_ready.set()

        def historicalData(self, reqId, bar):
            self.bars[reqId].append(
                {key: getattr(bar, key) for key in ("date", "open", "high", "low", "close")}
            )

        def historicalDataEnd(self, reqId, start, end):
            self.done[reqId].set()

        def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
            self.errors.append({"code": errorCode, "message": errorString})

    app = Probe()
    report = {"purpose": "read-only data access; not strategy performance", "markets": {}}
    try:
        app.connect(host, port, clientId=client_id)
        threading.Thread(target=app.run, daemon=True).start()
        if not app.ready.wait(10) or not app.accounts_ready.wait(5) or not app.paper:
            raise RuntimeError("IBKR DU paper-account handshake not confirmed")
        requests = (
            (9201, "AAPL", "STK", "SMART", "TRADES"),
            (9202, "EUR", "CASH", "IDEALPRO", "MIDPOINT"),
        )
        for request_id, symbol, security_type, venue, source in requests:
            contract = Contract()
            contract.symbol = symbol
            contract.secType = security_type
            contract.exchange = venue
            contract.currency = "USD"
            if security_type == "STK":
                contract.primaryExchange = "NASDAQ"
            app.bars[request_id] = []
            app.done[request_id] = threading.Event()
            app.reqHistoricalData(
                request_id, contract, "", "1 D", "1 min", source,
                1 if security_type == "STK" else 0, 2, False, [],
            )
            completed = app.done[request_id].wait(20)
            if not completed:
                app.cancelHistoricalData(request_id)
            report["markets"][symbol] = {
                **audit_bars(app.bars[request_id]),
                "completed": completed,
                "source": source,
            }
    finally:
        app.disconnect()
    report["errors"] = app.errors
    return report
