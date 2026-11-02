"""Immutable, versioned research ledger backed by SQLite."""

import json
import math
import sqlite3
from typing import Any, Dict, Optional


class ResearchRegistry:
    """Record signal/data provenance, preregistration and final results.

    Application writes are append-only. SQLite triggers also reject UPDATE and
    DELETE so an accidental maintenance query cannot silently rewrite evidence.
    """

    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS signals (
              id TEXT NOT NULL, version TEXT NOT NULL, family TEXT NOT NULL,
              owner TEXT NOT NULL, definition TEXT NOT NULL,
              PRIMARY KEY(id, version));
            CREATE TABLE IF NOT EXISTS observations (
              id INTEGER PRIMARY KEY, signal_id TEXT NOT NULL,
              version TEXT NOT NULL, market TEXT NOT NULL,
              instrument TEXT NOT NULL, event_us INTEGER NOT NULL,
              available_us INTEGER NOT NULL, dataset_version TEXT NOT NULL,
              payload TEXT NOT NULL,
              FOREIGN KEY(signal_id, version) REFERENCES signals(id, version),
              CHECK(available_us >= event_us));
            CREATE INDEX IF NOT EXISTS observations_available
              ON observations(available_us, signal_id, version);
            CREATE TABLE IF NOT EXISTS experiments (
              id TEXT PRIMARY KEY, registered_us INTEGER NOT NULL,
              oos_start_us INTEGER NOT NULL, specification TEXT NOT NULL,
              CHECK(registered_us < oos_start_us));
            CREATE TABLE IF NOT EXISTS results (
              id INTEGER PRIMARY KEY, experiment_id TEXT NOT NULL,
              recorded_us INTEGER NOT NULL, status TEXT NOT NULL,
              metrics TEXT NOT NULL, artifacts TEXT NOT NULL,
              FOREIGN KEY(experiment_id) REFERENCES experiments(id));
            CREATE UNIQUE INDEX IF NOT EXISTS one_final_result
              ON results(experiment_id) WHERE status = 'final';
            """
        )
        for table in ("signals", "observations", "experiments", "results"):
            self.db.executescript(
                f"""
                CREATE TRIGGER IF NOT EXISTS immutable_{table}_update
                BEFORE UPDATE ON {table} BEGIN
                  SELECT RAISE(ABORT, '{table} is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS immutable_{table}_delete
                BEFORE DELETE ON {table} BEGIN
                  SELECT RAISE(ABORT, '{table} is append-only');
                END;
                """
            )

    @staticmethod
    def _json(value):
        def normalize(item):
            if isinstance(item, dict):
                return {str(key): normalize(child) for key, child in item.items()}
            if isinstance(item, (list, tuple)):
                return [normalize(child) for child in item]
            if hasattr(item, "item"):
                item = item.item()
            if isinstance(item, float) and not math.isfinite(item):
                return None
            return item

        return json.dumps(normalize(value), sort_keys=True, allow_nan=False)

    @staticmethod
    def _required_text(**values):
        for name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required")

    def register_signal(self, identity, version, family, owner, definition):
        self._required_text(
            identity=identity, version=version, family=family, owner=owner
        )
        with self.db:
            self.db.execute(
                "INSERT INTO signals VALUES (?,?,?,?,?)",
                (identity, version, family, owner, self._json(definition)),
            )

    def observe(
        self, identity, version, market, instrument, event_us,
        available_us, dataset_version, payload
    ):
        if not all(type(value) is int for value in (event_us, available_us)):
            raise ValueError("timestamps must be UTC integer microseconds")
        self._required_text(
            market=market, instrument=instrument, dataset_version=dataset_version
        )
        with self.db:
            self.db.execute(
                """INSERT INTO observations
                (signal_id,version,market,instrument,event_us,available_us,
                 dataset_version,payload) VALUES (?,?,?,?,?,?,?,?)""",
                (
                    identity, version, market, instrument, event_us,
                    available_us, dataset_version, self._json(payload),
                ),
            )

    def available(self, decision_us, signal_id: Optional[str] = None):
        if type(decision_us) is not int:
            raise ValueError("decision_us must be an integer")
        sql = """SELECT signal_id,version,market,instrument,event_us,
          available_us,dataset_version,payload FROM observations
          WHERE available_us <= ?"""
        params = [decision_us]
        if signal_id is not None:
            sql += " AND signal_id = ?"
            params.append(signal_id)
        sql += " ORDER BY available_us,id"
        return self.db.execute(sql, params).fetchall()

    def preregister(self, identity, registered_us, oos_start_us, specification):
        required = {
            "signal_versions", "dataset_versions", "holding_seconds",
            "cost_model_version", "split_policy", "acceptance_rules",
        }
        missing = required - set(specification)
        if missing:
            raise ValueError(f"incomplete experiment specification: {sorted(missing)}")
        horizons = specification["holding_seconds"]
        if not horizons or any(type(value) is not int or value <= 0 for value in horizons):
            raise ValueError("positive integer holding horizons required")
        with self.db:
            self.db.execute(
                "INSERT INTO experiments VALUES (?,?,?,?)",
                (identity, registered_us, oos_start_us, self._json(specification)),
            )

    def experiment(self, identity) -> Optional[Dict[str, Any]]:
        row = self.db.execute(
            "SELECT registered_us,oos_start_us,specification FROM experiments WHERE id=?",
            (identity,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": identity,
            "registered_us": row[0],
            "oos_start_us": row[1],
            "specification": json.loads(row[2]),
        }

    def record_result(
        self, experiment_id, recorded_us, metrics,
        artifacts=None, status="final"
    ):
        if status not in {"interim", "final", "failed"}:
            raise ValueError("unknown result status")
        with self.db:
            self.db.execute(
                """INSERT INTO results
                (experiment_id,recorded_us,status,metrics,artifacts)
                VALUES (?,?,?,?,?)""",
                (
                    experiment_id, recorded_us, status, self._json(metrics),
                    self._json(artifacts or {}),
                ),
            )

    def close(self):
        self.db.close()
