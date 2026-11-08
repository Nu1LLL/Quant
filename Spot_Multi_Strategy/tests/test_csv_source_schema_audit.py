"""Synthetic format checks; no market values or sealed parser are reopened."""

import unittest

from research.csv_source_schema_audit import SchemaAuditError, audit_csv_schema


class CsvSchemaAuditTests(unittest.TestCase):
    def test_empty_missing_cells_are_detected_before_a_dot_only_parser(self):
        body = (b"observation_date,RATE\n2020-01-01,1.23\n"
                b"2020-01-02,\n2020-01-03,.\n")
        result = audit_csv_schema(
            body, date_column="observation_date", numeric_columns=("RATE",),
            parser_missing_tokens=frozenset({"."}))
        self.assertEqual(result["column_shapes"]["RATE"],
                         {"numeric_lexeme": 1, "empty": 1, "dot": 1})
        self.assertEqual(result["incompatible_shapes"]["RATE"], {"empty": 1})
        self.assertFalse(result["parser_compatible"])
        self.assertNotIn("1.23", str(result))

    def test_compatible_missing_shapes_and_invalid_structure(self):
        body = b"date,value\n2020-01-01,\n2020-01-02,.\n2020-01-03,-1e-2\n"
        result = audit_csv_schema(body, date_column="date",
                                  numeric_columns=("value",),
                                  parser_missing_tokens=frozenset({"", "."}))
        self.assertTrue(result["parser_compatible"])
        for bad in (body.replace(b"2020-01-02", b"2020-01-01"),
                    body + b"2020-01-04,2,extra\n",
                    b"date,value\n2020-01-01,NaN\n"):
            if b"NaN" in bad:
                self.assertFalse(audit_csv_schema(
                    bad, date_column="date", numeric_columns=("value",))
                    ["parser_compatible"])
            else:
                with self.assertRaises(SchemaAuditError):
                    audit_csv_schema(bad, date_column="date",
                                     numeric_columns=("value",))


if __name__ == "__main__":
    unittest.main()
