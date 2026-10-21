CREATE TABLE gui_snapshot(run_id TEXT PRIMARY KEY REFERENCES processing_run(id), schema TEXT NOT NULL, metadata TEXT NOT NULL, interpretation_id TEXT REFERENCES interpretation(id));
CREATE TABLE gui_snapshot_array(run_id TEXT NOT NULL REFERENCES gui_snapshot(run_id), name TEXT NOT NULL, array_id TEXT NOT NULL REFERENCES numeric_array(id), PRIMARY KEY(run_id,name));
CREATE TABLE legacy_attachment(id TEXT PRIMARY KEY, study_id TEXT NOT NULL REFERENCES study(id), name TEXT NOT NULL, sha256 TEXT NOT NULL, payload BLOB NOT NULL, fields TEXT NOT NULL, limitations TEXT NOT NULL);
