CREATE TABLE IF NOT EXISTS universities (
  id               TEXT PRIMARY KEY,
  name             TEXT NOT NULL,
  url              TEXT,
  program_level    TEXT DEFAULT 'phd' CHECK(program_level IN ('phd','masters','undergrad')),
  acceptance_rate  REAL CHECK(acceptance_rate BETWEEN 0 AND 1),
  min_gpa          REAL CHECK(min_gpa BETWEEN 0 AND 4.0),
  min_gre_verbal   INTEGER CHECK(min_gre_verbal BETWEEN 130 AND 170),
  min_gre_quant    INTEGER CHECK(min_gre_quant BETWEEN 130 AND 170),
  requires_sat     INTEGER DEFAULT 0,
  min_sat          INTEGER CHECK(min_sat BETWEEN 400 AND 1600),
  ap_classes_req   INTEGER DEFAULT 0,
  lor_count        INTEGER DEFAULT 2,
  scraped_at       TEXT
);

CREATE TABLE IF NOT EXISTS scholarships (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  university_id    TEXT NOT NULL REFERENCES universities(id) ON DELETE CASCADE,
  name             TEXT NOT NULL,
  deadline         TEXT NOT NULL,
  amount_usd       INTEGER
);

CREATE TABLE IF NOT EXISTS scrape_log (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  university_id    TEXT NOT NULL,
  run_at           TEXT NOT NULL,
  status           TEXT NOT NULL CHECK(status IN ('success','partial','failed')),
  null_fields      TEXT DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_scholarships_university
  ON scholarships(university_id);

CREATE INDEX IF NOT EXISTS idx_scrape_log_university
  ON scrape_log(university_id, run_at);