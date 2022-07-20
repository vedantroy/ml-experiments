import argparse

import duckdb
from fastargs.decorators import param

from fastargs import Param, Section, get_current_config
Section("files", "inputs, outputs, etc.").params(
    duck_db=Param(
        str,
        "the duck db",
    ),
    stats_file=Param(
        str,
        "where to write stats"
    )
)

parser = argparse.ArgumentParser(
    description="Get database stats"
)
config = get_current_config()

config.augment_argparse(parser)
config.collect_argparse_args(parser)

config.validate(mode="stderr")
config.summary()

query = """
WITH lists_split_into_rows AS (
    SELECT 
      id,
      unnest(tags) AS new_column
    FROM items
)
SELECT 
  new_column,
  count(*) AS my_count
FROM lists_split_into_rows
GROUP BY
  new_column
"""

@param("files.duck_db")
@param("files.stats_file")
def run(duck_db, stats_file):
    conn = duckdb.connect(duck_db, read_only=True)
    conn.execute(query)
    results = conn.fetchall()

    results.sort(key=lambda x: x[1], reverse=True)
    with open(stats_file, "w+") as f:
        for tag, occ in results:
            f.write(f"{tag}: {occ}\n")

run()