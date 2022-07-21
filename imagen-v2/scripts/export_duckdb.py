import duckdb

from fastargs import Param, Section
from fastargs.decorators import param
from utils import init_cli_args

# duckdb experiments
Section("files", "inputs, outputs, etc.").params(
    in_db=Param(
        str,
        "the duckdb with id => string[]",
        default="abr"
    ),
    #out_db=Param(str, "the sqlite database to write id => string"),
    #overwrite=Param(bool, "delete the destination db if it exists", default=False),
)

COPY_QUERY = """
INSERT INTO items2
SELECT id, concat_ws(' ', tags)
FROM items
"""

@param("files.in_db", "in_db")
def run(in_db):
    conn = duckdb.connect(in_db, read_only=False)
    tables = conn.execute("SHOW TABLES").fetchall()
    print(tables)
    conn.execute("CREATE TABLE items2(id INTEGER, tags VARCHAR, PRIMARY KEY(id));")
    conn.execute(COPY_QUERY)

init_cli_args("Merge tags in duckdb")
run()
