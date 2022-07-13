#! /usr/bin/env node

const path = require("path")
const fs = require("fs")
const readline = require("readline")
const child_process = require("child_process")
const assert = require("assert")

const tqdm = require("tqdm")
const load = require("better-sqlite3")
const _ = require("lodash")

const DATA_DIR = path.resolve(`${__dirname}/../../data`)
const CHUNKS_DIR = `${DATA_DIR}/danbooru/raw/json`
const DB_PATH = `${DATA_DIR}/danbooru/artifacts/tags.sqlite`

console.log(CHUNKS_DIR)
console.log(DB_PATH)

async function lines(filePath, { db }) {
    const wcStdout = child_process.execSync(`wc -l < ${filePath}`).toString()
    const nLines = parseInt(wcStdout)
    assert(!isNaN(nLines), `Invalid wc output: ${wcStdout}`)

    const fstream = fs.createReadStream(filePath)
    const rdr = readline.createInterface({
        input: fstream,
        crlfDelay: Infinity,
    })

    let inserted = 0;
    let skipped =  0;

    // make it work w/ tqdm
    rdr[Symbol.toStringTag] = "AsyncGenerator";

    function makeBulkInsertStmt(n) {
        const bulkInsertParams = _.repeat("(?, ?),", n)
        // strip the last comma
        .slice(0, -1)
        return  db.prepare(`INSERT INTO tags (id, tag) VALUES ${bulkInsertParams}`)
    }

    const BULK_INSERT = 10_000
    const bulkInsertStmt = makeBulkInsertStmt(BULK_INSERT)
    let bulkInsertBuffer = []
    for await (const line of tqdm(rdr, { total: nLines })) {
        const entry = JSON.parse(line)
        const { id, tag_string_general } = entry;
        if (typeof id !== "string" || typeof tag_string_general !== "string" || !tag_string_general) {
            skipped += 1
            continue;
        }
        const int = parseInt(id)
        if (isNaN(int)) {
            skipped += 1
            continue
        }
        bulkInsertBuffer.push(id)
        bulkInsertBuffer.push(tag_string_general)
        if (bulkInsertBuffer.length === 2 * BULK_INSERT) {
            bulkInsertStmt.run(...bulkInsertBuffer)
            bulkInsertBuffer = []
        }
        inserted += 1
    }
    if (bulkInsertBuffer.length > 0) {
        assert(bulkInsertBuffer.length % 2 === 0, `Invalid length: ${bulkInsertBuffer.length}`)
        const stmt = makeBulkInsertStmt(bulkInsertBuffer.length / 2)
        stmt.run(...bulkInsertBuffer)
    }
    return { inserted, skipped }
}

async function main() {
    const dbExists = fs.existsSync(DB_PATH)
    if (dbExists) {
        throw Error(`DB already exists at path: ${DB_PATH}`)
    }
    // Use ".tables" in the Sqlite CLI to list all tables
    const db = load(DB_PATH)
    // https://github.com/WiseLibs/better-sqlite3/issues/124
    // > The de-facto answer to almost all performance questions in SQLite is to activate WAL mode:
    db.pragma('journal_mode = WAL');
    db.exec(`CREATE TABLE tags
        (
           id  INT NOT NULL,
           tag TEXT NOT NULL,
           PRIMARY KEY (id)
        );
    `)

    const files = fs.readdirSync(CHUNKS_DIR)
    console.log("Processing files:")
    console.log(files)

    for (const file of files) {
        console.log("Processing file: " + file)
        const { skipped, inserted } = await lines(path.join(CHUNKS_DIR, file), { db })
        console.log(`${inserted}/${inserted + skipped} entries inserted`)
    }
    // merge the WAL file back into the DB
    // https://stackoverflow.com/questions/19574286/how-to-merge-contents-of-sqlite-3-7-wal-file-into-main-database-file
    db.exec("VACUUM;")
}

main()
