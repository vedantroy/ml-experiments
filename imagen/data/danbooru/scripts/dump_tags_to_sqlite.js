const path = require("path")
const fs = require("fs")
const readline = require("readline")
const child_process = require("child_process")
const assert = require("assert")

const tqdm = require("tqdm")
const load = require("better-sqlite3")

const DB_PATH = "../data/db.sqlite"
const CHUNKS_DIR = "../data/chunks"

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

    const stmt = db.prepare("INSERT INTO tags (id, tag) VALUES (?, ?)")
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
        // SQLITE does no type-checking, so theoretically you could insert strings into the int column
        stmt.run(id, tag_string_general)
        inserted += 1
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
}

main()
