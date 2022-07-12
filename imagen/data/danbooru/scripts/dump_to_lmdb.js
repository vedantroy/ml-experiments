const path = require("path")
const fs = require("fs")
const readline = require("readline")
const child_process = require("child_process")
const assert = require("assert")

const tqdm = require("tqdm")
const { open } = require("lmdb")

const DB_PATH = "../data/db"
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
    const tag = rdr[Symbol.toStringTag];

    for await (const line of tqdm(rdr, { total: nLines })) {
        const entry = JSON.parse(line)
        const { id } = entry;
        if (typeof id !== "string") {
            skipped += 1
            continue;
        }
        db.put(id, entry)
        inserted += 1
    }
    return { inserted, skipped }
}

async function main() {
    const dbExists = fs.existsSync(DB_PATH)
    if (dbExists) {
        throw Error(`DB already exists at path: ${DB_PATH}`)
    }
    const rootDb =  open(DB_PATH, {
        sharedStructuresKey: Symbol.for("STRUCTURES")
    })
    // TODO: Use a symbol for this as well
    const ENTRIES_KEY = "__length"

    const files = fs.readdirSync(CHUNKS_DIR)
    console.log("Processing files:")
    console.log(files)
    let allEntries = 0;
    for (const file of files) {
        console.log("Processing file: " + file)
        const { skipped, inserted } = await lines(path.join(CHUNKS_DIR, file), { db: rootDb })
        allEntries += inserted;
        console.log(`${inserted}/${inserted + skipped} entries inserted`)
    }
    if (rootDb.doesExist(ENTRIES_KEY)) {
        throw new Error(`Entries key: ${ENTRIES_KEY} is already set`)
    }
    rootDb.putSync(ENTRIES_KEY, allEntries)
}

main()
