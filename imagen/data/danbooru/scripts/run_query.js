const path = require("path")
const fs = require("fs")
const readline = require("readline")
const child_process = require("child_process")
const assert = require("assert")

const tqdm = require("tqdm")
const { open } = require("lmdb")

const DB_PATH = "../data/db"

const STRUCTURES_KEY = Symbol.for("STRUCTURES")
const ENTRIES_KEY = "__length"

async function main() {
    const dbExists = fs.existsSync(DB_PATH)
    if (!dbExists) {
        throw Error(`DB does not exist at path: ${DB_PATH}`)
    }

    const rootDb =  open(DB_PATH, {
        sharedStructuresKey: Symbol.for("STRUCTURES")
    })

    const items = rootDb.get(ENTRIES_KEY)
    for (const { key, value } of tqdm(rootDb.getRange(), { total: items })) {
        if (key === STRUCTURES_KEY || key === ENTRIES_KEY) continue;
    }
}

main()
