const path = require("path")
const fs = require("fs")
const tqdm = require("tqdm")
const { open } = require("lmdb")

const DB_PATH = "../data/db"
const TAGS_DIR = "../data/dataset/tags"

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

    if (!fs.existsSync(TAGS_DIR)){
        fs.mkdirSync(TAGS_DIR);
    }

    const items = rootDb.get(ENTRIES_KEY)
    for (const { key, value } of tqdm(rootDb.getRange(), { total: items })) {
        if (key === STRUCTURES_KEY || key === ENTRIES_KEY) continue;
        const { tag_string_general } = value;
        if (tag_string_general) {
            // Using the promises version causes JS to crash w/ OOM error
            fs.writeFileSync(path.join(TAGS_DIR, key), tag_string_general)
        }
    }
}

main()
