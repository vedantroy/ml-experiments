const fs = require("fs")

const tqdm = require("tqdm")
const { open, asBinary } = require("lmdb")

const DB_PATH = "../data/db"
const TAGS_DB_PATH = "../data/db_tags"

const STRUCTURES_KEY = Symbol.for("STRUCTURES")
const ENTRIES_KEY = "__length"

async function main() {
    const dbExists = fs.existsSync(DB_PATH)
    if (!dbExists) {
        throw Error(`DB does not exist at path: ${DB_PATH}`)
    }

    const tagsDbExists = fs.existsSync(TAGS_DB_PATH)
    if (tagsDbExists) {
        throw Error(`Tags DB already exists at path: ${TAGS_DB_PATH}`)
    }

    const rootDb =  open(DB_PATH, {
        sharedStructuresKey: Symbol.for("STRUCTURES")
    })

    const tagsDb =  open(TAGS_DB_PATH, {
        sharedStructuresKey: Symbol.for("STRUCTURES")
    })

    const items = rootDb.get(ENTRIES_KEY)
    for (const { key, value } of tqdm(rootDb.getRange(), { total: items })) {
        if (key === STRUCTURES_KEY || key === ENTRIES_KEY) continue;
        const { tag_string_general } = value;
        if (tag_string_general) {
            //tagsDb.put(key, asBinary(Buffer.from(tag_string_general, "utf8")))
            tagsDb.put(key, tag_string_general)
        }
    }
    tagsDb.put(ENTRIES_KEY, rootDb.get(ENTRIES_KEY))
}

main()
