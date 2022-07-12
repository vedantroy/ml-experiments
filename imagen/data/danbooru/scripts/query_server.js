#! /usr/bin/env node

const { open } = require("lmdb")
const fs = require("fs")
const readline = require("readline");
const path = require("path")

const rl = readline.createInterface({
  input: process.stdin,
  terminal: false,
});

const DB_PATH = path.normalize(`${__dirname}/../artifacts/lmdb/tags`)
const dbExists = fs.existsSync(DB_PATH);

if (!dbExists) {
  throw Error(`DB does not exist at path: ${DB_PATH}`);
}
const rootDb = open(DB_PATH, {
  sharedStructuresKey: Symbol.for("STRUCTURES"),
});

rl.on("line", line => {
    const value = rootDb.get(line)
    if (!value) {
        console.log(`MISSING_VALUE: ${line}`)
        return
    }
    console.log(value)
})
