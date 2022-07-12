const fs = require("fs");
const path = require("path");
const assert = require("assert");

const OUT = "../out.json"
const OUT_NL = "../out.nl.json"
const CHUNKS_DIR = "../chunks";

// 1 terabyte
const TARGET_BYTES = 1e12;
let bytes = 0;
const ids = [];

const files = fs.readdirSync(CHUNKS_DIR);
outer: for (const file of files) {
  console.log(`Processing file: ${file}`);
  const text = fs.readFileSync(path.join(CHUNKS_DIR, file)).toString();
  const posts = text
    .split("\n")
    // Last line is blank
    .slice(0, -1)
    .map((l) => JSON.parse(l));

  for (const post of posts) {
    const size = parseInt(post.file_size);
    assert(!isNaN(size), `Invalid file size: ${post.file_size}`);
    total_images++;
    const { id, tag_string_general, file_ext } = post;
    const validFile = file_ext === "jpg" || file_ext === "png"
    if (typeof id !== "string" || typeof tag_string_general !== "string" || typeof file_ext !== "string" || !validFile) {
      invalid_images++
      //console.log(`INVALID: ${invalid_images}/${total_images}`)
      continue
    }
    bytes += size;
    if (bytes >= TARGET_BYTES) {
      break outer;
    } else {
      console.log(`${(bytes / TARGET_BYTES * 100).toFixed(2)}`);
    }
    ids.push({
        id, file_ext, tags: tag_string_general
    })
  }
}

let out = ""
for (const x of ids) {
    out += JSON.stringify(x)
    out += "\n"
}

fs.writeFileSync(OUT_NL, out)
